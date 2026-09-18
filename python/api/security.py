"""
鉴权与限流中间件

背景
----
重构前 6 个端点**零鉴权**，且跨三语言实现全量搜索
`jwt|oauth|authenticat|authoriz|rate.?limit` **命中 0 条**。这意味任何能访问
端口的人都可以：
  - 上传文档（投毒 + 烧 LLM 费用）
  - 调 /api/admin/update 传任意 file_path 触发删除（第二条删库路径）
  - 匿名提问消耗配额

本模块提供最小可用的两道防线：
  1. API Key 鉴权（普通 key 与 admin key 分级）
  2. 基于滑动窗口的内存限流

设计取舍
--------
- API Key 而非 OAuth2：这是内网服务的合理起点，接入企业 SSO 时可替换本
  模块的依赖注入函数而不动端点代码。
- 内存限流：单副本够用。多副本需换 Redis 实现（Phase 2），接口已隔离在
  RateLimiter 类内。
- 用 hmac.compare_digest 做常量时间比较，避免时序侧信道。
"""

from __future__ import annotations

import hmac
import logging
import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, Request, status

from config import settings

logger = logging.getLogger(__name__)


# ── API Key 鉴权 ─────────────────────────────────────────────

def _constant_time_match(candidate: str, valid: set[str]) -> bool:
    """常量时间比较，避免通过响应时间推断 key"""
    ok = False
    for k in valid:
        if hmac.compare_digest(candidate, k):
            ok = True
    return ok


def _extract_key(authorization: str | None, x_api_key: str | None) -> str | None:
    """支持 `Authorization: Bearer <key>` 或 `X-API-Key: <key>`"""
    if x_api_key:
        return x_api_key.strip()
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None


async def require_api_key(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> str:
    """
    普通端点鉴权依赖。

    当 AUTH_ENABLED=false 时放行（本地开发），但会记录警告。
    生产环境由 Settings 校验保证 api_keys 非空。
    """
    if not settings.auth_enabled:
        logger.warning("鉴权已关闭（AUTH_ENABLED=false），端点处于开放状态")
        return "anonymous"

    key = _extract_key(authorization, x_api_key)
    if not key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少 API Key（请通过 X-API-Key 或 Authorization: Bearer 提供）",
            headers={"WWW-Authenticate": "Bearer"},
        )

    valid = settings.api_key_set | settings.admin_key_set
    if not _constant_time_match(key, valid):
        logger.warning("无效 API Key 尝试", extra={"extra_fields": {"path": request.url.path}})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="API Key 无效"
        )
    return key


async def require_admin_key(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> str:
    """
    管理员端点鉴权依赖（/api/admin/*）。

    这是修复「未授权删库」的关键：/api/admin/update 接受任意 file_path，
    必须收权到 admin。
    """
    if not settings.auth_enabled:
        logger.warning("鉴权已关闭，/admin 端点处于开放状态，切勿用于生产")
        return "anonymous"

    key = _extract_key(authorization, x_api_key)
    if not key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="管理端点需要管理员 API Key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not _constant_time_match(key, settings.admin_key_set):
        logger.warning(
            "非管理员访问管理端点",
            extra={"extra_fields": {"path": request.url.path}},
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="需要管理员权限"
        )
    return key


# ── 数据级权限 ───────────────────────────────────────────────

def scopes_for_key(key: str) -> list[str]:
    """
    返回该 API Key 可访问的数据范围（ACL）。

    规则（最小权限原则）：
      - 管理员 Key       -> ["*"]（不限制）
      - 显式配置了 scopes -> 使用配置值
      - 其他 Key         -> ["public"]（只能看公开文档）

    这样即使持有合法 Key，也无法检索到超出授权的文档
    （如 HR Key 看不到财务文档）。
    """
    if not key:
        return ["public"]
    if _constant_time_match(key, settings.admin_key_set):
        return ["*"]
    m = settings.key_scope_map
    if key in m:
        return m[key]
    return ["public"]


async def require_api_key_with_scope(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> tuple[str, list[str]]:
    """鉴权并返回 (key, scopes)，供需要数据级过滤的接口使用"""
    key = await require_api_key(request, authorization, x_api_key)
    return key, scopes_for_key(key)


def key_fingerprint(key: str) -> str:
    """
    API Key 指纹：只保留哈希前缀，用于审计、归属与用量隔离，避免明文入库。

    放在 security 层（而不是 api/main.py）是因为用量统计需要按 actor 隔离：
    ui_routes 也要用它，若从 main 反向导入会形成循环依赖。
    """
    import hashlib

    return "key_" + hashlib.sha256((key or "").encode()).hexdigest()[:12]


# ── 限流 ─────────────────────────────────────────────────────

class RateLimiter:
    """
    滑动窗口限流器（内存实现）。

    多副本部署时需替换为 Redis 版本；调用方只依赖 allow() 接口，
    因此替换不影响端点代码。
    """

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> tuple[bool, int]:
        """返回 (是否放行, 当前窗口内已用次数)"""
        now = time.monotonic()
        window_start = now - 60.0
        bucket = self._hits[key]

        while bucket and bucket[0] < window_start:
            bucket.popleft()

        if len(bucket) >= self.per_minute:
            return False, len(bucket)

        bucket.append(now)
        return True, len(bucket)

    def reset(self) -> None:
        self._hits.clear()


rate_limiter = RateLimiter(settings.rate_limit_per_minute)


async def enforce_rate_limit(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> None:
    """
    限流依赖。以 API Key 为维度；无 key 时退化为按客户端 IP。

    注意：这个依赖不负责鉴权（鉴权是 require_api_key 的职责），只做配额。
    """
    key = _extract_key(authorization, x_api_key)
    bucket_key = f"key:{key}" if key else f"ip:{request.client.host if request.client else 'unknown'}"

    allowed, used = rate_limiter.allow(bucket_key)
    if not allowed:
        logger.warning(
            "触发限流",
            extra={"extra_fields": {"bucket": bucket_key, "used": used}},
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"请求过于频繁，限制 {settings.rate_limit_per_minute} 次/分钟",
            headers={"Retry-After": "60"},
        )
