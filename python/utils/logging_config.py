"""
统一日志配置 —— 结构化日志 + 请求上下文

为什么需要这个
--------------
重构前 `python/` 下**没有任何 `import logging`**，全项目靠 print 和静默
`except: pass`。后果是：依赖挂掉、检索失败、LLM 报错全部无痕，生产事故
无法定位。

本模块提供：
  1. configure_logging()  —— 启动时调用一次，统一格式与级别
  2. request_id 上下文    —— 每个请求一个 id，串起该请求的所有日志
  3. JSON 或人类可读两种格式（由 LOG_FORMAT 控制）

用法：
    from utils.logging_config import configure_logging, set_request_id
    configure_logging()
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextvars import ContextVar

# 当前请求 id —— 用 ContextVar 保证 asyncio 并发下互不干扰
_request_id: ContextVar[str] = ContextVar("request_id", default="-")


def set_request_id(rid: str | None = None) -> str:
    """设置当前上下文请求 id，返回最终使用的值"""
    rid = rid or uuid.uuid4().hex[:12]
    _request_id.set(rid)
    return rid


def get_request_id() -> str:
    return _request_id.get()


class _ContextFilter(logging.Filter):
    """把 request_id 注入每条日志记录"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        return True


class _JsonFormatter(logging.Formatter):
    """JSON 格式 —— 生产环境便于采集到 ELK/Loki"""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # 附带自定义字段（如 latency_ms、endpoint）
        for k, v in getattr(record, "extra_fields", {}).items():
            payload[k] = v
        return json.dumps(payload, ensure_ascii=False)


class _TextFormatter(logging.Formatter):
    """人类可读格式 —— 本地开发用"""

    def format(self, record: logging.LogRecord) -> str:
        rid = getattr(record, "request_id", "-")
        base = (
            f"{time.strftime('%H:%M:%S', time.localtime(record.created))} "
            f"[{record.levelname:<5}] [{rid}] {record.name}: {record.getMessage()}"
        )
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def configure_logging(level: str | None = None) -> None:
    """
    配置根 logger。幂等 —— 重复调用不会叠加 handler。

    环境变量：
      LOG_LEVEL   默认 INFO
      LOG_FORMAT  默认 text；设为 json 输出结构化日志
    """
    level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    fmt = os.getenv("LOG_FORMAT", "text").lower()

    root = logging.getLogger()
    # 幂等：清掉已有 handler，避免 uvicorn reload 时重复输出
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_ContextFilter())
    handler.setFormatter(_JsonFormatter() if fmt == "json" else _TextFormatter())

    root.addHandler(handler)
    root.setLevel(level)

    # 压掉第三方库的噪音，但保留警告以上
    for noisy in ("httpx", "httpcore", "neo4j", "chromadb", "urllib3", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def log_extra(**kwargs) -> dict:
    """构造结构化附加字段：logger.info('msg', extra=log_extra(latency_ms=12))"""
    return {"extra_fields": kwargs}
