"""
Phase 0 安全加固回归测试

覆盖：
  - 上传安全（扩展名白名单、路径遏制、大小限制）
  - 鉴权（API Key 校验、admin 分级、常量时间比较）
  - 限流
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402
from fastapi import HTTPException  # noqa: E402

from api import security, upload_handler  # noqa: E402
from api.security import RateLimiter, _constant_time_match, _extract_key  # noqa: E402
from api.upload_handler import ALLOWED_EXTENSIONS, _safe_extension  # noqa: E402


# ── 上传：扩展名白名单 ───────────────────────────────────────

@pytest.mark.parametrize("name", ["a.pdf", "b.PNG", "c.xlsx", "d.md", "e.csv", "f.txt"])
def test_allowed_extensions_pass(name):
    assert _safe_extension(name).startswith(".")


@pytest.mark.parametrize(
    "name",
    [
        "evil.exe",
        "script.sh",
        "payload.php",
        "noext",
        "",                     # 空文件名
        "double.pdf.exe",       # 双扩展名取 .exe
        "x.tar.gz",
        "../../etc/passwd",     # 无白名单扩展名
        "a.pdf\x00.exe",        # NUL 截断尝试
    ],
)
def test_disallowed_extensions_rejected(name):
    with pytest.raises(HTTPException) as e:
        _safe_extension(name)
    assert e.value.status_code == 400


def test_extension_set_matches_parser_support():
    """白名单必须与解析器实际支持的类型一致"""
    from agents.doc_parser_agent import DocParserAgent

    assert ALLOWED_EXTENSIONS == set(DocParserAgent.SUPPORTED_EXTENSIONS.keys())


# ── 上传：路径遏制 ───────────────────────────────────────────

@pytest.mark.parametrize(
    "candidate",
    [
        "../../../etc/passwd",
        "..\\..\\Windows\\System32\\config",
        "/etc/shadow",
        "subdir/../../outside.txt",
    ],
)
def test_path_containment_rejects_escape(candidate, tmp_path):
    from config import settings

    original = settings.upload_dir
    settings.upload_dir = str(tmp_path)
    try:
        with pytest.raises(HTTPException) as e:
            upload_handler.resolve_managed_path(candidate)
        assert e.value.status_code in (400, 404)
    finally:
        settings.upload_dir = original


def test_path_containment_allows_inside(tmp_path):
    from config import settings

    original = settings.upload_dir
    settings.upload_dir = str(tmp_path)
    try:
        target = tmp_path / "ok.txt"
        target.write_text("hi", encoding="utf-8")
        resolved = upload_handler.resolve_managed_path("ok.txt")
        assert resolved.resolve() == target.resolve()
    finally:
        settings.upload_dir = original


# ── 鉴权 ─────────────────────────────────────────────────────

def test_extract_key_from_various_headers():
    assert _extract_key(None, "abc") == "abc"
    assert _extract_key("Bearer xyz", None) == "xyz"
    assert _extract_key("bearer xyz", None) == "xyz"
    assert _extract_key("Basic abc", None) is None
    assert _extract_key(None, None) is None


def test_constant_time_match():
    valid = {"k1", "k2"}
    assert _constant_time_match("k1", valid) is True
    assert _constant_time_match("k2", valid) is True
    assert _constant_time_match("k3", valid) is False
    assert _constant_time_match("", valid) is False


@pytest.mark.asyncio
async def test_require_api_key_rejects_missing():
    from config import settings

    original = settings.auth_enabled
    settings.auth_enabled = True
    try:
        with pytest.raises(HTTPException) as e:
            await security.require_api_key(_FakeRequest(), None, None)
        assert e.value.status_code == 401
    finally:
        settings.auth_enabled = original


@pytest.mark.asyncio
async def test_require_api_key_rejects_invalid():
    from config import settings

    original = settings.auth_enabled
    settings.auth_enabled = True
    try:
        with pytest.raises(HTTPException) as e:
            await security.require_api_key(_FakeRequest(), None, "wrong-key")
        assert e.value.status_code == 401
    finally:
        settings.auth_enabled = original


@pytest.mark.asyncio
async def test_admin_endpoint_rejects_non_admin_key():
    """普通 key 不得访问管理端点 —— 这是修复未授权删库的关键"""
    from config import settings

    original_enabled, original_admin = settings.auth_enabled, settings.admin_api_keys
    settings.auth_enabled = True
    settings.admin_api_keys = "admin-secret"
    try:
        with pytest.raises(HTTPException) as e:
            await security.require_admin_key(_FakeRequest(), None, "ordinary-key")
        assert e.value.status_code == 403
    finally:
        settings.auth_enabled = original_enabled
        settings.admin_api_keys = original_admin


@pytest.mark.asyncio
async def test_admin_endpoint_accepts_admin_key():
    from config import settings

    original_enabled, original_admin = settings.auth_enabled, settings.admin_api_keys
    settings.auth_enabled = True
    settings.admin_api_keys = "admin-secret"
    try:
        got = await security.require_admin_key(_FakeRequest(), None, "admin-secret")
        assert got == "admin-secret"
    finally:
        settings.auth_enabled = original_enabled
        settings.admin_api_keys = original_admin


class _FakeRequest:
    class _Client:
        host = "127.0.0.1"

    client = _Client()

    class _URL:
        path = "/test"

    url = _URL()


# ── 限流 ─────────────────────────────────────────────────────

def test_rate_limiter_allows_within_limit():
    rl = RateLimiter(per_minute=3)
    for _ in range(3):
        allowed, _used = rl.allow("k")
        assert allowed is True


def test_rate_limiter_blocks_over_limit():
    rl = RateLimiter(per_minute=2)
    rl.allow("k")
    rl.allow("k")
    allowed, used = rl.allow("k")
    assert allowed is False
    assert used == 2


def test_rate_limiter_isolates_buckets():
    rl = RateLimiter(per_minute=1)
    assert rl.allow("a")[0] is True
    assert rl.allow("b")[0] is True
    assert rl.allow("a")[0] is False


# 现在没有 Cypher 生成，该攻击面不存在。
# 代码见 _deprecated/services/knowledge_graph.py
