"""验证生产环境 fail-fast 配置校验（出口标准之一）"""
import os
import subprocess
import sys

CASES = [
    (
        "prod 用默认密码 + 空 API Key -> 必须拒绝启动",
        {"ENVIRONMENT": "prod", "OPENAI_API_KEY": "", "NEO4J_PASSWORD": "password", "API_KEYS": "", "ADMIN_API_KEYS": ""},
        False,
    ),
    (
        "prod 弱密码 password -> 必须拒绝",
        {"ENVIRONMENT": "prod", "OPENAI_API_KEY": "sk-real-key-123", "NEO4J_PASSWORD": "password", "API_KEYS": "k1", "ADMIN_API_KEYS": "a1"},
        False,
    ),
    (
        "prod 有 auth 但 API_KEYS 为空 -> 必须拒绝",
        {"ENVIRONMENT": "prod", "OPENAI_API_KEY": "sk-real-key-123", "NEO4J_PASSWORD": "strong-pw-xyz", "API_KEYS": "", "ADMIN_API_KEYS": "a1"},
        False,
    ),
    (
        "prod 配置齐全 -> 必须启动成功",
        {"ENVIRONMENT": "prod", "OPENAI_API_KEY": "sk-real-key-123", "NEO4J_PASSWORD": "strong-pw-xyz", "API_KEYS": "k1", "ADMIN_API_KEYS": "a1"},
        True,
    ),
    (
        "dev 允许弱默认值（本地开发）-> 可启动",
        {"ENVIRONMENT": "dev", "OPENAI_API_KEY": "", "NEO4J_PASSWORD": "password", "API_KEYS": "", "ADMIN_API_KEYS": ""},
        True,
    ),
]

SCRIPT = "import sys; sys.path.insert(0, '.'); from config import settings; print('LOADED')"
failed = 0

for label, env, should_load in CASES:
    full_env = {**os.environ, **env}
    # 清掉可能来自测试进程的污染
    for k in ("VECTOR_STORE_TYPE",):
        full_env.pop(k, None)
    r = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        capture_output=True, text=True, env=full_env, timeout=60,
    )
    loaded = "LOADED" in r.stdout
    ok = loaded == should_load
    if not ok:
        failed += 1
    print(f"[{'PASS' if ok else 'FAIL'}] {label}")
    print(f"        loaded={loaded} expected={should_load}")
    if not ok:
        err = (r.stderr or "").strip().splitlines()
        print("        stderr tail:", err[-1] if err else "(none)")

print()
print("FAILURES:", failed)
sys.exit(1 if failed else 0)
