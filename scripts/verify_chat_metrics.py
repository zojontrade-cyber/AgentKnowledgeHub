"""端到端验证：/api/qa/ask 的计量字段 + /api/ui/qa-session 回填

验证的是**真实 HTTP 路径**（不是直接调 agent），因为计量写库在编排节点里。
用法：先启动 API，再 python scripts/verify_chat_metrics.py [--url http://127.0.0.1:8080]
"""

import argparse
import json
import sys
import uuid

import httpx

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8080")
ap.add_argument("--key", default="dev-key-1")
args = ap.parse_args()

URL = args.url.rstrip("/")
H = {"X-API-Key": args.key}
FAILS = []

# trust_env=False 是必须的：本机注册表里配置了系统代理，httpx 默认会读取，
# 于是连 127.0.0.1 都走代理 -> 全部 ReadTimeout（实测 20s 挂死）。
# 这不是服务端问题，服务端在 6.8s 内正常返回 200。
CLI = httpx.Client(timeout=300, trust_env=False)


def check(label, ok, extra=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {extra}" if extra else ""))
    if not ok:
        FAILS.append(label)


print("=" * 78)
print(f"目标 {URL}")
print("=" * 78)
print("1. 匿名单次问答（不传 session_id）")
print("=" * 78)
r = CLI.post(f"{URL}/api/qa/ask", headers=H,
             json={"question": "标准工作时间是几点到几点？"})
check("HTTP 200", r.status_code == 200, str(r.status_code))
if r.status_code != 200:
    print(r.text[:600])
    sys.exit(1)
d = r.json()
u, s = d.get("usage") or {}, d.get("session") or {}
print(f"     session_id={d.get('session_id')!r}  turn={d.get('turn')}  "
      f"trace_id={d.get('trace_id')!r}  degraded={d.get('metrics_degraded')}")
print(f"     usage: calls={u.get('llm_calls')} prompt={u.get('prompt_tokens')} "
      f"completion={u.get('completion_tokens')} total={u.get('total_tokens')}")
print(f"            hit={u.get('cache_hit_tokens')} miss={u.get('cache_miss_tokens')} "
      f"rate={u.get('cache_hit_rate')} supported={u.get('cache_supported')} "
      f"measured={u.get('cache_measured_calls')} hit_calls={u.get('cache_hit_calls')}")
print(f"            latency={u.get('latency_ms')}ms")
print(f"     session: {json.dumps(s, ensure_ascii=False)}")
for c in u.get("per_call") or []:
    print(f"       #{c['call_index']} {c['stage']:<10} model={c['model']:<16} "
          f"prompt={c['prompt_tokens']:<6} completion={c['completion_tokens']:<5} "
          f"hit={c['cache_hit_tokens']:<6} miss={c['cache_miss_tokens']:<6} "
          f"supported={c['cache_supported']}")

check("匿名请求不交出 session_id（防其变成真 session）", d.get("session_id") is None)
check("匿名请求的 session 汇总里也不含服务端生成的 id",
      not (d.get("session") or {}).get("session_id"),
      repr((d.get("session") or {}).get("session_id")))
check("匿名请求 turn=1", d.get("turn") == 1)
check("trace_id 非空（与日志 request_id 同源）", bool(d.get("trace_id")))
check("llm_calls=3（intent/rewrite/generate）", u.get("llm_calls") == 3, str(u.get("llm_calls")))
check("stage 归因正确",
      [c["stage"] for c in (u.get("per_call") or [])] == ["intent", "rewrite", "generate"],
      str([c["stage"] for c in (u.get("per_call") or [])]))
check("total == prompt + completion",
      u.get("total_tokens") == (u.get("prompt_tokens") or 0) + (u.get("completion_tokens") or 0))
check("缓存可测（DeepSeek 报 cache 字段）", u.get("cache_supported") is True)
check("每次可测调用 hit+miss == prompt",
      all((c["cache_hit_tokens"] + c["cache_miss_tokens"]) == c["prompt_tokens"]
          for c in (u.get("per_call") or []) if c["cache_supported"]))
check("未降级", d.get("metrics_degraded") is False)

print()
print("=" * 78)
print("2. 会话累计（带客户端自有 session_id）")
print("=" * 78)
sid = uuid.uuid4().hex[:16]
turns = []
d = {}
for i, q in enumerate(["员工每年应完成多少学时的培训？", "年假最多有多少天？"], 1):
    r = CLI.post(f"{URL}/api/qa/ask", headers=H,
                 json={"question": q, "session_id": sid})
    d = r.json()
    turns.append(d.get("turn"))
    s = d.get("session") or {}
    print(f"     第 {i} 次: session_id={d.get('session_id')!r} turn={d.get('turn')} "
          f"session.turns={s.get('turns')} session.total={s.get('total_tokens')} "
          f"rate={s.get('cache_hit_rate')}")

check("轮次由服务端递增 1,2", turns == [1, 2], str(turns))
check("回传客户端自有的 session_id", d.get("session_id") == sid)
check("会话累计 turns=2", (d.get("session") or {}).get("turns") == 2)
check("会话累计 tokens > 单轮 tokens",
      (d.get("session") or {}).get("total_tokens", 0)
      > (d.get("usage") or {}).get("total_tokens", 0))

print()
print("=" * 78)
print("3. /api/ui/qa-session 回填（刷新页面用）")
print("=" * 78)
r = CLI.get(f"{URL}/api/ui/qa-session", headers=H,
            params={"session_id": sid, "limit": 10})
check("HTTP 200", r.status_code == 200, str(r.status_code))
d = r.json()
print(f"     summary: {json.dumps(d.get('summary'), ensure_ascii=False)}")
for t in d.get("turns") or []:
    print(f"     turn {t['turn']}: tokens={t['total_tokens']} "
          f"hit={t['cache_hit_tokens']} miss={t['cache_miss_tokens']} "
          f"latency={t['latency_ms']}ms")
check("summary.turns=2", (d.get("summary") or {}).get("turns") == 2)
check("明细 2 轮", len(d.get("turns") or []) == 2)
check("明细轮次正序", [t["turn"] for t in d.get("turns") or []] == [1, 2])

print()
print("=" * 78)
print("4. 隔离：别的 Key 读不到该会话")
print("=" * 78)
r = CLI.get(f"{URL}/api/ui/qa-session", headers={"X-API-Key": "dev-admin-key-1"},
            params={"session_id": sid})
other = (r.json().get("summary") or {}) if r.status_code == 200 else {}
print(f"     另一 Key 看到 turns={other.get('turns')} total={other.get('total_tokens')}")
check("不同 actor 读不到（turns=0）", other.get("turns") == 0)

print()
print("=" * 78)
print("5. 非法 session_id 视为未提供")
print("=" * 78)
r = CLI.post(f"{URL}/api/qa/ask", headers=H,
             json={"question": "工资每月几号发放？", "session_id": "bad id with spaces"})
d = r.json()
print(f"     session_id={d.get('session_id')!r} turn={d.get('turn')}")
check("非法 id 不建立会话（session_id 为 null）", d.get("session_id") is None)

print()
print("=" * 78)
print(f"结果：{'全部通过' if not FAILS else '失败 ' + str(FAILS)}")
print("=" * 78)
sys.exit(1 if FAILS else 0)
