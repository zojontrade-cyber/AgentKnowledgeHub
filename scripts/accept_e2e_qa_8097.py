"""端到端验收：通过运行中的 API 提问，打印真实 UTF-8 文本。

端口可用 --port 覆盖（默认 8080）。
"""
import json
import os
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

PORT = os.getenv("API_PORT", "8080")
URL = f"http://127.0.0.1:{PORT}/api/qa/ask"
KEY = "dev-key-1"

QUESTIONS = [
    "年假有多少天",
    "请假要走什么流程",
    "出差住宿费能报多少",
    "保密级别分几级",
    "公司的标准工作时间是什么时候到什么时候",
]


def ask(q: str) -> dict:
    body = json.dumps({"question": q, "top_k": 5}).encode("utf-8")
    req = urllib.request.Request(
        URL, data=body,
        headers={"Content-Type": "application/json; charset=utf-8",
                 "X-API-Key": KEY},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


for q in QUESTIONS:
    print("=" * 78)
    print("Q:", q)
    try:
        r = ask(q)
    except Exception as e:
        print("  ERR:", e)
        print()
        continue
    ans = (r.get("answer") or "").strip()
    print("A:", ans[:300] if ans else "(空)")
    srcs = r.get("sources") or []
    print(f"  来源 {len(srcs)} 条:")
    for s in srcs[:5]:
        src = str(s.get("source", ""))
        score = s.get("score")
        head = (s.get("content") or "").split("\n")[0][:70]
        try:
            sc = f"{float(score):.3f}"
        except Exception:
            sc = str(score)
        print(f"    [{sc}] {src.split(chr(92))[-1][:40]}")
        print(f"           {head}")
    print()
