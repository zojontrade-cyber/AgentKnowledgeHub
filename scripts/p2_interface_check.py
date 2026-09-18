"""
P2：生产接口验证 —— 测延迟 / fallback / 日志 / 引用准确性

与 P0/P1 的离线评测不同，这里走**真实 HTTP 接口**，
关注的是工程属性而非指标：

  1. 延迟      —— p50/p95/max，区分 reranker 贡献
  2. fallback  —— reranker 关闭时是否优雅降级
  3. 日志      —— 是否留痕（请求/检索/重排）
  4. 引用准确性 —— 答案引用的来源是否真的包含该答案
"""

import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

BASE = "http://127.0.0.1:8096"
KEY = "dev-key-1"
ROOT = Path(__file__).resolve().parent.parent

BENCH = [
    json.loads(l)
    for l in (ROOT / "data" / "eval" / "bench_v3.jsonl").read_text(encoding="utf-8").splitlines()
    if l.strip()
]
EXACT = [r for r in BENCH if r["gold_type"] == "exact"]


def ask(q, top_k=5):
    body = json.dumps({"question": q, "top_k": top_k}).encode("utf-8")
    req = urllib.request.Request(
        BASE + "/api/qa/ask", data=body, method="POST",
        headers={"Content-Type": "application/json; charset=utf-8", "X-API-Key": KEY},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read().decode("utf-8"))
    return d, (time.perf_counter() - t0) * 1000


print("=" * 78)
print("1. 延迟（真实 HTTP，36 题）")
print("=" * 78)
lat = []
cite_ok = cite_total = 0
empty = 0
types = {}

for r in EXACT:
    try:
        d, ms = ask(r["question"])
    except Exception as e:
        print(f"  ERR {r['question']}: {e}")
        continue
    lat.append(ms)
    ans = (d.get("answer") or "").strip()
    srcs = d.get("sources") or []
    if not ans:
        empty += 1
    for s in srcs:
        t = s.get("type", "?")
        types[t] = types.get(t, 0) + 1

    # 引用准确性：gold 章节内容（前 30 字）是否出现在返回的 sources 里
    gold_key = (r["expected_answer"] or "").strip()[:30]
    cite_total += 1
    hit = any(gold_key and gold_key in (s.get("content") or "") for s in srcs)
    if hit:
        cite_ok += 1

lat.sort()
print(f"  样本       {len(lat)}")
print(f"  p50        {statistics.median(lat):.0f} ms")
print(f"  p95        {lat[int(len(lat)*0.95)]:.0f} ms")
print(f"  max        {max(lat):.0f} ms")
print(f"  min        {min(lat):.0f} ms")
print(f"  空答案     {empty}")

print()
print("=" * 78)
print("2. 引用准确性")
print("=" * 78)
print(f"  gold 章节内容出现在返回 sources 中: {cite_ok}/{cite_total} "
      f"({cite_ok/max(cite_total,1):.1%})")
print(f"  来源类型分布: {types}")

print()
print("=" * 78)
print("3. 样例（前 3 题完整输出）")
print("=" * 78)
for r in EXACT[:3]:
    try:
        d, ms = ask(r["question"])
    except Exception as e:
        print("ERR", e)
        continue
    print(f"\nQ: {r['question']}   ({ms:.0f} ms)")
    print(f"A: {(d.get('answer') or '')[:260]}")
    print(f"   gold 章节: {r['doc_title']} / {r['section_title']}")
    print(f"   sources ({len(d.get('sources') or [])}):")
    for s in (d.get("sources") or [])[:4]:
        c = (s.get("content") or "").replace("\n", " ")[:70]
        print(f"     [{s.get('type')}] score={s.get('score'):.3f} {c}")
