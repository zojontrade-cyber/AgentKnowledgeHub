"""
P2 实验：BM25 top20 → reranker → top5

背景（P1 已确立）
-----------------
在 129 个可召回章节上：
    BM25 flat   Chunk@1 76.3%   Chunk@5 97.4%
即：**召回已基本解决（97.4%），瓶颈是排序（76.3%）**。
这正是 reranker 的作用区间。

实验设计（用户指定）
--------------------
不要比较 BM25 vs Hybrid。应比较：

  A  BM25 Top10（无 reranker）           —— baseline
  B  BM25 Top20 → reranker → Top5        —— 精排

指标：Chunk@1 / Chunk@5 / MRR / Latency / 调用成本
预测（用户）：Top1 提升明显；Recall@5 提升有限；MRR 提升明显。

只读实验 —— 不改生产代码，不修改索引。
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

_PY = Path(Path(__file__).resolve().parent.parent / "python")
os.chdir(_PY)
sys.path.insert(0, str(_PY))
sys.stdout.reconfigure(encoding="utf-8")

from agents.doc_parser_agent import DocParserAgent
from services.bm25 import BM25Index
from services.reranker import reranker

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(
    r"C:\Users\滨滨\Desktop\Knowledge-Base-Intelligent-Q-A-main\data\clean\institutional"
)
BENCH = ROOT / "data" / "eval" / "bench_v3.jsonl"

POOL = 50      # 粗召回池
TOPK = 5       # 最终条数
CONCURRENCY = 4


def build():
    parser = DocParserAgent()
    chapters = []
    for f in sorted(SRC.glob("*.md")):
        dt, secs = parser.parse_tree(
            parser.strip_metadata(f.read_text(encoding="utf-8", errors="ignore"))
        )
        if not dt:
            continue
        for i, sec in enumerate(secs):
            st, body = sec.title, sec.body
            if not body:
                continue
            if DocParserAgent.classify_section(st, body) == "doc_overview":
                continue      # 与生产一致：排除文档说明
            chapters.append({
                "doc": dt, "idx": i, "sec": st, "body": body,
                "text": f"{dt}\n章节：{st}\n{body}",
            })
    return chapters


def m(ranks):
    n = len(ranks)
    if not n:
        return (0.0, 0.0, 0.0)
    return (
        sum(1 for r in ranks if r == 1) / n,
        sum(1 for r in ranks if r <= 5) / n,
        sum(1 / r for r in ranks if r < 999) / n,
    )


async def main():
    chapters = build()
    idx = BM25Index()
    idx.build([c["text"] for c in chapters], [str(i) for i in range(len(chapters))])
    print(f"可召回章节: {len(chapters)}")

    bench = [json.loads(l) for l in BENCH.read_text(encoding="utf-8").splitlines() if l.strip()]
    exact = [r for r in bench if r["gold_type"] == "exact"]
    print(f"评测: exact {len(exact)} 题")
    print()

    sem = asyncio.Semaphore(CONCURRENCY)

    async def run_one(r):
        q = r["question"]
        gdoc, gidx = r["doc_title"], r["gold_chunk_index"]

        # ── A: BM25 top10（无 reranker）────────────────
        t0 = time.perf_counter()
        hits = idx.rank_ids(q, top_k=POOL)
        a_list = [(chapters[int(sid)], sc) for sid, sc in hits[:TOPK]]
        a_ms = (time.perf_counter() - t0) * 1000

        a_rank = 999
        for rank, (c, _) in enumerate(a_list, 1):
            if c["doc"] == gdoc and c["idx"] == gidx:
                a_rank = rank
                break

        # ── B: top20 → reranker → top5 ─────────────────
        pool = [(chapters[int(sid)], sc) for sid, sc in hits]
        async with sem:
            t1 = time.perf_counter()
            rr = await reranker.rerank(q, [c["text"] for c, _ in pool], top_k=TOPK)
            b_ms = (time.perf_counter() - t1) * 1000

        if rr:
            b_list = [pool[i] for i, _ in rr]
        else:
            b_list = pool[:TOPK]

        b_rank = 999
        for rank, (c, _) in enumerate(b_list, 1):
            if c["doc"] == gdoc and c["idx"] == gidx:
                b_rank = rank
                break

        return {
            "q": q, "a": a_rank, "b": b_rank,
            "a_ms": a_ms, "b_ms": b_ms,
            "gold_doc": gdoc, "gold_idx": gidx,
            "b_top1": b_list[0][0]["doc"] + " / " + b_list[0][0]["sec"] if b_list else "",
        }

    results = await asyncio.gather(*(run_one(r) for r in exact))

    a_ranks = [x["a"] for x in results]
    b_ranks = [x["b"] for x in results]

    print("=" * 80)
    print(f"{'方案':<34}{'Chunk@1':>10}{'Chunk@5':>10}{'MRR':>9}{'平均耗时':>10}")
    print("-" * 80)
    for name, ranks, ms in (
        (f"A  BM25 top{TOPK}（无 reranker）", a_ranks, [x["a_ms"] for x in results]),
        (f"B  BM25 top{POOL} → rerank → top{TOPK}", b_ranks, [x["b_ms"] for x in results]),
    ):
        t1, r5, mrr = m(ranks)
        print(f"{name:<34}{t1:>10.1%}{r5:>10.1%}{mrr:>9.3f}{sum(ms)/len(ms):>9.0f}ms")

    print()
    print("=" * 80)
    print("逐题对比")
    print("=" * 80)
    print(f"{'question':<26}{'A(flat)':>9}{'B(rerank)':>11}{'变化':>8}")
    print("-" * 80)
    improved = worsened = same = 0
    for x in sorted(results, key=lambda y: y["a"]):
        a, b = x["a"], x["b"]
        sa = str(a) if a < 999 else "miss"
        sb = str(b) if b < 999 else "miss"
        if b < a:
            mark, improved = "↑", improved + 1
        elif b > a:
            mark, worsened = "↓", worsened + 1
        else:
            mark, same = "=", same + 1
        print(f"{x['q'][:24]:<26}{sa:>9}{sb:>11}{mark:>8}")

    print()
    print(f"  Top1 改善 {improved} / 变差 {worsened} / 持平 {same}")
    print(f"  reranker 调用: {len(exact)} 次，每文档 {POOL} 篇")

    outp = ROOT / "data" / "eval" / "p2_rerank_pool50.json"
    outp.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  明细写入 {outp}")

    await reranker.close()


asyncio.run(main())

