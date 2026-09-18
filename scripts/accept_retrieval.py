"""
验收：通过**生产代码路径**重跑 bench_v2，确认 BM25 落地生效。

与 p1_dense_vs_bm25.py 的区别：
  p1 用脚本内独立实现的 BM25/稠密
  本脚本走 VectorStoreService.search()，即生产真实路径
两者结果应一致；不一致说明生产代码有问题。
"""

import asyncio
import json
import os
import sys
from pathlib import Path

_PY = Path(Path(__file__).resolve().parent.parent / "python")
os.chdir(_PY)
sys.path.insert(0, str(_PY))
sys.stdout.reconfigure(encoding="utf-8")

from config import settings
from services.vector_store import VectorStoreService

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "data" / "eval" / "bench_v2.jsonl"


async def run(mode: str):
    settings.retrieval_mode = mode
    vs = VectorStoreService()
    await vs.init()

    bench = [json.loads(l) for l in BENCH.read_text(encoding="utf-8").splitlines() if l.strip()]
    doc_items = [r for r in bench if r["in_corpus"] and r["doc_title"]]
    chunk_items = [r for r in doc_items if r["gold_chunk_index"] is not None and not r["ambiguous"]]

    doc_ranks, chunk_ranks = [], []
    for r in doc_items:
        res = await vs.search(r["question"], top_k=20)
        # 文档级：该文档是否出现在结果中，及其最佳名次
        drank = 999
        crank = 999
        for i, (doc, _) in enumerate(res, 1):
            meta = doc.get("metadata") or {}
            title = meta.get("title") or ""
            if title == r["doc_title"] and drank == 999:
                drank = i
            if (title == r["doc_title"]
                    and meta.get("chunk_index") == r["gold_chunk_index"]
                    and crank == 999):
                crank = i
        doc_ranks.append(drank)
        if r in chunk_items:
            chunk_ranks.append(crank)

    def m(ranks):
        n = len(ranks)
        if not n:
            return (0, 0, 0)
        return (
            sum(1 for x in ranks if x == 1) / n,
            sum(1 for x in ranks if x <= 5) / n,
            sum(1 / x for x in ranks if x < 999) / n,
        )

    dt1, dr5, dmrr = m(doc_ranks)
    ct1, cr5, cmrr = m(chunk_ranks)
    return {
        "mode": mode,
        "n_doc": len(doc_ranks), "n_chunk": len(chunk_ranks),
        "doc": (dt1, dr5, dmrr), "chunk": (ct1, cr5, cmrr),
    }


async def main():
    print("生产路径验收（VectorStoreService.search）")
    print("=" * 88)
    print(f"{'mode':<10}{'Doc T1':>9}{'Doc R@5':>10}{'Doc MRR':>10}"
          f"{'Chunk T1':>10}{'Chunk R@5':>11}{'Chunk MRR':>11}")
    print("-" * 88)
    out = {}
    for mode in ("dense", "bm25"):
        r = await run(mode)
        out[mode] = r
        dt1, dr5, dmrr = r["doc"]
        ct1, cr5, cmrr = r["chunk"]
        print(f"{mode:<10}{dt1:>9.1%}{dr5:>10.1%}{dmrr:>10.3f}"
              f"{ct1:>10.1%}{cr5:>11.1%}{cmrr:>11.3f}")
    print()
    print(f"（文档级 {out['bm25']['n_doc']} 题 / chunk 级 {out['bm25']['n_chunk']} 题）")

    p = ROOT / "data" / "eval" / "acceptance_retrieval.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写入 {p}")


asyncio.run(main())
