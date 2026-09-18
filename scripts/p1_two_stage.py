"""
P1 实验：BM25 flat retrieval  vs  BM25 两阶段（文档级 → 章节级）

只读实验 —— 不改动任何生产代码，不修改索引。

设计
----
Baseline（flat）：
    对一个 query，直接在 129 个可召回章节上做 BM25，取 top-k。

Experiment（two-stage）：
    Stage 1: 在 30 篇文档上做 BM25（文档文本 = 标题 + 全部章节正文），
             取 doc top-K。
    Stage 2: **只在 Stage 1 选中的文档内部**做章节级 BM25，取 top-k。

指标（按用户要求分层报告）
--------------------------
Layer 1 — 文档级（全部 38 题）
    Doc Recall@1 / @5 / Doc MRR

Layer 2 — 章节级，两种口径都报：
    条件口径：仅统计 Layer 1 已命中正确文档的题（看"文档内定位能力"上限）
    全量口径：全部 38 题（看端到端实际效果）

另报：平均候选数（两阶段是否真的减少上下文）、错误类型分布。

用法：python scripts/p1_two_stage.py
"""

import asyncio
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

_PY = Path(Path(__file__).resolve().parent.parent / "python")
os.chdir(_PY)
sys.path.insert(0, str(_PY))
sys.stdout.reconfigure(encoding="utf-8")

from agents.doc_parser_agent import DocParserAgent
from services.bm25 import BM25Index

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(
    r"C:\Users\滨滨\Desktop\Knowledge-Base-Intelligent-Q-A-main\data\clean\institutional"
)
BENCH = ROOT / "data" / "eval" / "bench_v3.jsonl"

STAGE1_K = 5   # 文档级召回数量
STAGE2_K = 5   # 章节级召回数量


def build_corpus():
    """构造与生产一致的章节语料（含 searchable 语义），以及文档级文本。"""
    parser = DocParserAgent()
    docs: dict[str, list[tuple[str, str]]] = {}
    for f in sorted(SRC.glob("*.md")):
        dt, secs = parser.parse_tree(
            parser.strip_metadata(f.read_text(encoding="utf-8", errors="ignore"))
        )
        if dt:
            docs[dt] = [(s.title, s.body) for s in secs if s.body]

    chapters = []       # 全部章节（用于定位 gold）
    searchable = []     # 参与召回的章节（排除 doc_overview）
    for dt, secs in docs.items():
        for i, (st, body) in enumerate(secs):
            rec = {"doc": dt, "idx": i, "sec": st, "body": body,
                   "searchable": True}
            chapters.append(rec)

    # 与生产一致：用 classify_section 排除文档说明
    for c in chapters:
        c["searchable"] = DocParserAgent.classify_section(c["sec"], c["body"]) != "doc_overview"

    # 文档级文本 = 标题 + 全部章节正文
    doc_texts = []
    doc_titles = []
    for dt, secs in docs.items():
        doc_titles.append(dt)
        doc_texts.append(dt + "\n" + "\n".join(b for _, b in secs))

    return docs, chapters, doc_titles, doc_texts


def metrics(ranks: list[int]) -> tuple[float, float, float]:
    n = len(ranks)
    if not n:
        return (0.0, 0.0, 0.0)
    return (
        sum(1 for r in ranks if r == 1) / n,
        sum(1 for r in ranks if r <= 5) / n,
        sum(1 / r for r in ranks if r < 999) / n,
    )


async def main():
    docs, chapters, doc_titles, doc_texts = build_corpus()

    # ── 索引 ────────────────────────────────────────────
    # flat：只索引 searchable 章节
    flat_idx = BM25Index()
    flat_items = [c for c in chapters if c["searchable"]]
    flat_idx.build(
        [f"{c['doc']}\n章节：{c['sec']}\n{c['body']}" for c in flat_items],
        [str(i) for i in range(len(flat_items))],
    )

    # 文档级索引
    doc_idx = BM25Index()
    doc_idx.build(doc_texts, [str(i) for i in range(len(doc_texts))])

    # 章节级：预先为每篇文档建子索引
    per_doc_idx: dict[str, tuple[BM25Index, list[dict]]] = {}
    for dt in doc_titles:
        items = [c for c in chapters if c["doc"] == dt and c["searchable"]]
        if not items:
            continue
        idx = BM25Index()
        idx.build(
            [f"{c['doc']}\n章节：{c['sec']}\n{c['body']}" for c in items],
            [str(i) for i in range(len(items))],
        )
        per_doc_idx[dt] = (idx, items)

    print(f"文档 {len(doc_titles)} 篇 / 全部章节 {len(chapters)} / 可召回 {len(flat_items)}")
    print(f"stage1_k={STAGE1_K}  stage2_k={STAGE2_K}")
    print()

    bench = [
        json.loads(l)
        for l in BENCH.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    exact = [r for r in bench if r["gold_type"] == "exact"]
    print(f"评测：exact 类 {len(exact)} 题")
    print()

    flat_doc_ranks, flat_ch_ranks = [], []
    ts_doc_ranks, ts_ch_ranks = [], []
    err_types = Counter()
    cond_flat, cond_ts = [], []      # 条件口径（仅文档级已命中）
    cand_counts = []

    detail = []

    for r in exact:
        q = r["question"]
        gdoc = r["doc_title"]
        gidx = r["gold_chunk_index"]

        # ── flat：直接章节级 ─────────────────────────────
        fhits = flat_idx.rank_ids(q, top_k=STAGE2_K)
        flat_chunk_rank = 999
        for rank, (sid, _) in enumerate(fhits, 1):
            c = flat_items[int(sid)]
            if c["doc"] == gdoc and c["idx"] == gidx:
                flat_chunk_rank = rank
                break
        flat_ch_ranks.append(flat_chunk_rank)

        # flat 的文档级名次：在更大候选池里找 gold 所属文档首次出现的名次。
        # 这样与 two-stage 的 Doc@k 口径可比（都衡量"能否找到正确制度"）。
        flat_doc_rank = 999
        wide = flat_idx.rank_ids(q, top_k=30)
        for rank, (sid, _) in enumerate(wide, 1):
            if flat_items[int(sid)]["doc"] == gdoc:
                flat_doc_rank = rank
                break
        flat_doc_ranks.append(flat_doc_rank)

        # ── two-stage ───────────────────────────────────
        dhits = doc_idx.rank_ids(q, top_k=STAGE1_K)
        doc_order = [doc_titles[int(i)] for i, _ in dhits]
        try:
            d_rank = doc_order.index(gdoc) + 1
        except ValueError:
            d_rank = 999
        ts_doc_ranks.append(d_rank)

        # stage 2：只在 stage1 结果文档内检索
        cands = []
        for dt in doc_order:
            if dt in per_doc_idx:
                idx2, items2 = per_doc_idx[dt]
                for sid, sc in idx2.rank_ids(q, top_k=STAGE2_K):
                    cands.append((sc, items2[int(sid)]))
        cands.sort(key=lambda x: (-x[0], x[1]["doc"]))
        cands = cands[:STAGE2_K]
        cand_counts.append(len(cands))

        ts_rank = 999
        for rank, (_, c) in enumerate(cands, 1):
            if c["doc"] == gdoc and c["idx"] == gidx:
                ts_rank = rank
                break
        ts_ch_ranks.append(ts_rank)

        # 条件口径：仅统计"第一层已命中正确文档"的题
        #   two-stage：stage1 doc@k 命中
        #   flat     ：flat 候选池前列已覆盖正确文档
        if d_rank <= STAGE1_K:
            cond_ts.append(ts_rank)
        if flat_doc_rank <= STAGE1_K:
            cond_flat.append(flat_chunk_rank)

        # 错误类型
        if ts_rank == 1:
            err_types["正确"] += 1
        elif d_rank == 999:
            err_types["错文档"] += 1
        else:
            err_types["错章节"] += 1

        detail.append({
            "question": q, "gold_doc": gdoc, "gold_idx": gidx,
            "flat_doc_rank": flat_doc_rank,
            "flat_chunk_rank": flat_chunk_rank,
            "ts_doc_rank": d_rank, "ts_chunk_rank": ts_rank,
        })

    # ── 报告 ────────────────────────────────────────────
    print("=" * 84)
    print("Layer 1 — 文档级（全部 38 题）")
    print("=" * 84)
    for name, ranks in (("BM25 flat", flat_doc_ranks), ("Two-stage", ts_doc_ranks)):
        t1, r5, mrr = metrics(ranks)
        print(f"  {name:<14} Doc@1 {t1:>6.1%}   Doc@5 {r5:>6.1%}   Doc MRR {mrr:.3f}")
    print("  注：flat 无独立文档级阶段，用其章节结果所属文档近似")

    print()
    print("=" * 84)
    print("Layer 2 — 章节级")
    print("=" * 84)
    print(f"{'方法':<14}{'口径':<12}{'Chunk@1':>10}{'Chunk@5':>10}{'Chunk MRR':>11}{'n':>6}")
    print("-" * 84)
    for name, ranks in (("BM25 flat", flat_ch_ranks), ("Two-stage", ts_ch_ranks)):
        t1, r5, mrr = metrics(ranks)
        print(f"{name:<14}{'全量':<12}{t1:>10.1%}{r5:>10.1%}{mrr:>11.3f}{len(ranks):>6}")
    for name, ranks in (("BM25 flat", cond_flat), ("Two-stage", cond_ts)):
        t1, r5, mrr = metrics(ranks)
        print(f"{name:<14}{'条件(文档已中)':<12}{t1:>10.1%}{r5:>10.1%}{mrr:>11.3f}{len(ranks):>6}")

    print()
    print("=" * 84)
    print("其他指标")
    print("=" * 84)
    print(f"  Two-stage 平均候选数: {sum(cand_counts)/len(cand_counts):.1f}  "
          f"(flat 固定 {STAGE2_K})")
    print(f"  错误类型分布: {dict(err_types)}")

    print()
    print("=" * 84)
    print("逐题明细")
    print("=" * 84)
    print(f"{'question':<24}{'f_doc':>7}{'f_chk':>7}{'ts_doc':>8}{'ts_chk':>8}")
    print("-" * 84)
    for d in detail:
        def s(v):
            return str(v) if v < 999 else "miss"
        print(f"{d['question'][:22]:<24}{s(d['flat_doc_rank']):>7}"
              f"{s(d['flat_chunk_rank']):>7}{s(d['ts_doc_rank']):>8}"
              f"{s(d['ts_chunk_rank']):>8}")

    outp = ROOT / "data" / "eval" / "p1_two_stage.json"
    outp.write_text(json.dumps({"detail": detail, "err_types": dict(err_types)},
                               ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"明细写入 {outp}")


asyncio.run(main())
