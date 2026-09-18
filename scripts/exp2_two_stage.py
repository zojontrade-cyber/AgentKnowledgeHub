"""
实验 2：两阶段检索（文档级召回 -> 章节精排）

你的设计：
  query
    ↓
  文档级召回（30 个向量）
    ↓
  在该文档内做章节精排（5-10 个候选）
    ↓
  答案

对比单阶段（159 章节直接竞争）。
"""

import asyncio
import math
import os
import sys
from pathlib import Path

# .env 由 pydantic-settings 按 **cwd** 解析，脚本必须先把 cwd 固定到 python/
_PY = Path(__file__).resolve().parent.parent / "python"
os.chdir(_PY)
sys.path.insert(0, str(_PY))
sys.stdout.reconfigure(encoding="utf-8")

from langchain_openai import OpenAIEmbeddings

from agents.doc_parser_agent import DocParserAgent
from config import settings


def cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-9)


emb = OpenAIEmbeddings(
    model=settings.embedding_model,
    api_key=settings.effective_embedding_api_key,
    base_url=settings.effective_embedding_base_url,
)

SRC = Path(r"C:\Users\滨滨\Desktop\Knowledge-Base-Intelligent-Q-A-main\data\clean\institutional")
parser = DocParserAgent()

QUERIES = [
    ("公司的标准工作时间是什么时候到什么时候", "员工考勤管理制度", "工作时间"),
    ("请假要走什么流程", "员工考勤管理制度", "请假"),
    ("保密级别分几级", "保密管理制度", "密级"),
    ("出差住宿费能报多少", "差旅报销管理规定", "住宿"),
    ("加班费怎么算", "加班管理制度", "加班费"),
    ("年假有多少天", "年休假管理办法", "年假"),
    ("合同谁来审批", "合同管理办法", "审批"),
    ("客户资料能导出吗", "客户资料管理规定", "导出"),
]


async def main():
    # 构造：文档级摘要 + 章节级
    docs = {}       # title -> {"summary": str, "sections": [(sec_title, body)]}
    for f in sorted(SRC.glob("*.md")):
        raw = f.read_text(encoding="utf-8", errors="ignore")
        clean = parser.strip_metadata(raw)
        dt, sections = parser.parse_tree(clean)
        if not dt:
            continue
        secs = [(s.title, s.body) for s in sections if s.body and len(s.body) >= 30]
        if not secs:
            continue
        parts = []
        for st, bd in secs:
            parts.append(f"{st}：{bd[:80]}")
        docs[dt] = {
            "summary": dt + "\n" + "\n".join(parts),
            "sections": secs,
            "file": f.name,
        }

    titles = list(docs.keys())
    print(f"文档数: {len(titles)}   章节总数: {sum(len(d['sections']) for d in docs.values())}")
    print()

    # 编码文档级摘要
    doc_vecs = await emb.aembed_documents([docs[t]["summary"] for t in titles])

    # ── 阶段 1：文档级召回 ───────────────────────────────
    print("=" * 78)
    print("阶段 1：文档级召回（30 个向量竞争）")
    print("=" * 78)
    doc_hits = 0
    doc_ranks = []
    for q, expect_doc, expect_sec in QUERIES:
        qv = await emb.aembed_query(q)
        scored = sorted(
            ((cos(qv, v), t) for v, t in zip(doc_vecs, titles)),
            key=lambda x: x[0], reverse=True,
        )
        rank = next((i + 1 for i, (_, t) in enumerate(scored) if t == expect_doc), 0)
        if rank == 1:
            doc_hits += 1
        doc_ranks.append(rank or 99)
        mark = "★" if rank == 1 else (" " if rank and rank <= 5 else "✗")
        print(f"  {mark} {q[:24]:<26} 文档排名={rank or '>10':<5} "
              f"top1={scored[0][1][:18]:<20} 分数={scored[0][0]:.4f}")

    print(f"  → 文档级 Top-1: {doc_hits}/{len(QUERIES)}   "
          f"Top-5: {sum(1 for r in doc_ranks if r <= 5)}/{len(QUERIES)}")
    print(f"  → MRR: {sum(1/r for r in doc_ranks if r < 99)/len(QUERIES):.3f}")

    # ── 阶段 2：在召回的文档内做章节精排 ──────────────────
    print()
    print("=" * 78)
    print("阶段 2：文档内章节精排（每篇仅 5-10 个候选）")
    print("=" * 78)
    sec_hits = 0
    sec_ranks = []
    for q, expect_doc, expect_sec in QUERIES:
        qv = await emb.aembed_query(q)
        secs = docs[expect_doc]["sections"]
        texts = [f"{expect_doc}\n章节：{st}\n{bd}" for st, bd in secs]
        vecs = await emb.aembed_documents(texts)
        scored = sorted(
            ((cos(qv, v), st, bd) for v, (st, bd) in zip(vecs, secs)),
            key=lambda x: x[0], reverse=True,
        )
        rank = next(
            (i + 1 for i, (_, st, bd) in enumerate(scored)
             if expect_sec in st or expect_sec in bd[:60]),
            0,
        )
        if rank == 1:
            sec_hits += 1
        sec_ranks.append(rank or 99)
        mark = "★" if rank == 1 else (" " if rank and rank <= 3 else "✗")
        print(f"  {mark} {q[:24]:<26} 章节排名={rank or '?':<4} "
              f"候选={len(secs)}  top1={scored[0][1][:20]}")

    print(f"  → 章节级 Top-1: {sec_hits}/{len(QUERIES)}")
    print(f"  → 章节级 MRR: {sum(1/r for r in sec_ranks if r < 99)/len(QUERIES):.3f}")

    print()
    print("=" * 78)
    print("对比单阶段（159 章节直接竞争）")
    print("=" * 78)
    all_items = [(t, st, bd) for t in titles for st, bd in docs[t]["sections"]]
    all_texts = [f"{t}\n章节：{st}\n{bd}" for t, st, bd in all_items]
    all_vecs = await emb.aembed_documents(all_texts)
    single_hits = 0
    single_ranks = []
    for q, expect_doc, expect_sec in QUERIES:
        qv = await emb.aembed_query(q)
        scored = sorted(
            ((cos(qv, v), it) for v, it in zip(all_vecs, all_items)),
            key=lambda x: x[0], reverse=True,
        )
        rank = next(
            (i + 1 for i, (_, (t, st, bd)) in enumerate(scored)
             if t == expect_doc and (expect_sec in st or expect_sec in bd[:60])),
            0,
        )
        if rank == 1:
            single_hits += 1
        single_ranks.append(rank or 99)
    print(f"  → 单阶段 Top-1: {single_hits}/{len(QUERIES)}")
    print(f"  → 单阶段 MRR: {sum(1/r for r in single_ranks if r < 99)/len(QUERIES):.3f}")
    print(f"  → 单阶段 Recall@5: {sum(1 for r in single_ranks if r <= 5)}/{len(QUERIES)}")

    print()
    print("=" * 78)
    print("结论")
    print("=" * 78)
    two_mrr = sum(1/r for r in doc_ranks if r < 99) / len(QUERIES)
    print(f"  两阶段 MRR: {(two_mrr + sum(1/r for r in sec_ranks if r < 99)/len(QUERIES))/2:.3f}")
    print(f"  单阶段 MRR: {sum(1/r for r in single_ranks if r < 99)/len(QUERIES):.3f}")


asyncio.run(main())
