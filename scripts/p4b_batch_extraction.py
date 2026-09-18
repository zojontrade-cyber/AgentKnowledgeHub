"""
P4-B：章节批处理抽取实验（找 Pareto 点）

设计（按用户要求）
------------------
  A  1 chunk/call   （现状 baseline）
  B  2 chunks/call
  C  3 chunks/call
  ！不超过 3 —— 一次塞 6 章节会让 LLM 做"主题压缩"，
     丢失章节级细节（文档级实验已证明：实体 -52%，且丢的是数值）。

评价指标（**不看 Entity 总数**）
-------------------------------
  1. QA fact recall        ← 最高优先
     42 题中每题的"答案事实"（如"10 个工作日"、"9:00-18:00"）
     是否仍出现在抽取结果里
  2. Answer-bearing entity recall
     gold 数值型实体（时间/数字）的保留率
  3. Entity recall（辅助）
     与 baseline 实体集合的交集比例
  4. token 成本

成功标准：QA fact recall ≈100%，token ↓≥40%，调用 ↓≥50%
"""

import asyncio
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

_PY = Path(Path(__file__).resolve().parent.parent / "python")
os.chdir(_PY)
sys.path.insert(0, str(_PY))
sys.stdout.reconfigure(encoding="utf-8")

import chromadb

from agents.doc_parser_agent import DocType, DocumentChunk
from agents.knowledge_extract_agent import EXTRACTION_SYSTEM_PROMPT, KnowledgeExtractAgent
from config import settings

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "data" / "eval" / "bench_v3.jsonl"

PROMPT_OVERHEAD = len(EXTRACTION_SYSTEM_PROMPT) + 15


def norm(s: str) -> str:
    return re.sub(r"[\s\u3000]+", "", unicodedata.normalize("NFKC", s or ""))


def facts_of(gold: str) -> list[str]:
    """从 gold 答案里抽"事实值"：数字/时间等"""
    g = norm(gold)
    g = re.sub(r"\|[-:\s|]+\|", "|", g).replace("|", " ")
    nums = re.findall(r"\d+(?:[:.]\d+)*", g)
    return [n for n in nums if len(n) >= 1][:6]


async def main():
    client = chromadb.PersistentClient(path=settings.chroma_path)
    col = client.get_or_create_collection("knowledge_chunks")
    got = col.get(include=["documents", "metadatas"])

    by_doc: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for d, m in zip(got["documents"], got["metadatas"]):
        m = m or {}
        by_doc[m.get("title") or m.get("doc_id")].append((m.get("section_title"), d or ""))

    # 评测题 -> 涉及的文档
    bench = [json.loads(l) for l in BENCH.read_text(encoding="utf-8").splitlines() if l.strip()]
    exact = [r for r in bench if r["gold_type"] == "exact"]

    ex = KnowledgeExtractAgent()
    arms = {}

    for arm, batch in (("A_1chunk", 1), ("B_2chunk", 2), ("C_3chunk", 3)):
        print("=" * 78)
        print(f"运行 {arm}（{batch} chunk/call）")
        print("=" * 78)
        t0 = time.perf_counter()
        calls = 0
        ents_all: list[str] = []
        per_doc_ents: dict[str, set] = defaultdict(set)

        for title, secs in by_doc.items():
            chunks = [
                DocumentChunk(
                    content=c, doc_id=title, chunk_index=i,
                    doc_type=DocType.MARKDOWN,
                    metadata={"section_title": sec, "title": title},
                )
                for i, (sec, c) in enumerate(secs)
            ]
            for i in range(0, len(chunks), batch):
                grp = chunks[i : i + batch]
                # 合并成一次调用：用换行拼接（保留章节边界）
                merged = "\n\n".join(
                    f"## {g.metadata.get('section_title','')}\n{g.content}" for g in grp
                )
                res = await ex.extract_single(merged, chunk_id=f"{title}#{i}")
                calls += 1
                for e in res.entities:
                    ents_all.append(e.name)
                    per_doc_ents[title].add(norm(e.name))
                for rel in res.relations:
                    per_doc_ents[title].add(norm(rel.head))
                    per_doc_ents[title].add(norm(rel.tail))

        ms = (time.perf_counter() - t0) * 1000
        arms[arm] = {
            "batch": batch, "calls": calls, "ms": ms,
            "entities": len(ents_all), "unique": len(set(norm(x) for x in ents_all)),
            "per_doc": {k: v for k, v in per_doc_ents.items()},
        }
        print(f"  调用 {calls}  实体 {len(ents_all)}  唯一 {len(set(norm(x) for x in ents_all))}"
              f"  耗时 {ms/1000:.1f}s")
        print()

    # ── 评价：QA fact recall ──────────────────────────────
    print("=" * 78)
    print("QA fact recall（每题 gold 的事实值是否被抽取到）")
    print("=" * 78)

    base = arms["A_1chunk"]["per_doc"]
    for arm in arms:
        hits = total = 0
        misses = []
        for r in exact:
            doc = r["doc_title"]
            if doc not in arms[arm]["per_doc"]:
                continue
            blob = " ".join(arms[arm]["per_doc"][doc])
            for f in facts_of(r.get("expected_answer") or ""):
                total += 1
                if norm(f) in blob:
                    hits += 1
                elif len(misses) < 8:
                    misses.append((r["question"][:20], f))
        pct = hits / total if total else 0
        arms[arm]["fact_recall"] = pct
        arms[arm]["fact_hits"] = hits
        arms[arm]["fact_total"] = total
        print(f"  {arm:<12} {hits}/{total} = {pct:.1%}")
        if arm != "A_1chunk" and misses:
            print(f"     漏掉样例: {misses[:5]}")

    # ── Entity recall ────────────────────────────────────
    print()
    print("=" * 78)
    print("Entity recall（相对 baseline 的实体保留率）")
    print("=" * 78)
    for arm in arms:
        inter = union = 0
        for doc, names in base.items():
            other = arms[arm]["per_doc"].get(doc, set())
            inter += len(names & other)
            union += len(names | other)
        arms[arm]["entity_recall"] = inter / union if union else 0
        arms[arm]["entity_intersect"] = inter
        print(f"  {arm:<12} Jaccard {arms[arm]['entity_recall']:.1%}  "
              f"(交集 {inter})")

    # ── token 成本 ───────────────────────────────────────
    print()
    print("=" * 78)
    print("成本对比")
    print("=" * 78)
    total_body = sum(len(c or "") for d in by_doc.values() for _, c in d)
    print(f"  {'方案':<12}{'调用':>8}{'正文token':>11}{'prompt token':>14}{'总token':>11}{'省':>7}")
    base_tok = None
    for arm, a in arms.items():
        body_tok = int(total_body / 1.5)
        prompt_tok = int(PROMPT_OVERHEAD / 1.5) * a["calls"]
        tot = body_tok + prompt_tok
        if base_tok is None:
            base_tok = tot
        a["total_tokens"] = tot
        print(f"  {arm:<12}{a['calls']:>8}{body_tok:>11,}{prompt_tok:>14,}"
              f"{tot:>11,}{1-tot/base_tok:>7.0%}")

    print()
    print("=" * 78)
    print("Pareto 判定（成功标准: fact recall ≈100%, token ↓≥40%, 调用 ↓≥50%）")
    print("=" * 78)
    for arm, a in arms.items():
        ok_fact = a["fact_recall"] >= 0.98
        ok_tok = (1 - a["total_tokens"] / base_tok) >= 0.40
        ok_call = (1 - a["calls"] / arms["A_1chunk"]["calls"]) >= 0.50
        verdict = "✅ 达标" if (ok_fact and ok_tok and ok_call) else (
            "⚠️ 部分" if ok_fact else "❌ fact 不达标")
        print(f"  {arm:<12} fact={a['fact_recall']:.1%}  "
              f"token↓{1-a['total_tokens']/base_tok:.0%}  "
              f"调用↓{1-a['calls']/arms['A_1chunk']['calls']:.0%}   {verdict}")

    out = ROOT / "data" / "eval" / "p4b_batch_extraction.json"
    out.write_text(json.dumps(
        {k: {kk: vv for kk, vv in v.items() if kk != "per_doc"} for k, v in arms.items()},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写入 {out}")


asyncio.run(main())
