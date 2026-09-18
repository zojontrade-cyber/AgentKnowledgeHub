"""
P4-C Step 1：Extraction Recall Audit

不改 prompt。逐条分析 A 臂（chunk 级，基线）漏掉的 15 条事实，
判定根因属于：
    P  prompt 问题      —— prompt 没要求这类信息
    S  schema 问题      —— 三元组结构无法表达该事实
    M  模型能力问题      —— 文本里有，但模型没抽
    J  judge 误判       —— 其实抽到了，判定错

方法：对每条漏掉的事实，**打印该 chunk 的原文**与**抽取结果的完整内容**，
人工（我）对照判定根因。
"""

import json
import os
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

_PY = Path(Path(__file__).resolve().parent.parent / "python")
os.chdir(_PY)
sys.path.insert(0, str(_PY))
sys.stdout.reconfigure(encoding="utf-8")

import chromadb

from config import settings

ROOT = Path(__file__).resolve().parent.parent

# ── 载入抽取结果（A 臂）──────────────────────────────────
part = ROOT / "data" / "eval" / "p4b_part_A_1chunk.jsonl"
records = [json.loads(l) for l in part.read_text(encoding="utf-8").splitlines() if l.strip()]
by_doc: dict[str, list[dict]] = defaultdict(list)
for r in records:
    by_doc[r["doc"]].append(r)
print(f"A 臂抽取记录: {len(records)} 条，覆盖 {len(by_doc)} 篇文档")

# ── 载入判定结果 ─────────────────────────────────────────
fr = json.loads((ROOT / "data" / "eval" / "p4b_fact_recall.json").read_text(encoding="utf-8"))
missing = fr["A_1chunk"]["missing"]
print(f"判定缺失的事实: {len(missing)} 条")
print()

# ── 载入原始 chunk 文本 ──────────────────────────────────
client = chromadb.PersistentClient(path=settings.chroma_path)
col = client.get_or_create_collection("knowledge_chunks")
got = col.get(include=["documents", "metadatas"])
raw_by_doc: dict[str, list[tuple[str, str]]] = defaultdict(list)
for d, m in zip(got["documents"], got["metadatas"]):
    m = m or {}
    raw_by_doc[m.get("title") or m.get("doc_id")].append((m.get("section_title"), d or ""))

bench = {json.loads(l)["question"]: json.loads(l)
         for l in (ROOT / "data" / "eval" / "bench_v3.jsonl").read_text(encoding="utf-8").splitlines()
         if l.strip()}

print("=" * 92)
print("逐条漏抽分析")
print("=" * 92)

for q, fact, reason in missing:
    b = bench.get(q, {})
    doc = b.get("doc_title")
    sec = b.get("section_title")
    print()
    print("-" * 92)
    print(f"Q: {q}")
    print(f"事实: {fact}")
    print(f"gold 章节: {doc} / {sec}")
    print(f"judge 理由: {reason[:90]}")

    # 原文（该章节）
    raw = next((c for s, c in raw_by_doc.get(doc, []) if s == sec), "")
    if raw:
        print(f"\n原文（{len(raw)} 字符）:")
        print(f"   {raw.strip()[:400]}")

    # 抽取结果（该文档全部）
    ext = "\n".join(r["text"] for r in by_doc.get(doc, []))
    print(f"\n抽取结果（该文档，{len(ext)} 字符）:")
    # 只显示与该事实相关的部分
    kw = [w for w in re.split(r"[\s，。、]+", fact) if len(w) >= 2][:4]
    shown = 0
    for line in ext.split("\n"):
        if any(k in line for k in kw):
            print(f"   {line[:130]}")
            shown += 1
            if shown >= 8:
                break
    if shown == 0:
        print(f"   （无包含关键词 {kw} 的条目）")
        print(f"   该文档抽取结果前 6 行:")
        for line in ext.split("\n")[:6]:
            print(f"     {line[:120]}")
