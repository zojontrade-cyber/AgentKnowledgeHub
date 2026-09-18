"""
P4 前置测量：三种抽取策略的实际成本对比

对比（**不调 LLM**，用字符/调用数估算，先算清楚再决定）：
  A 现状：chunk-level，159 次调用
  B 文档级：document-level，30 次调用
  C 章节级（合并同文档多 chunk）：先看实际章节数

关键：P3.1 已测出 system prompt 675 字符 vs chunk 正文中位 107 字符
      -> 86% 的 token 花在重复 prompt 上。
"""

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

_PY = Path(Path(__file__).resolve().parent.parent / "python")
os.chdir(_PY)
sys.path.insert(0, str(_PY))
sys.stdout.reconfigure(encoding="utf-8")

import chromadb

from config import settings
from agents.knowledge_extract_agent import EXTRACTION_SYSTEM_PROMPT

print("=" * 80)
print("抽取 prompt 结构")
print("=" * 80)
print(f"  system prompt: {len(EXTRACTION_SYSTEM_PROMPT)} 字符")
print(f"  用户消息模板: '请从以下文本中抽取知识：\\n\\n' = 15 字符")
overhead = len(EXTRACTION_SYSTEM_PROMPT) + 15
print(f"  固定开销/次: {overhead} 字符")

# ── 语料 ──────────────────────────────────────────────
client = chromadb.PersistentClient(path=settings.chroma_path)
col = client.get_or_create_collection("knowledge_chunks")
got = col.get(include=["documents", "metadatas"])

by_doc = defaultdict(list)
for d, m in zip(got["documents"], got["metadatas"]):
    m = m or {}
    by_doc[m.get("title") or m.get("doc_id")].append((m.get("section_title"), d or ""))

n_chunks = len(got["ids"])
n_docs = len(by_doc)
n_sections = sum(len(v) for v in by_doc.values())
chunk_chars = sum(len(d or "") for d in got["documents"])
doc_chars = 0
for k, v in by_doc.items():
    # 文档级文本 ≈ 标题 + 各章节正文（与生产 display 一致）
    doc_chars += len(k) + sum(len(c) for _, c in v)

print()
print("=" * 80)
print("三种策略的成本估算")
print("=" * 80)
print(f"  语料: {n_docs} 文档 / {n_sections} 章节 / {n_chunks} chunk")
print()

def tok(chars):
    """中文约 1.5 字符/token"""
    return int(chars / 1.5)

rows = [
    ("A 现状 chunk-level", n_chunks, chunk_chars),
    ("B 章节级 section-level", n_sections, chunk_chars),
    ("C 文档级 document-level", n_docs, doc_chars),
]

print(f"  {'策略':<26}{'调用数':>8}{'正文token':>11}{'prompt token':>14}{'总 token':>11}{'省':>8}")
print("-" * 80)
base = None
for name, calls, body in rows:
    body_tok = tok(body)
    prompt_tok = tok(overhead) * calls
    total = body_tok + prompt_tok
    if base is None:
        base = total
    save = 1 - total / base
    print(f"  {name:<26}{calls:>8}{body_tok:>11,}{prompt_tok:>14,}{total:>11,}{save:>8.0%}")

print()
print("=" * 80)
print("关键洞察")
print("=" * 80)
print(f"  chunk 平均正文长度: {chunk_chars/n_chunks:.0f} 字符")
print(f"  文档平均正文长度:   {doc_chars/n_docs:.0f} 字符")
print(f"  单次固定开销/正文比: {overhead/(chunk_chars/n_chunks):.1f}x")
print()
print("  -> chunk 级抽取时，prompt 开销是正文的 "
      f"{overhead/(chunk_chars/n_chunks):.0f} 倍，")
print("     绝大部分 token 花在**重复发送同一段 prompt** 上。")
print()
print(f"  文档级把 {n_chunks} 次调用降到 {n_docs} 次，")
print(f"  prompt 开销从 {tok(overhead)*n_chunks:,} 降到 {tok(overhead)*n_docs:,} token。")
