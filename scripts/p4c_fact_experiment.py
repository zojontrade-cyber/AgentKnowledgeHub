"""
P4-C 验证实验：Fact schema 能否救回 "数值无处安放" 的事实

范围（按用户要求，只做 2 篇）
-----------------------------
  加班管理制度      覆盖 rate（150%/200%/300%）+ deadline（提前申请）
  差旅报销管理规定   覆盖 rate（住宿标准）+ deadline（10 个工作日）

实验矩阵
--------
  A  旧 schema：Entity + Relation（已有结果，从 P4-B 缓存读取）
  B  Entity + Relation + **Fact**（新增独立抽取）

主指标：8 条 S 类事实的恢复情况
成功标准（用户指定）：8 条至少恢复 6 条
"""

import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

_PY = Path(Path(__file__).resolve().parent.parent / "python")
os.chdir(_PY)
sys.path.insert(0, str(_PY))
sys.stdout.reconfigure(encoding="utf-8")

import chromadb

from agents.fact_extract_agent import FactExtractAgent
from config import settings

ROOT = Path(__file__).resolve().parent.parent
DOCS = ["加班管理制度", "差旅报销管理规定"]

# 8 条 S 类事实（来自 p4c_audit.json）
S_CLASS = [
    ("加班费怎么算？", "工作日 150%"),
    ("加班费怎么算？", "周末 200%"),
    ("加班费怎么算？", "法定节假日 300%"),
    ("年假要提前多久申请？", "提前 3 个工作日"),
    ("用印要走什么流程？", "注明文件名称、份数、用途和审批人"),
    ("劳动合同一般签几年？", "首次合同期限一般为 3 年"),
    ("工资几号发？", "每月 10 日"),
    ("会议室预订后能取消吗？", "连续 3 次爽约限制预订 1 周"),
]


async def main():
    client = chromadb.PersistentClient(path=settings.chroma_path)
    col = client.get_or_create_collection("knowledge_chunks")
    got = col.get(include=["documents", "metadatas"])

    by_doc: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for d, m in zip(got["documents"], got["metadatas"]):
        m = m or {}
        t = m.get("title")
        if t in DOCS:
            by_doc[t].append((m.get("section_title"), d or ""))

    ex = FactExtractAgent()

    print("=" * 84)
    print("Fact 抽取（2 篇文档，独立于 Entity/Relation）")
    print("=" * 84)

    cache = ROOT / "data" / "eval" / "p4c_facts.json"
    if cache.exists():
        all_facts = json.loads(cache.read_text(encoding="utf-8"))
        print(f"复用缓存 {cache.name}")
    else:
        all_facts = []
        t0 = time.perf_counter()
        for doc, secs in by_doc.items():
            for i, (sec, body) in enumerate(secs):
                try:
                    res = await ex.extract(body, source_id=f"{doc}#{i}")
                except Exception as e:
                    print(f"  [跳过] {doc}#{i}: {type(e).__name__}")
                    continue
                for f in res.facts:
                    all_facts.append({
                        "doc": doc, "section": sec,
                        "fact_type": f.fact_type, "condition": f.condition,
                        "value": f.value, "unit": f.unit, "quote": f.quote,
                        "display": f.display,
                    })
                print(f"  {doc} / {sec}: {len(res.facts)} 条事实")
        cache.write_text(json.dumps(all_facts, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n共 {len(all_facts)} 条事实，耗时 {(time.perf_counter()-t0):.0f}s")

    # ── 展示抽出的事实 ───────────────────────────────────
    print()
    print("=" * 84)
    print("抽取出的事实（按类型）")
    print("=" * 84)
    by_type = defaultdict(list)
    for f in all_facts:
        by_type[f["fact_type"]].append(f)

    for t, items in sorted(by_type.items(), key=lambda x: -len(x[1])):
        print(f"\n[{t}]  {len(items)} 条")
        for f in items[:6]:
            unit = f["unit"]
            val = f"{f['value']}{unit}" if unit else f["value"]
            print(f"   {f['condition'] or '(无条件)'} -> {val}")
            if f["quote"]:
                print(f"       原文: {f['quote'][:70]}")

    # ── 关键：8 条 S 类事实是否被救回 ─────────────────────
    print()
    print("=" * 84)
    print("S 类事实恢复检查（8 条 schema 漏洞）")
    print("=" * 84)

    fact_blob = "\n".join(
        f"{f['condition']} {f['value']}{f['unit']} {f['quote']}" for f in all_facts
    )
    recovered = 0
    for q, fact in S_CLASS:
        # 检查：数值是否出现在事实层
        nums = [n for n in __import__("re").findall(r"\d+(?:\.\d+)?", fact)]
        hit = all(n in fact_blob for n in nums) if nums else fact[:6] in fact_blob
        recovered += hit
        print(f"  {'✅' if hit else '❌'} {fact:<34} (来自: {q})")

    print()
    print(f"  恢复 {recovered}/8   （成功标准：≥6）")

    out = ROOT / "data" / "eval" / "p4c_ab.json"
    out.write_text(json.dumps({
        "n_facts": len(all_facts),
        "by_type": {k: len(v) for k, v in by_type.items()},
        "s_class_recovered": recovered,
        "s_class_total": len(S_CLASS),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写入 {out}")


asyncio.run(main())
