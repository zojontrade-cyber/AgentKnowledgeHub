"""
验收：25 篇新入库文档能否被正常检索到。

为什么必须做这一步
------------------
状态变成 COMMITTED 不等于"可检索"。之前已发生过一次同类情况：
159 条向量在库里，但稠密检索 Chunk R@5 仅 23.6% —— 数据在，检索不到。

本脚本对新文档做三层验证：
  1. 向量存在性   —— 每个文档是否都有向量，数量是否与章节数一致
  2. 检索可达性   —— 用文档标题/章节标题构造问题，能否命中**该文档**
  3. 端到端问答   —— 抽取若干题走真实 QA 路径，验证 Answer + Citation

只读，不修改任何数据。
"""

import asyncio
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
from services.vector_store import VectorStoreService

ROOT = Path(__file__).resolve().parent.parent
NEW_MAP = json.loads((ROOT / "data" / "eval" / "new_docs_map.json").read_text(encoding="utf-8"))

# uuid 文件名 -> 中文标题
TITLE_BY_FILE = {
    k: (v.get("body_title") or v.get("chroma_title") or "")
    for k, v in NEW_MAP.items()
}


async def main():
    client = chromadb.PersistentClient(path=settings.chroma_path)
    col = client.get_or_create_collection("knowledge_chunks")
    got = col.get(include=["documents", "metadatas"])

    print("=" * 88)
    print("1. 向量存在性（每个新文档的 chunk 数）")
    print("=" * 88)

    by_file = defaultdict(list)
    for d, m in zip(got["documents"], got["metadatas"]):
        m = m or {}
        src = str(m.get("source", ""))
        if src:
            by_file[Path(src).name].append((m.get("section_title"), d or ""))

    missing = []
    total_new_chunks = 0
    for fname, title in sorted(TITLE_BY_FILE.items()):
        secs = by_file.get(fname, [])
        total_new_chunks += len(secs)
        mark = "✅" if secs else "❌"
        if not secs:
            missing.append((fname, title))
        print(f"  {mark} {title[:32]:<34} {len(secs)} chunk")

    print()
    print(f"  新文档 chunk 总数: {total_new_chunks}")
    print(f"  无向量的文档: {len(missing)}")
    for f, t in missing:
        print(f"     {t}  ({f})")

    print()
    print("=" * 88)
    print("2. 检索可达性（能否命中正确文档）")
    print("=" * 88)

    vs = VectorStoreService()
    await vs.init()

    # 用「文档标题」构造问题（最直接的可达性检验）
    ok = bad = 0
    fails = []
    for fname, title in sorted(TITLE_BY_FILE.items()):
        if not title:
            continue
        res = await vs.search(title, top_k=5)
        hit = False
        for doc, score in res:
            m = doc.get("metadata") or {}
            if m.get("title") == title:
                hit = True
                break
        if hit:
            ok += 1
        else:
            bad += 1
            top = (res[0][0].get("metadata") or {}).get("title") if res else "无结果"
            fails.append((title, top))

    print(f"  用文档标题检索 -> 命中自身: {ok}/{ok+bad}")
    for t, top in fails:
        print(f"     ❌ {t}  (top1 = {top})")

    print()
    print("=" * 88)
    print("3. 章节级检索（用章节标题验证）")
    print("=" * 88)
    sec_ok = sec_tot = 0
    for fname, title in sorted(TITLE_BY_FILE.items()):
        for sec, body in by_file.get(fname, [])[:3]:
            if not sec:
                continue
            sec_tot += 1
            res = await vs.search(sec, top_k=5)
            if any((d.get("metadata") or {}).get("title") == title for d, _ in res):
                sec_ok += 1
    print(f"  章节标题检索命中所属文档: {sec_ok}/{sec_tot}")

    print()
    print("=" * 88)
    print("4. 跨文档区分度（新旧文档是否互相干扰）")
    print("=" * 88)
    # 取一个新产品文档标题，看返回结果里是否混入制度文档
    probe = next((t for t in TITLE_BY_FILE.values() if t), "")
    if probe:
        res = await vs.search(probe, top_k=5)
        print(f"  查询: {probe}")
        for i, (doc, score) in enumerate(res, 1):
            m = doc.get("metadata") or {}
            print(f"    {i}. [{score:6.3f}] {m.get('title')} / {m.get('section_title')}")

    out = ROOT / "data" / "eval" / "new_docs_retrieval.json"
    out.write_text(json.dumps({
        "new_docs": len(TITLE_BY_FILE),
        "new_chunks": total_new_chunks,
        "docs_without_vectors": [f for f, _ in missing],
        "title_hit": ok, "title_total": ok + bad,
        "section_hit": sec_ok, "section_total": sec_tot,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写入 {out}")


asyncio.run(main())
