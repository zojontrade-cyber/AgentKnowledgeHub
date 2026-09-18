"""
迁移：为已有 chunk 补充 searchable / section_type 元数据。

为什么需要它
------------
`searchable` 过滤逻辑只对**新入库**的块生效。生产库里 159 个 chunk
全部是本次变更前写入的，没有任何一个带 searchable 字段，
因此过滤不会起作用 —— 代码改了，行为不变。

为什么在原地更新而不是重跑入库
------------------------------
重跑入库会**重新调用 embedding API**（159 次请求 + 费用），
且会改变向量值，使「变更前/后」不可比。
原地只改 metadata，向量与文档内容完全不动，变量最小。

用法：
    python scripts/migrate_add_searchable.py            # 预演（不写入）
    python scripts/migrate_add_searchable.py --apply    # 实际写入
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

_PY = Path(Path(__file__).resolve().parent.parent / "python")
os.chdir(_PY)
sys.path.insert(0, str(_PY))
sys.stdout.reconfigure(encoding="utf-8")

import chromadb

from agents.doc_parser_agent import DocParserAgent
from config import settings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="实际写入（默认仅预演）")
    args = ap.parse_args()

    client = chromadb.PersistentClient(path=settings.chroma_path)
    col = client.get_or_create_collection("knowledge_chunks")

    got = col.get(include=["metadatas"])
    ids = got.get("ids") or []
    metas = got.get("metadatas") or []
    print(f"读取 {len(ids)} 个 chunk（chroma_path={settings.chroma_path}）")

    if not ids:
        print("集合为空，无需迁移")
        return

    new_metas = []
    kind_counter: Counter = Counter()
    changed = 0

    for cid, m in zip(ids, metas):
        m = dict(m or {})
        sec_title = m.get("section_title", "")

        # 优先用 classifier 判定；正文不可得时**(标题单独判定)**
        # 这里只用标题 —— 说明类章节的标题本身就是判据，
        # 而重取正文需要额外的 documents 读取，且回答阶段用
        # parent_content 已足够。
        sec_type = DocParserAgent.classify_section(sec_title, "")
        searchable = 0 if sec_type == "doc_overview" else 1

        kind_counter[sec_type] += 1
        if m.get("searchable") != searchable or m.get("section_type") != sec_type:
            changed += 1
        m["searchable"] = searchable
        m["section_type"] = sec_type
        new_metas.append(m)

    print()
    print("分类结果：")
    for k, v in kind_counter.most_common():
        print(f"   {k:<14} {v}")

    print()
    print(f"将被更新: {changed} / {len(ids)}")
    excluded = [i for i, m in zip(ids, new_metas) if not m["searchable"]]
    print(f"排除出召回(searchable=0): {len(excluded)}")
    for cid in excluded[:8]:
        print(f"   {cid}")

    if not args.apply:
        print()
        print("== 预演模式，未写入。加 --apply 实际执行 ==")
        return

    # 只更新 metadata，向量与 documents 保持不动
    col.update(ids=ids, metadatas=new_metas)
    print()
    print(f"已更新 {len(ids)} 条 metadata（向量未改动）")

    # 复核
    check = col.get(include=["metadatas"])
    n_search = sum(1 for m in check["metadatas"] if int((m or {}).get("searchable", 1)))
    n_excl = len(check["metadatas"]) - n_search
    print(f"复核：可召回 {n_search} / 已排除 {n_excl}")


if __name__ == "__main__":
    main()
