"""检查向量库里实际存的 metadata 结构"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
sys.stdout.reconfigure(encoding="utf-8")

from services.vector_store import VectorStoreService

vs = VectorStoreService()
import asyncio


async def main():
    await vs.init()
    col = vs._store
    recs = col.get(include=["metadatas", "documents"], limit=8)

    print("=== 向量库 metadata 实际结构 ===")
    for cid, meta, doc in zip(recs["ids"], recs["metadatas"], recs["documents"]):
        print(f"  id={cid}")
        print(f"    metadata keys: {list(meta.keys())}")
        print(f"    title  = {meta.get('title')!r}")
        print(f"    source = {str(meta.get('source'))[-40:]!r}")
        print(f"    content= {doc[:50]!r}")
        print()


asyncio.run(main())
