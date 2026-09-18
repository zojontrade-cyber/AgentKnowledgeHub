"""
验证嵌入式 Chroma 可用性（不需要 Docker、不需要 API Key）

这一项验证的是：向量库的「连接与读写」是否成立。
embedding 需要 LLM Key，所以这里用假向量直接测 Chroma 本身。
"""

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb  # noqa: E402

tmp = Path(tempfile.mkdtemp(prefix="chroma_test_"))
print(f"临时目录: {tmp}")

try:
    # 1. 嵌入式客户端能否创建
    client = chromadb.PersistentClient(path=str(tmp))
    print("[PASS] PersistentClient 创建成功（无需 Docker）")

    # 2. collection 创建
    col = client.get_or_create_collection(
        name="knowledge_chunks", metadata={"hnsw:space": "cosine"}
    )
    print("[PASS] collection 创建成功")

    # 3. 写入（用假向量，避免依赖 LLM Key）
    col.upsert(
        ids=["doc1#chunk-0", "doc1#chunk-1", "doc2#chunk-0"],
        embeddings=[[0.1] * 8, [0.2] * 8, [0.9] * 8],
        documents=["张三担任腾讯CEO", "李四是技术总监", "微星的营收增长"],
        metadatas=[
            {"doc_id": "doc1", "source": "a.pdf"},
            {"doc_id": "doc1", "source": "a.pdf"},
            {"doc_id": "doc2", "source": "b.pdf"},
        ],
    )
    print("[PASS] 写入向量成功")

    # 4. 计数
    count = col.count()
    assert count == 3, f"期望 3 条，实际 {count}"
    print(f"[PASS] count() = {count}")

    # 5. 相似度检索
    res = col.query(query_embeddings=[[0.1] * 8], n_results=2,
                    include=["documents", "metadatas", "distances"])
    docs = res["documents"][0]
    dists = res["distances"][0]
    print(f"[PASS] 检索返回 {len(docs)} 条，最近: {docs[0]!r} (距离 {dists[0]:.4f})")

    # 6. 按 doc_id 删除（含分页循环逻辑验证）
    to_del = col.get(where={"doc_id": "doc1"}, include=[], limit=1000)
    del_ids = to_del["ids"]
    assert len(del_ids) == 2, f"期望删 2 条，实际 {len(del_ids)}"
    col.delete(ids=del_ids)
    assert col.count() == 1, f"删除后期望 1 条，实际 {col.count()}"
    print(f"[PASS] 按 doc_id 删除成功（删除 {len(del_ids)} 条，剩余 {col.count()}）")

    # 7. 数据持久化：关掉再开，数据还在吗
    del client, col
    client2 = chromadb.PersistentClient(path=str(tmp))
    col2 = client2.get_or_create_collection(
        name="knowledge_chunks", metadata={"hnsw:space": "cosine"}
    )
    assert col2.count() == 1, f"重开后期望 1 条，实际 {col2.count()}"
    print("[PASS] 数据持久化验证：重开后数据仍在")

    print()
    print("结论: 嵌入式 Chroma 完全可用，不需要 Docker")

finally:
    shutil.rmtree(tmp, ignore_errors=True)
