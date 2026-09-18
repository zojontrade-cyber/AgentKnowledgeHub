"""
向量存储服务 — 支持 ChromaDB（嵌入式/服务端）与 PGVector

职责:
  1. 文档块向量化 (Embedding)
  2. 向量存储 & 检索
  3. 按 doc_id 删除（用于文档更新时整篇替换）

ChromaDB 的两种运行方式
-----------------------
本服务支持 CHROMA_MODE 配置切换：

  persistent（默认，**无需 Docker**）
      嵌入式模式，数据落在本地目录 CHROMA_PATH。
      chromadb 本身是纯 Python 库，进程内运行，零外部依赖。
      适合本地开发与单机部署。

  http（需要独立服务端）
      连接 Docker 里跑的 Chroma 服务。
      适合多副本生产部署（多个 API 实例共享同一向量库）。

切换方式（.env）：
    CHROMA_MODE=persistent
    CHROMA_PATH=./chroma_data

    # 或
    CHROMA_MODE=http
    CHROMA_HOST=localhost
    CHROMA_PORT=8000
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_openai import OpenAIEmbeddings

from agents.doc_parser_agent import DocumentChunk
from config import settings
from services.bm25 import BM25Index

logger = logging.getLogger(__name__)


class VectorStoreService:
    """向量库统一接口，底层可切换 ChromaDB（persistent/http）/ PGVector"""

    COLLECTION_NAME = "knowledge_chunks"

    def __init__(self) -> None:
        # embeddings 延迟创建：OpenAIEmbeddings 在构造时就要求 api_key，
        # 若在此处立即实例化，则「未配置 Key」会直接导致服务无法启动，
        # 连不依赖 LLM 的存储层都跑不起来。改为首次使用时创建。
        self._embeddings: Any = None
        self._store: Any = None
        self._backend = settings.vector_store_type
        self._chroma_mode = settings.chroma_mode
        self._client: Any = None
        # BM25 主召回索引（懒构建，见 _ensure_bm25）
        self._bm25: BM25Index | None = None
        self._bm25_stale = True

    @property
    def embeddings(self) -> Any:
        """惰性构造 embeddings（首次访问时才校验凭据）

        使用 effective_* 属性，支持 embedding 与对话走不同服务商
        （例如对话用 DeepSeek、embedding 用硅基流动）。
        """
        if self._embeddings is None:
            key = settings.effective_embedding_api_key
            if not key:
                raise RuntimeError(
                    "未配置 embedding 凭据，无法生成向量。请在 .env 中设置 "
                    "EMBEDDING_API_KEY（或 OPENAI_API_KEY）。"
                )
            from langchain_openai import OpenAIEmbeddings

            self._embeddings = OpenAIEmbeddings(
                model=settings.embedding_model,
                api_key=key,
                base_url=settings.effective_embedding_base_url,
                timeout=settings.embedding_timeout_seconds,
            )
        return self._embeddings

    # ── initialization ───────────────────────────────────────

    async def init(self) -> None:
        if self._backend == "chroma":
            await self._init_chroma()
        else:
            await self._init_pgvector()

    async def _init_chroma(self) -> None:
        """初始化 Chroma。persistent 模式无需任何外部服务。"""
        import chromadb

        if self._chroma_mode == "persistent":
            # 嵌入式：数据落本地目录，进程内运行
            from pathlib import Path

            path = Path(settings.chroma_path)
            path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(path))
            logger.info(
                "Chroma 已启动（嵌入式模式）",
                extra={"extra_fields": {"path": str(path.resolve())}},
            )
        else:
            self._client = chromadb.HttpClient(
                host=settings.chroma_host, port=settings.chroma_port
            )
            logger.info(
                "Chroma 已连接（HTTP 服务端模式）",
                extra={"extra_fields": {
                    "host": settings.chroma_host,
                    "port": settings.chroma_port,
                }},
            )

        self._store = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    async def _init_pgvector(self) -> None:
        from langchain_community.vectorstores import PGVector

        self._store = PGVector(
            connection_string=settings.pgvector_dsn,
            collection_name=self.COLLECTION_NAME,
            embedding_function=self.embeddings,
        )

    async def reset_collection(self) -> None:
        """
        删除并重建 collection。

        为什么需要它：Chroma 的 collection 会**固定 embedding 维度**。
        更换 embedding 模型（如 1024 维的 bge-m3 换成 4096 维的
        Qwen3-Embedding-8B）后，即使清空了所有数据，旧 collection 仍
        期望 1024 维，写入会报：
            InvalidArgumentError: Collection expecting embedding with
            dimension of 1024, got 4096
        因此切换模型后必须删除整个 collection 重建。
        """
        if self._backend != "chroma" or self._client is None:
            return
        try:
            self._client.delete_collection(self.COLLECTION_NAME)
        except Exception:
            pass  # 不存在时忽略
        self._store = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._mark_bm25_stale()

    async def ping(self) -> bool:
        """轻量连通性探测，供 /api/health 使用"""
        try:
            if self._backend == "chroma":
                if self._store is None:
                    return False
                self._store.count()
                return True
            await self._store.asimilarity_search_with_score("ping", k=1)
            return True
        except Exception:
            return False

    # ── BM25 主召回 ──────────────────────────────────────────

    async def _ensure_bm25(self) -> BM25Index | None:
        """
        懒构建 BM25 索引（首次检索或写入后失效时重建）。

        为什么要重建而不是逐块维护：
          语料仅 159 章节，全量重建约 10ms，远低于维护逐块索引
          （删除、更新时的 df 回滚）的复杂度成本。
          语料上万后再改逐块维护。

        索引文本用 display（含「文档标题 > 章节标题」路径），因为
        BM25 依赖**词项字面命中** —— 用户问「差旅报销」时，
        "差旅报销管理规定" 这个标题本身就是最强信号。
        这与稠密检索相反：稠密要的是纯净正文，BM25 要的是完整字面。
        """
        if self._backend != "chroma" or self._store is None:
            return None

        if self._bm25 is not None and not self._bm25_stale:
            return self._bm25

        try:
            got = self._store.get(include=["documents", "metadatas"])
        except Exception:
            logger.warning("BM25 索引构建失败：无法读取 Chroma", exc_info=True)
            return None

        docs = got.get("documents") or []
        ids = got.get("ids") or []
        metas = got.get("metadatas") or []
        if not docs:
            logger.warning("BM25 索引为空：向量库无文档")
            return None

        # searchable 缺省视为 True：兼容本次变更前入库的旧数据
        # （旧 chunk 无该字段，但它们的 section_type 也已不可知）
        searchable = [
            bool((m or {}).get("searchable", 1)) for m in metas
        ]

        idx = self._bm25 or BM25Index()
        idx.build(
            [d or "" for d in docs],
            [str(i) for i in ids],
            searchable,
        )
        self._bm25 = idx
        self._bm25_stale = False

        excluded = sum(1 for x in searchable if not x)
        logger.info(
            "BM25 索引已构建",
            extra={"extra_fields": {
                "docs": idx.size,
                "searchable": idx.searchable_count,
                "excluded": excluded,
            }},
        )
        return idx

    def _mark_bm25_stale(self) -> None:
        """写入/删除后调用，下次检索时重建索引"""
        self._bm25_stale = True

    # ── CRUD ─────────────────────────────────────────────────

    async def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        """
        向量化并存储文档块。

        关键：**embedding 输入与展示文本分离**
          - embedding 用 c.content（仅「章节标题 + 章节正文」，无通用套话）
          - documents（Chroma 存的可读文本）用 c.display（带标题路径，便于引用展示）

        早期版本两者都用拼接了标题前缀的同一字符串，导致向量被
        "本制度规定…适用于…" 这类通用套话主导，出现
        「文档里明明有答案却检索不到」（实测含答案块 0.42 分，
        低于无关文档的 0.57 分）。
        """
        if not chunks:
            return 0

        # 进入 embedding 的文本：纯净、聚焦
        texts = [c.content for c in chunks]
        # 存储的可读文本：带标题路径，供引用展示
        displays = [c.display for c in chunks]
        ids = [c.chunk_id for c in chunks]

        # 注意：Chroma 的 metadata 只接受标量值，None 会报错，统一转空串。
        metadatas = [
            {
                "doc_id": c.doc_id,
                "doc_type": c.doc_type.value,
                "source": c.metadata.get("source", "") or "",
                "title": c.metadata.get("title", "") or "",
                "heading_path": c.metadata.get("heading_path", "") or "",
                "section_title": c.metadata.get("section_title", "") or "",
                "parent_id": c.metadata.get("parent_id", "") or "",
                "parent_content": c.metadata.get("parent_content", "") or "",
                "chunk_index": int(c.chunk_index),
            }
            for c in chunks
        ]

        if self._backend == "chroma":
            vectors = await self.embeddings.aembed_documents(texts)
            # upsert 语义：同 chunk_id 覆盖，天然幂等
            self._store.upsert(
                ids=ids, embeddings=vectors, documents=displays, metadatas=metadatas
            )
            self._mark_bm25_stale()
        else:
            await self._store.aadd_texts(texts=texts, metadatas=metadatas, ids=ids)

        return len(chunks)

    async def search(self, query: str, top_k: int = 5) -> list[tuple[dict, float]]:
        """
        语义搜索，返回 (文档, 分数) 列表。

        RETRIEVAL_MODE 控制走哪条通道（默认 bm25）：
          bm25    —— 主召回。实测 Chunk R@5 89.1%（见 services/bm25.py 头注释）
          dense   —— 旧行为，保留用于对照实验与回退
          hybrid  —— BM25 + 稠密 RRF 融合。**实测低于纯 BM25**，
                     仅为兼容保留，不建议在生产启用
        """
        mode = (settings.retrieval_mode or "bm25").lower()

        if mode == "bm25":
            return await self._search_bm25(query, top_k)
        if mode == "hybrid":
            return await self._search_hybrid(query, top_k)
        return await self._search_dense(query, top_k)

    async def _search_bm25(self, query: str, top_k: int = 5) -> list[tuple[dict, float]]:
        """BM25 字面召回 —— 主通道"""
        if self._backend != "chroma" or self._store is None:
            return []

        idx = await self._ensure_bm25()
        if idx is None or idx.size == 0:
            # 索引不可用时回退稠密，保证可用性优先
            logger.warning("BM25 不可用，回退稠密检索")
            return await self._search_dense(query, top_k)

        ranked = idx.rank_ids(query, top_k=top_k)
        ids = [cid for cid, _ in ranked]
        if not ids:
            return []

        try:
            got = self._store.get(
                ids=ids, include=["documents", "metadatas"]
            )
        except Exception:
            logger.warning("BM25 取回文档失败", exc_info=True)
            return []

        # Chroma 的 get 不保证返回顺序，按 BM25 名次重排
        by_id = {
            str(cid): (doc, meta)
            for cid, doc, meta in zip(
                got.get("ids", []),
                got.get("documents", []),
                got.get("metadatas", []),
            )
        }
        out: list[tuple[dict, float]] = []
        for cid, score in ranked:
            pair = by_id.get(str(cid))
            if not pair:
                continue
            doc, meta = pair
            out.append((
                {
                    "content": doc,
                    "source": (meta or {}).get("source", ""),
                    "metadata": meta or {},
                },
                float(score),
            ))
        return out

    async def _search_hybrid(self, query: str, top_k: int = 5) -> list[tuple[dict, float]]:
        """
        BM25 + 稠密 RRF 融合。

        警告：实测该模式 **低于纯 BM25**（Chunk R@5 50.9% vs 89.1%，
        权重偏向 BM25 时 70.9% 仍不及）。原因是稠密一路对正确答案
        经常完全排不进候选（rank>50），却给大量无关项投票，RRF 奖励
        「两路都还行」而非「一路极好」。保留仅为对照实验。
        """
        bm = await self._search_bm25(query, top_k=top_k * 4)
        dn = await self._search_dense(query, top_k=top_k * 4)

        # 用 chunk_id 作为跨通道的稳定主键
        def key_of(item: tuple[dict, float]) -> str:
            doc, _ = item
            meta = doc.get("metadata") or {}
            # 真实 metadata 无 chunk_id 字段，用 doc_id + chunk_index 组合
            did = meta.get("doc_id")
            ci = meta.get("chunk_index")
            if did is not None and ci is not None:
                return f"{did}#{ci}"
            return str(doc.get("content", ""))[:120]

        rrf: dict[str, float] = {}
        payload: dict[str, tuple[dict, float]] = {}
        for lst, weight in ((bm, settings.hybrid_bm25_weight), (dn, 1.0 - settings.hybrid_bm25_weight)):
            for rank, item in enumerate(lst, 1):
                k = key_of(item)
                rrf[k] = rrf.get(k, 0.0) + weight / (60 + rank)
                payload.setdefault(k, item)

        ordered = sorted(rrf.items(), key=lambda x: (-x[1], x[0]))[:top_k]
        return [(payload[k][0], score) for k, score in ordered]

    async def _search_dense(self, query: str, top_k: int = 5) -> list[tuple[dict, float]]:
        """稠密向量检索（旧主通道，现降级为对照/回退）"""
        if self._backend == "chroma":
            q_vec = await self.embeddings.aembed_query(query)
            # 过滤必须在查询阶段下推，不能取回 top_k 后再筛 ——
            # 否则被排除的 doc_overview 块先占满名额，过滤后结果不足 k 条。
            # 这里按比例放大 n_results 作为**粗筛**，再在下方精确过滤。
            fetch_k = max(top_k * 3, top_k + 20)
            results = self._store.query(
                query_embeddings=[q_vec],
                n_results=fetch_k,
                include=["documents", "metadatas", "distances"],
            )
            out: list[tuple[dict, float]] = []
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            dists = results.get("distances", [[]])[0]
            for doc, meta, dist in zip(docs, metas, dists):
                meta = meta or {}
                if not int(meta.get("searchable", 1)):
                    continue
                score = 1.0 - dist  # cosine distance → similarity
                out.append((
                    {
                        "content": doc,
                        "source": meta.get("source", ""),
                        "metadata": meta,
                    },
                    score,
                ))
                if len(out) >= top_k:
                    break
            return out
        else:
            results = await self._store.asimilarity_search_with_score(query, k=top_k)
            return [
                (
                    {
                        "content": doc.page_content,
                        "source": doc.metadata.get("source", ""),
                        "metadata": doc.metadata,
                    },
                    score,
                )
                for doc, score in results
            ]

    async def delete_by_doc_id(self, doc_id: str) -> int:
        """按 doc_id 删除所有相关向量"""
        if self._backend != "chroma":
            return 0

        # 注意：Chroma 的 get 默认有返回上限，大文档可能删不干净。
        # 显式传入足够大的 limit 并循环直到取空，避免残留旧向量。
        deleted = 0
        while True:
            existing = self._store.get(where={"doc_id": doc_id}, include=[], limit=1000)
            ids = existing.get("ids", [])
            if not ids:
                break
            self._store.delete(ids=ids)
            deleted += len(ids)
        if deleted:
            self._mark_bm25_stale()
        return deleted

    async def get_stats(self) -> dict:
        """获取向量库统计信息"""
        if self._backend == "chroma":
            count = self._store.count()
            return {
                "backend": "chroma",
                "mode": self._chroma_mode,
                "total_vectors": count,
                "collection": self.COLLECTION_NAME,
            }
        return {"backend": "pgvector", "collection": self.COLLECTION_NAME}
