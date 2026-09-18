"""
重排序服务 —— 用 Reranker 模型对向量召回结果精排

为什么需要
----------
纯向量检索的问题是「召回可以、排序不行」：
  - 它把所有文本压成一个向量，细粒度的语义差异被抹平
  - 实测你的 24 篇「XX管理制度」文档分数全部挤在 0.53-0.69，
    不同主题差异不到 0.1，导致正确答案经常排在无关文档之后

标准解法是两阶段检索（这也是生产级 RAG 的标配）：
  第一阶段：向量/关键词粗召回 top-N（N=20~50，追求召回率）
  第二阶段：Reranker 对 (query, doc) 逐对精算相关性，重排取 top-K

Reranker 是 cross-encoder：它把 query 和 doc 拼在一起过模型，
能捕捉两者之间的交互特征，因此精度远高于双塔式的向量相似度。

本项目用硅基流动的 BAAI/bge-reranker-v2-m3（与 embedding 同一 Key）。
"""

from __future__ import annotations

import logging
from typing import Any

from config import settings

logger = logging.getLogger(__name__)

# 硅基流动的 rerank 端点路径
_RERANK_PATH = "/rerank"


class RerankerService:
    """
    调用兼容 Cohere/Jina 风格的 /rerank 接口。

    注意：不同服务商的请求/响应结构略有差异，这里以硅基流动为准，
    并做了容错解析。
    """

    def __init__(self) -> None:
        # 三重条件缺一不可：
        #   1. rerank_enabled —— 显式开关（可临时关停，无需删配置）
        #   2. rerank_model   —— 必须配置了模型
        #   3. API Key        —— 没有凭据调用必然失败
        self._enabled = (
            bool(settings.rerank_enabled)
            and bool(settings.rerank_model)
            and bool(settings.effective_embedding_api_key)
        )
        self._client: Any = None
        # 连续失败计数 —— 用于在降级时给出**显式**告警，
        # 避免 reranker 静默失效导致排序质量悄悄退回基线。
        self._fail_streak = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def fail_streak(self) -> int:
        return self._fail_streak

    @property
    def client(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(
                base_url=settings.effective_embedding_base_url,
                headers={
                    "Authorization": f"Bearer {settings.effective_embedding_api_key}",
                    "Content-Type": "application/json",
                },
                timeout=settings.rerank_timeout_seconds,
            )
        return self._client

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """
        对文档列表按与 query 的相关性重排。

        返回 [(原始索引, 相关性分数), ...]，按分数降序，长度 <= top_k。
        失败时返回空列表，调用方应回退到原始顺序（不能让 rerank 故障
        导致整个问答失败）。
        """
        if not self._enabled or not documents:
            return []

        try:
            resp = await self.client.post(
                _RERANK_PATH,
                json={
                    "model": settings.rerank_model,
                    "query": query,
                    "documents": documents,
                    "top_n": min(top_k, len(documents)),
                    "return_documents": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            self._fail_streak = 0
        except Exception:
            self._fail_streak += 1
            # 首次失败用 warning；连续失败升级为 error，因为此时
            # 排序质量已静默退回 BM25 基线（Chunk@1 76.3% -> 差距 10.5pp）
            log = logger.error if self._fail_streak >= 3 else logger.warning
            log(
                "Rerank 调用失败，回退到原始排序（连续失败 %d 次）",
                self._fail_streak,
                exc_info=True,
            )
            return []

        # 兼容两种常见响应结构
        results = data.get("results")
        if results is None and "data" in data:
            results = data["data"]
        if not isinstance(results, list):
            logger.warning("Rerank 响应格式无法识别: %s", str(data)[:200])
            return []

        out: list[tuple[int, float]] = []
        for item in results:
            idx = item.get("index")
            score = item.get("relevance_score", item.get("score", 0.0))
            if idx is None:
                continue
            out.append((int(idx), float(score)))

        out.sort(key=lambda x: x[1], reverse=True)
        return out[:top_k]

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


# 模块级单例（供 qa_agent 复用，避免每次问答都新建连接）
reranker = RerankerService()
