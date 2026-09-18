"""
BM25 稀疏检索 —— 企业制度知识库的主召回通道

为什么需要它（实测数据，见 data/eval/p1_results.json）
------------------------------------------------------
在 159 个制度章节、78 题文档级 / 55 题 chunk 级评测集上（retrieval-only，无 reranker）：

    方法          Doc R@5    Chunk R@5   Chunk MRR
    Dense         59.0%      23.6%       0.140
    BM25          84.6%      89.1%       0.594
    RRF(等权)     80.8%      50.9%       0.348
    RRF(0.3:0.7)  84.6%      70.9%       0.475

结论：
  1. 稠密检索在本语料上 Chunk R@5 仅 23.6%，**基本不可用**
  2. 纯 BM25 达 89.1%，且零 API 成本、零外部依赖
  3. 任何形式的 RRF 融合都**低于纯 BM25** —— 稠密一路给正确答案投 0 票，
     却给大量无关项投票，融合反而拉低排序

根因：制度语料含大量判别性极强的低频词项（"请假"、"年假"、"保密"、
"印章"、"报销"）。BM25 视其为高 IDF 强信号；稠密模型把 4096 维空间
"平均化"后，这些词项的信号被稀释殆尽。

分词策略：中文无空格，这里用「字符 bigram + 单字」而非 jieba：
  - 零依赖（项目已禁 Docker / 尽量少装包）
  - 对未登录词（"云启科技"、"OA 系统"）鲁棒
  - 实测 Chunk R@5 89.1%，无需引入分词器
"""

from __future__ import annotations

import logging
import math
import re
import threading
import unicodedata
from collections import Counter

logger = logging.getLogger(__name__)


def tokenize(text: str) -> list[str]:
    """
    中文分词：字符 bigram + 单字。

    为什么同时要 unigram 和 bigram：
      - bigram 提供区分度（"请假" 比 "请"、"假" 更有信息量）
      - unigram 保证召回（query 里的单字也能命中长文档中的词）
    """
    t = unicodedata.normalize("NFKC", text or "")
    # 保留中日韩文字、字母、数字；丢弃标点与空白
    t = re.sub(r"[^\w\u4e00-\u9fff]+", "", t)
    if not t:
        return []
    tokens = [t[i : i + 2] for i in range(len(t) - 1)]
    tokens.extend(t)
    return tokens


class BM25Index:
    """
    内存 BM25 索引。

    线程安全：build/rank 通过 RLock 串行化。
    当前是**全量内存**实现 —— 159 个章节约 3 万 token，占用可忽略。
    语料增长到 10 万级时应换成磁盘索引（如 rank_bm25 + 持久化词表）。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._lock = threading.RLock()

        self._doc_tokens: list[list[str]] = []
        self._tf: list[Counter] = []
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0
        self._n: int = 0
        # 原始文本，便于调试与一致性校验
        self._raw: list[str] = []
        # 外部主键（Chroma 的 chunk_id），下标与 _doc_tokens 一一对应。
        # 必须存真实 ID —— 早期实现用位置下标去 Chroma get(id=str(i))，
        # 而真实 ID 形如 "89d0959c15cea16c#chunk-0"，导致取回永远为空。
        self._ids: list[str] = []
        # 是否参与召回。False 的文档**仍然建索引**（以便 parent 展开时
        # 可取回），但 rank/rank_ids 会跳过它们。
        # 关键：过滤必须在**打分阶段**生效，不能在 top-k 之后过滤 ——
        # 否则被排除的块仍占满 top-k 名额，等于没排除。
        self._searchable: list[bool] = []

    # ── 构建 ──────────────────────────────────────────────

    def build(
        self,
        texts: list[str],
        ids: list[str] | None = None,
        searchable: list[bool] | None = None,
    ) -> None:
        """(重)建索引。O(N * L)，159 篇约 10ms。

        ids 为外部主键（Chroma chunk_id），与 texts 等长且顺序一致。
        传入后 rank_ids() 可直接返回真实 ID，避免位置下标的映射错误。

        searchable 与 texts 等长；False 的项仍进入词表统计（保证 IDF
        反映真实语料分布），但不会被 rank 返回。
        """
        with self._lock:
            self._raw = list(texts)
            self._ids = list(ids) if ids is not None else [str(i) for i in range(len(texts))]
            if len(self._ids) != len(texts):
                raise ValueError(
                    f"ids 与 texts 长度不一致: {len(self._ids)} != {len(texts)}"
                )
            self._searchable = (
                list(searchable) if searchable is not None else [True] * len(texts)
            )
            if len(self._searchable) != len(texts):
                raise ValueError(
                    f"searchable 与 texts 长度不一致: "
                    f"{len(self._searchable)} != {len(texts)}"
                )
            self._doc_tokens = [tokenize(t) for t in texts]
            self._tf = [Counter(d) for d in self._doc_tokens]
            self._n = len(self._doc_tokens)
            self._avgdl = (
                sum(len(d) for d in self._doc_tokens) / self._n if self._n else 0.0
            )

            df: Counter = Counter()
            for d in self._doc_tokens:
                for w in set(d):
                    df[w] += 1
            self._idf = {
                w: math.log(1.0 + (self._n - c + 0.5) / (c + 0.5))
                for w, c in df.items()
            }
            logger.info(
                "BM25 索引已构建",
                extra={"extra_fields": {
                    "docs": self._n,
                    "vocab": len(self._idf),
                    "avgdl": round(self._avgdl, 1),
                }},
            )

    @property
    def size(self) -> int:
        with self._lock:
            return self._n

    # ── 查询 ──────────────────────────────────────────────

    def score(self, query: str, idx: int) -> float:
        q = tokenize(query)
        if not q:
            return 0.0
        with self._lock:
            if idx >= self._n:
                return 0.0
            tf = self._tf[idx]
            dl = len(self._doc_tokens[idx])
            s = 0.0
            for w in q:
                f = tf.get(w)
                if not f:
                    continue
                idf = self._idf.get(w, 0.0)
                s += idf * f * (self.k1 + 1.0) / (
                    f + self.k1 * (1.0 - self.b + self.b * dl / max(self._avgdl, 1e-9))
                )
            return s

    def rank(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """
        返回 [(位置下标, 分数)]，按分数降序。**跳过 searchable=False 的项。**

        排序次键用 idx 保证**结果可复现**（同分时顺序稳定），
        避免因排序不稳定导致评测结果抖动。
        """
        with self._lock:
            n = self._n
            searchable = list(self._searchable)
        if n == 0:
            return []
        scored = [
            (i, self.score(query, i))
            for i in range(n)
            if i < len(searchable) and searchable[i]
        ]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:top_k]

    def rank_ids(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """返回 [(外部主键, 分数)]。检索层应优先用这个，避免下标映射。"""
        with self._lock:
            ids = list(self._ids)
        return [(ids[i], s) for i, s in self.rank(query, top_k) if i < len(ids)]

    @property
    def searchable_count(self) -> int:
        """参与召回的文档数（不含被排除的 doc_overview）"""
        with self._lock:
            return sum(1 for x in self._searchable if x)
