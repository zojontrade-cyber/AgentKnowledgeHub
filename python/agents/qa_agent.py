"""
问答 Agent — 意图驱动的检索 + 答案生成

检索链路（当前实际生效的）
--------------------------
    query
      → _classify_intent   意图分类（function calling 强制枚举，保证取值合法）
      → _rewrite_query     查询改写（一个问题 → 多个查询变体）
      → _retrieve_by_plan  BM25 多查询并发召回
      → _rerank_contexts   cross-encoder 精排
      → _generate_answer   生成答案（稳定前缀 Prompt，见 services/prompt_builder.py）

为什么用 BM25 而不是稠密向量：实测稠密检索 Chunk Recall@5 仅 23.6%，
换 BM25 后 89.1%。详见 docs/retrieval-bm25-migration.md。

关于 INTENT_RETRIEVAL_PLAN
--------------------------
该映射表只有 `vector` 一路真正生效 —— `_retrieve_by_plan` **只读
`plan["vector"]`**，因此五种意图当前产出**完全相同**的检索动作。
意图只通过生成段的 `<request_metadata>` 影响**答案组织方式**。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from config import settings
from services.llm_factory import LazyLLM
from services.prompt_builder import (
    PromptSource,
    build_answer_messages,
    build_legacy_answer_messages,
)
from services.reranker import reranker
from services.usage_meter import record_openai, record_prompt_sections, stage

logger = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    FACTOID = "factoid"           # 事实型问题
    ANALYTICAL = "analytical"     # 分析型问题
    COMPARATIVE = "comparative"   # 对比型问题
    PROCEDURAL = "procedural"     # 流程型问题
    EXPLORATORY = "exploratory"   # 探索型问题


class _IntentSchema(BaseModel):
    """结构化意图输出 —— 由 with_structured_output 强制约束"""

    intent: QueryIntent = Field(
        description="问题意图类别，必须是给定的枚举值之一"
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="分类置信度 0-1"
    )
    reason: str = Field(default="", description="一句话说明判断依据")


@dataclass
class RetrievedContext:
    """
    检索结果的三层数据契约。

    这三层**必须分离**，早期把它们混在一个 `content` 字段里，
    导致了一个用户可见的数据契约错误：

        retrieved_chunk  —— 命中的那个章节（排序依据，也是**用户看到的证据**）
        llm_context      —— 展开后的父文档（喂给 LLM，信息更全）
        display          —— 序列化给前端的引用

    原实现的错误链路：
        BM25 命中 A 章节
          -> _expand_small_to_big 把 content 替换成 parent_content
          -> LLM 拿到全文（正确）
          -> sources 也返回 parent_content（**错误**）
    后果：sources[0].content 永远是"## 1. 文档说明 …" —— 答案可能正确，
    但用户核对的引用看起来是错的。实测引用准确率仅 44.4%。

    现在：
        content       保持为**命中章节**（展示与核对用）
        llm_context   展开后的父文档（仅供生成，不返回给用户）
    """

    content: str
    source: str
    score: float
    retrieval_type: str  # 当前只有 "vector"
    metadata: dict[str, Any] = field(default_factory=dict)
    # 供 LLM 使用的展开上下文（父文档全文）。为空表示无需展开。
    # 刻意**不**参与 __repr__/序列化，避免误传给前端。
    llm_context: str = ""

    @property
    def display_content(self) -> str:
        """返回给前端的展示文本 = 命中章节本身"""
        return self.content

    @property
    def title(self) -> str:
        return str((self.metadata or {}).get("title") or "")

    @property
    def section(self) -> str:
        return str((self.metadata or {}).get("section_title") or "")

    @property
    def parent_available(self) -> bool:
        """是否存在可展开的父文档（供前端提示"可查看完整制度"）"""
        return bool(self.llm_context)


@dataclass
class QAResult:
    question: str
    answer: str
    contexts: list[RetrievedContext]
    intent: QueryIntent
    confidence: float
    reasoning_steps: list[str] = field(default_factory=list)


INTENT_PROMPT = """\
你是一个查询意图分类器。根据用户问题，判断其意图类别。

类别定义：
- factoid: 事实型（谁/什么/哪里/何时，单一事实点）
- analytical: 分析型（为什么/怎么理解，需要推理）
- comparative: 对比型（A和B有什么区别/关系，涉及多个实体）
- procedural: 流程型（怎么做/步骤/流程）
- exploratory: 探索型（有哪些/概述/整体情况）

只输出 intent 字段，不要解释你的判断过程。"""

QUERY_REWRITE_PROMPT = """\
你是一个查询改写专家。将用户问题改写为更适合检索的形式。
要求：
1. 提取核心实体和关键词
2. 生成 1-3 个检索查询
3. 返回 JSON: {"queries": ["查询1", "查询2"], "entities": ["实体1"], "keywords": ["关键词1"]}
只返回 JSON，不要其他文字。
"""

ENTITY_LINKING_PROMPT = """\
从以下问题中提取所有可能的实体名称（人名、组织、技术、产品、概念等）。
只返回实体名称本身，不要生成任何查询语句或代码。
返回 JSON: {"entities": ["实体1", "实体2"]}
只返回 JSON。
"""

ANSWER_PROMPT = """\
你是一个专业的企业知识问答助手。根据检索到的上下文信息回答用户问题。

要求：
1. 答案必须基于提供的上下文，不要编造
2. 如果上下文信息不足，明确告知用户
3. 引用信息来源（如 [来源: xxx]）
4. 如果涉及多个信息源，综合分析后给出结论
5. 保持专业、准确、简洁
"""

# 注（2026-09）：生成用的 Prompt 已迁到 services/prompt_builder.py：
#   ANSWER_SYSTEM_PROMPT —— 稳定 system 段，**所有问题逐字节相同**（不要插值）
#   build_answer_messages() —— 动态段，顺序固定 metadata → context → query
# 原因：旧的 `ANSWER_PROMPT + "本题意图为「x」，…"` 把意图写进 system，
# 使不同意图的问题从 ~149 字符处就分叉，自己打断了自己的前缀复用
# （实测见 docs/chat-metrics.md 5.1）。
# 下面两个常量仅保留给旧脚本引用，`_generate_answer` 已不再使用。


# 意图 → 检索策略。
#
# ⚠️ 实际只有 `vector` 一路生效（`_retrieve_by_plan` 只读 `plan["vector"]`），
# 因此五种意图当前产出**完全相同**的检索动作。
# 保留这张表是为了将来新增检索通道时留出位置 ——
# 但它**不是**"意图驱动检索策略"的证据，对外不要这么描述。
INTENT_RETRIEVAL_PLAN: dict[QueryIntent, dict[str, Any]] = {
    QueryIntent.FACTOID: {"vector": True},
    QueryIntent.ANALYTICAL: {"vector": True},
    QueryIntent.COMPARATIVE: {"vector": True},
    QueryIntent.PROCEDURAL: {"vector": True},
    QueryIntent.EXPLORATORY: {"vector": True},
}

ANSWER_STYLE: dict[QueryIntent, str] = {
    QueryIntent.FACTOID: "直接给出事实要点，不要展开。",
    QueryIntent.ANALYTICAL: "先给结论，再分点说明推理依据。",
    QueryIntent.COMPARATIVE: "用对比结构回答，明确指出两者的异同。",
    QueryIntent.PROCEDURAL: "按步骤或流程顺序组织答案。",
    QueryIntent.EXPLORATORY: "先给整体概览，再列出关键要点。",
}


class QAAgent:
    """
    问答 Agent（意图驱动）

    工作流:
      query → intent_classify → rewrite → plan → parallel_retrieve
            → rerank → generate_answer
    """

    def __init__(self, vector_store: Any = None) -> None:
        self.llm = LazyLLM(temperature=0)
        self.vector_store = vector_store
        # 数据级权限范围（None = 不限制）
        self._acl_scopes: list[str] | None = None

    # ── public API ───────────────────────────────────────────

    async def answer(
        self, question: str, acl_scopes: list[str] | None = None
    ) -> QAResult:
        """
        完整问答流程。

        参数 acl_scopes —— 数据级权限范围（None 表示不限制，即管理员）。
        取值如 ["public", "hr"]。检索时只返回该范围内的文档，
        实现「用户只能看到自己有权访问的知识」。
        """
        self._acl_scopes = acl_scopes
        intent, intent_conf = await self._classify_intent(question)
        rewritten = await self._rewrite_query(question)

        # 意图决定检索计划 —— 这一步是重构的核心
        plan = INTENT_RETRIEVAL_PLAN[intent]

        contexts = await self._retrieve_by_plan(question, rewritten, plan)

        # 两阶段检索的第二阶段：cross-encoder 精排。
        # 阶段一（上面的向量召回）追求召回率，排序质量有限；
        # 阶段二用 Reranker 对 (query, doc) 逐对精算，把真正相关的排到前面。
        rerank_note = ""
        if contexts:
            before = contexts[0].content[:40]
            contexts = await self._rerank_contexts(question, contexts)
            if contexts and contexts[0].content[:40] != before:
                rerank_note = "（已重排）"

        top_contexts = contexts[:8]
        answer_text, reasoning = await self._generate_answer(
            question, top_contexts, intent
        )

        reasoning.insert(0, f"识别问题意图: {intent.value} (置信度 {intent_conf:.2f})")
        reasoning.insert(1, f"检索策略: {self._describe_plan(plan)}{rerank_note}")

        return QAResult(
            question=question,
            answer=answer_text,
            contexts=top_contexts,
            intent=intent,
            confidence=self._calc_confidence(top_contexts),
            reasoning_steps=reasoning,
        )

    # ── intent classification ────────────────────────────────

    @stage("intent")
    async def _classify_intent(self, question: str) -> tuple[QueryIntent, float]:
        """
        意图分类 —— 用原生 function calling 强制枚举取值。

        为什么不用 with_structured_output 的默认实现：
          多数兼容接口（实测 DeepSeek、部分国产模型）不支持
          `response_format`（json_schema），调用会返回
          400 "This response_format type is unavailable"。
          而改用 `method="json_mode"` 时，实测模型会把 intent 填成自由文本
          （如 "比较人物差异"），而非枚举值 -> QueryIntent(...) 会抛异常。

        因此这里直接走 tool/function calling 并显式声明 enum，
        由服务端在解码阶段约束取值，这是唯一能保证合法的方式。
        """
        try:
            import json as _json

            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                timeout=settings.llm_timeout_seconds,
                max_retries=settings.llm_max_retries,
            )
            resp = await client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": INTENT_PROMPT},
                    {"role": "user", "content": question},
                ],
                temperature=0,
                tools=[{
                    "type": "function",
                    "function": {
                        "name": "classify_intent",
                        "description": "对用户问题做意图分类",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "intent": {
                                    "type": "string",
                                    "enum": [i.value for i in QueryIntent],
                                    "description": "问题意图类别",
                                },
                                "confidence": {
                                    "type": "number",
                                    "description": "分类置信度 0-1",
                                },
                            },
                            "required": ["intent"],
                        },
                    },
                }],
                tool_choice={
                    "type": "function",
                    "function": {"name": "classify_intent"},
                },
            )
            # 计量：这次调用走原生 client（不是 LazyLLM），需显式记录。
            # stage 由 @stage("intent") 装饰器提供。
            record_openai(resp)
            tool_calls = resp.choices[0].message.tool_calls
            if tool_calls:
                args = _json.loads(tool_calls[0].function.arguments)
                intent = QueryIntent(args["intent"])
                return intent, float(args.get("confidence", 0.0))
            logger.warning("意图分类未返回 tool_calls，降级为规则匹配")
        except Exception:
            logger.warning("意图分类调用失败，降级为规则匹配", exc_info=True)

        return self._classify_intent_fallback(question), 0.0

    @staticmethod
    def _classify_intent_fallback(question: str) -> QueryIntent:
        """
        规则兜底 —— 仅在结构化输出失败时使用。

        比旧实现的"默认 FACTOID"更有信息量：用关键词做粗判，识别不出才回落
        factoid。
        """
        q = question.lower()
        rules: list[tuple[QueryIntent, tuple[str, ...]]] = [
            (QueryIntent.COMPARATIVE, ("区别", "对比", "相比", "差异", "vs", "关系")),
            (QueryIntent.PROCEDURAL, ("怎么", "如何", "步骤", "流程", "怎样")),
            (QueryIntent.EXPLORATORY, ("有哪些", "概述", "整体", "全部", "总结")),
            (QueryIntent.ANALYTICAL, ("为什么", "原因", "分析", "如何理解")),
        ]
        for intent, keywords in rules:
            if any(kw in q for kw in keywords):
                return intent
        return QueryIntent.FACTOID

    # ── query rewriting ──────────────────────────────────────

    @stage("rewrite")
    async def _rewrite_query(self, question: str) -> dict:
        import json

        messages = [
            SystemMessage(content=QUERY_REWRITE_PROMPT),
            HumanMessage(content=question),
        ]
        try:
            resp = await self.llm.ainvoke(messages)
            cleaned = resp.content.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(cleaned)
            if not isinstance(data, dict):
                raise ValueError("改写结果不是对象")
            return data
        except Exception:
            logger.warning("查询改写失败，使用原问题", exc_info=True)
            return {"queries": [question], "entities": [], "keywords": []}

    # ── retrieval dispatcher ─────────────────────────────────

    async def _retrieve_by_plan(
        self, question: str, rewritten: dict, plan: dict[str, Any]
    ) -> list[RetrievedContext]:
        """
        按检索计划调度检索器，再统一重排序。

        当前只有 BM25 一路。保留 plan 结构是为了将来新增检索通道时留位置。
        """
        tasks: list[Any] = []
        labels: list[str] = []

        if plan.get("vector"):
            tasks.append(self._vector_retrieve(rewritten))
            labels.append("vector")

        if not tasks:
            return []

        results = await asyncio.gather(*tasks, return_exceptions=True)

        merged: list[RetrievedContext] = []
        for label, res in zip(labels, results):
            if isinstance(res, Exception):
                # 单个检索器失败不应导致整个问答失败，但必须留痕
                logger.warning("检索器 %s 失败: %s", label, res, exc_info=res)
                continue
            merged.extend(res)

        return self._hybrid_rerank(merged)

    @staticmethod
    def _describe_plan(plan: dict[str, Any]) -> str:
        parts = []
        if plan.get("vector"):
            parts.append("BM25 检索")
        return " + ".join(parts) or "无"

    # ── 两阶段检索的第二阶段：Rerank ──────────────────────────

    async def _rerank_contexts(
        self, question: str, contexts: list[RetrievedContext]
    ) -> list[RetrievedContext]:
        """
        对召回结果做 cross-encoder 精排。

        - 只对候选集的前 rerank_candidates 条重排，控制延迟
        - Reranker 不可用/调用失败时**原样返回**，不影响主链路可用性
        """
        if not reranker.enabled or not contexts:
            return contexts

        candidates = contexts[: settings.rerank_candidates]
        rest = contexts[settings.rerank_candidates :]
        if len(candidates) < 2:
            return contexts

        docs = [c.content for c in candidates]
        try:
            ranked = await reranker.rerank(question, docs, top_k=len(candidates))
        except Exception:
            logger.warning("Rerank 异常，使用原顺序", exc_info=True)
            return contexts

        if not ranked:
            return contexts

        reordered: list[RetrievedContext] = []
        seen_idx: set[int] = set()
        for idx, score in ranked:
            if idx >= len(candidates) or idx in seen_idx:
                continue
            seen_idx.add(idx)
            ctx = candidates[idx]
            # 保留原始分以供对比，把 rerank 分写入 score
            ctx.metadata = {**(ctx.metadata or {}), "pre_rerank_score": ctx.score}
            ctx.score = score
            reordered.append(ctx)

        # 未进入 rerank 结果的部分按原顺序接在后面
        for i, ctx in enumerate(candidates):
            if i not in seen_idx:
                reordered.append(ctx)

        logger.info(
            "Rerank 完成",
            extra={"extra_fields": {
                "candidates": len(candidates),
                "top_score": reordered[0].score if reordered else 0,
            }},
        )
        return reordered + rest


    # ── vector retrieval ─────────────────────────────────────

    async def _vector_retrieve(self, rewritten: dict) -> list[RetrievedContext]:
        if not self.vector_store:
            return []

        queries = rewritten.get("queries", []) or []
        if not queries:
            return []

        # 并发执行多查询检索（旧实现是串行 for 循环）
        async def _search(q: str) -> list[RetrievedContext]:
            try:
                results = await self.vector_store.search(q, top_k=5)
            except Exception:
                logger.warning("向量检索失败: %s", q, exc_info=True)
                return []

            out: list[RetrievedContext] = []
            for doc, score in results:
                meta = doc.get("metadata", {}) or {}
                out.append(RetrievedContext(
                    content=doc.get("content", ""),
                    source=doc.get("source", "vector_store"),
                    score=float(score),
                    retrieval_type="vector",
                    metadata=meta,
                ))
            return out

        batches = await asyncio.gather(*(_search(q) for q in queries[:3]))
        out: list[RetrievedContext] = []
        for b in batches:
            out.extend(b)

        # 数据级 ACL：只保留调用方有权访问的文档
        out = self._apply_acl(out)
        return self._expand_small_to_big(out)

    def _apply_acl(self, contexts: list[RetrievedContext]) -> list[RetrievedContext]:
        """
        按文档可见范围过滤检索结果。

        判定依据：每个分块的 metadata.source 对应 documents 表里的
        source_path，据此查到该文档的 acl_scope 并比对调用方的角色集。

        设计取舍：**过滤失败时保留结果**（fail-open）还是丢弃（fail-closed）？
        这里选择 fail-open 但记警告 —— 因为 ACL 元数据缺失时全丢会导致
        系统不可用；生产环境应改为 fail-closed 并补齐元数据。
        """
        scopes = self._acl_scopes
        if scopes is None:
            return contexts  # 不限制（管理员）

        try:
            from services.database import get_db

            db = get_db()
            allowed_paths: dict[str, bool] = {}
            kept: list[RetrievedContext] = []
            denied = 0
            for c in contexts:
                src = str((c.metadata or {}).get("source", ""))
                if not src:
                    kept.append(c)
                    continue
                if src not in allowed_paths:
                    doc = db.fetchone(
                        "SELECT acl_scope FROM documents WHERE source_path = ? "
                        "ORDER BY created_at DESC LIMIT 1",
                        (src,),
                    )
                    if doc is None:
                        # 元数据缺失：fail-open，记日志
                        logger.warning(
                            "文档缺少 ACL 元数据，默认放行",
                            extra={"extra_fields": {"source": src[:80]}},
                        )
                        allowed_paths[src] = True
                    else:
                        scope = (doc["acl_scope"] or "public")
                        allowed_paths[src] = (
                            scope == "public" or scope in scopes
                        )
                if allowed_paths[src]:
                    kept.append(c)
                else:
                    denied += 1

            if denied:
                logger.info(
                    "ACL 过滤掉部分结果",
                    extra={"extra_fields": {"denied": denied, "kept": len(kept)}},
                )
            return kept
        except Exception:
            logger.warning("ACL 过滤失败，返回未过滤结果", exc_info=True)
            return contexts

    @staticmethod
    def _expand_small_to_big(contexts: list[RetrievedContext]) -> list[RetrievedContext]:
        """
        小-大分块（small-to-big）的「大块」阶段。

        向量/BM25 用**小章节**命中（区分度高），但小章节信息量有限，
        直接交给 LLM 会导致答案「准确但不完整」。这里为每个命中项
        附加其所属**完整文档**（parent_content）作为 llm_context。

        **去重键必须用 chunk，不能用 parent_id。**
        ------------------------------------------------
        metadata.parent_id 形如 `<doc_id>#doc` —— 它是**文档级**的，
        同一文档的 4~6 个章节全部共享同一个 parent_id。

        早期实现按 parent_id 去重，后果是**每个文档只保留一个章节**：
            差旅报销管理规定 命中 4 个章节（申请/报销流程/发票/特殊情况）
              -> 只剩 1 个（先遇到的那个，通常是"差旅申请"）
        于是文档内真正的答案章节（"4. 报销流程"）被**静默丢弃**。

        实测（"出差回来多久内要报销？"）：
            vs.search 返回 5 条，gold 在 rank 2
              -> 按 parent_id 去重后只剩 2 条，gold 消失
              -> 最终 sources 里没有 gold（答案对，引用错）

        该缺陷此前被掩盖：旧实现同时把 content 替换成 parent 全文，
        所以每个文档"看起来"也只剩一条、且内容相同。P2.1 让 content
        保持为命中章节后，这个错误的去重键才暴露出来。
        """
        # 按 chunk 去重：同一 chunk 可能被多个改写查询命中，
        # 合并时保留最高分，并把 hit_count 记进 metadata。
        by_chunk: dict[str, RetrievedContext] = {}
        order: list[str] = []

        for c in contexts:
            meta = c.metadata or {}
            cid = str(meta.get("doc_id") or "") + "#" + str(meta.get("chunk_index"))
            if not meta.get("doc_id"):
                # 无 doc_id 的结果不参与此合并
                cid = f"__raw__{len(by_chunk)}"
            parent_body = meta.get("parent_content") or ""

            if cid in by_chunk:
                prev = by_chunk[cid]
                # 同一 chunk 被多路命中：保留最高分，累计命中次数
                prev.metadata["hit_count"] = int(prev.metadata.get("hit_count", 1)) + 1
                if c.score > prev.score:
                    prev.score = c.score
                continue

            by_chunk[cid] = RetrievedContext(
                content=c.content,
                source=c.source,
                score=c.score,
                retrieval_type=c.retrieval_type,
                metadata={**meta, "expanded": bool(parent_body), "hit_count": 1},
                llm_context=parent_body or c.llm_context,
            )
            order.append(cid)

        return [by_chunk[cid] for cid in order]


    # ── hybrid reranking ─────────────────────────────────────

    @staticmethod
    def _hybrid_rerank(contexts: list[RetrievedContext]) -> list[RetrievedContext]:
        """
        混合重排序：按检索来源加权。

        当前只有一路检索，权重恒为 1.0 —— 保留权重表以便将来扩展。
        """
        weight_map = {
            "vector": 1.0,
        }
        for ctx in contexts:
            ctx.score *= weight_map.get(ctx.retrieval_type, 1.0)

        seen: set[str] = set()
        unique: list[RetrievedContext] = []
        for ctx in contexts:
            key = ctx.content[:100]
            if key and key not in seen:
                seen.add(key)
                unique.append(ctx)

        unique.sort(key=lambda c: c.score, reverse=True)
        return unique

    # ── answer generation ────────────────────────────────────

    @stage("generate")
    async def _generate_answer(
        self,
        question: str,
        contexts: list[RetrievedContext],
        intent: QueryIntent,
    ) -> tuple[str, list[str]]:
        # 来源标识：展示**中文名**，不传服务端完整路径。
        #
        # 为什么需要映射：落盘文件名是服务端生成的 UUID（上传接口出于
        # 防路径穿越刻意不采用客户端文件名），直接展示会给用户看到
        # `156fc719bb654112a177c5a2b9cd2933.md` 这种无法辨认的字符串。
        #
        # 映射来源：documents.original_name（上传时记录的中文名），
        # 按 source_path 关联。查不到时回退为文件名，保证不出现空标签。
        #
        # 注意：source_path 仍是唯一底层真实路径，检索/文件访问不变，
        # 这里只改**展示**。
        name_by_path: dict[str, str] = {}
        try:
            from services.database import get_db

            for row in get_db().fetchall(
                "SELECT source_path, original_name FROM documents"
            ):
                sp = str(row["source_path"] or "")
                nm = str(row["original_name"] or "").strip()
                if sp and nm:
                    name_by_path[sp] = nm
                    # 同时按文件名建键：metadata.source 存的是绝对路径，
                    # 而 source_path 是相对路径，两者需都能命中
                    name_by_path[Path(sp).name] = nm
        except Exception:
            logger.warning("查询文档显示名失败，回退为文件名", exc_info=True)

        def _display_name(src: str) -> str:
            if not src:
                return ""
            return (
                name_by_path.get(src)
                or name_by_path.get(Path(src).name)
                or Path(src).name
            )

        def _src_label(c: RetrievedContext) -> str:
            # 当前检索结果只有 vector 一种类型
            if c.source:
                return _display_name(c.source)
            return "未知来源"

        # 喂给 LLM 的是**展开后的父文档**（llm_context），不是命中的小章节。
        # small-to-big 的价值就在这里：小章节用于精确定位，父文档用于
        # 给出完整答案。若这里有 llm_context 却仍用 content，会退回到
        # "答案准确但不完整"（实测完整性 3.06/5）。
        def _ctx_body(c: RetrievedContext) -> str:
            return c.llm_context or c.content

        # Prompt 结构（稳定 system + 固定顺序动态段）见 services/prompt_builder.py。
        # 这里只负责把"文档名 + 正文"解析出来，不参与排版：
        # 排版一旦内联在本方法里，稳定前缀就容易被后续改动无意破坏。
        prompt_sources = [
            PromptSource(document=_src_label(c), body=_ctx_body(c))
            for c in contexts
        ]
        if settings.answer_prompt_mode == "legacy":
            # A/B 对照用（改前的结构：意图风格进 system、标签带类型与分数）
            messages, sections = build_legacy_answer_messages(
                question, prompt_sources, intent.value,
                types=[c.retrieval_type for c in contexts],
                scores=[c.score for c in contexts],
            )
        else:
            messages, sections = build_answer_messages(
                question, prompt_sources, intent.value
            )
        # 记账：把本次 prompt 的"稳定/动态"字符数交给计量层，
        # 用于把 Provider 报的 hit token 拆成"稳定前缀命中"与"检索上下文命中"。
        record_prompt_sections(sections)

        reasoning_steps = [
            f"检索到 {len(contexts)} 条相关上下文",
            "上下文构成: " + (
                ", ".join(
                    f"{k}={v}"
                    for k, v in sorted(
                        {
                            t: sum(1 for c in contexts if c.retrieval_type == t)
                            for t in {c.retrieval_type for c in contexts}
                        }.items()
                    )
                ) or "无"
            ),
        ]

        resp = await self.llm.ainvoke(messages)
        reasoning_steps.append("答案生成完成")
        return resp.content, reasoning_steps

    @staticmethod
    def _calc_confidence(contexts: list[RetrievedContext]) -> float:
        if not contexts:
            return 0.0
        avg_score = sum(c.score for c in contexts) / len(contexts)
        return min(avg_score, 1.0)
