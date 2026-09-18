"""
ReAct 问答 Agent —— LLM 自主决定检索通道与停止时机

与 Pipeline 版（qa_agent.QAAgent）的根本区别
-------------------------------------------
Pipeline 版（现状）:
    意图分类（5 选 1，**仅此一次决策**）
      -> 查表得静态检索计划
      -> 按 plan 字段**硬编码**并发调用检索器
      -> 重排 -> 生成
    决策颗粒度：入口 1 次，之后全自动。

ReAct 版（本文件）:
    意图只作 **hint**（提示如何组织答案，不限制工具选择）
      -> 循环：
           Thought + Action(tool, args)     <- LLM 显式选择工具与参数
             -> 执行工具
             -> Observation 回灌
             -> LLM 判断：够了 -> 停止；不够 -> 下一个 Action
      -> 汇总所有 Observation
      -> 最终 Rerank
      -> 复用**现有生成链**（保留三层引用契约）
    决策颗粒度：每一步。

职责边界（三方分离，用户指定）
------------------------------
    Intent Hint       决定「答案怎么组织」（注入 ANSWER_STYLE）
    ReAct Agent       决定「查什么、怎么查、何时停」
    Final Generator   依据证据生成答案并落实引用契约

为什么要独立文件 + 开关
-----------------------
架构刚冻结（docs/architecture-freeze.md），且已实测过
「更智能的机制 ≠ 更好」。
因此本实现通过 `qa_mode` 与 pipeline 版并存，可 A/B、可回退。

工具集（当前 1 个已注册）
------------------------
    search_docs      BM25 关键词检索 —— 唯一工具（实测 Chunk R@5 89.1%）
    lookup_facts     **留接口但未注册** —— 事实层当前只覆盖 2/55 篇
                     （52 条事实），注册后会让 LLM 把步数浪费在空工具上，
                     待事实层扩到全量再启用。
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from openai import AsyncOpenAI

from agents.qa_agent import (
    ANSWER_PROMPT,
    ANSWER_STYLE,
    QueryIntent,
    QAAgent,
    QAResult,
    RetrievedContext,
)
from config import settings
from services.usage_meter import record_openai, stage

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# ReAct 系统提示
# ══════════════════════════════════════════════════════════════
REACT_SYSTEM_PROMPT = """\
你是一个企业知识库检索 Agent。你的任务是**通过多轮检索收集足够证据**，
最终回答用户问题。

## 工作方式
1. 先判断需要什么信息
2. 调用合适的工具检索（每次可以调用多个）
3. 观察返回结果，判断信息是否足够
   - 不够：继续调用工具补充（可以换关键词、换工具、查别的实体）
   - 够了：**停止调用工具**（不再发起工具调用即表示结束）
4. 系统会把你收集到的证据交给另一个模块生成最终答案，你**不需要**写答案

## 工具选择建议
- 当前只有 search_docs 一个工具（BM25 关键词检索）
- 它适合所有类型的问题：条款、数值、流程、规定、概览
- 若一次检索没拿到需要的信息，应**换关键词**再试
  （例如从"加班"细化到"加班费 150%"，或改用文档里的原词）

## 重要约束
- 检索词要包含**具体名词与数值**（如"加班费 150%"、"年假 天数"），
  不要只写"加班"这种宽泛词
- 若工具返回空结果，不要重复调用同样的查询词 —— 换词再试
- **最多 4 轮**。若已收集到能回答问题的主要信息，应立即停止
- 不要为了"更全面"而无限检索 —— 检索到核心条款即可停止

## 意图提示
{intent_hint}

注意：以上意图提示**仅用于辅助你判断需要哪类证据**，
不限制你的工具选择。你可以自由调用任何工具。
"""

INTENT_HINT_TEMPLATE = """用户问题可能属于：{intent}（置信度 {conf:.2f}）
该信息仅用于辅助判断回答组织方式，不得限制工具选择，也不得作为强制路由条件。"""


def _build_tools() -> list[dict[str, Any]]:
    """
    构建工具 schema。

    当前只注册 `search_docs`（BM25）。

    `lookup_facts` 仍是预留接口（未注册）：事实层当前只覆盖 2/55 篇
    （52 条），注册后会让 LLM 把步数浪费在几乎无产出的工具上。
    待事实层扩到全量再启用。
    """
    tools: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "search_docs",
                "description": (
                    "在企业制度/产品文档中做关键词检索（BM25）。"
                    "适合查找具体条款、数值标准、流程步骤、禁止性规定。"
                    "这是最主要的检索工具。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "检索关键词，应包含具体名词与数值。"
                                "例：'加班费 150%'、'年假 天数 折算'、'密码 长度 位数'"
                            ),
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "返回条数，默认 5，范围 1-8",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
    ]

    return tools


def _fmt_args(args: dict) -> str:
    """把工具参数压成一行短摘要，供 reasoning_steps 展示。

    修复说明：`_react_loop` 一直在调 `_fmt_args(args)`，但本函数**从未定义**，
    因此 ReAct 模式在第一次工具调用时必抛
    `NameError: name '_fmt_args' is not defined` —— 整个 react 链路不可用。
    之前未被发现，是因为默认 qa_mode=pipeline，且 ReAct 的 A/B 评测跑在
    这段引用引入之前。
    """
    if not args:
        return ""
    parts = []
    for k, v in args.items():
        s = str(v)
        parts.append(f"{k}={s[:40]}")
    return ", ".join(parts)


def _ctx_key(c: RetrievedContext) -> str:
    """检索结果的去重键 —— **必须是块级**。

    同类修复：本函数同样缺失（`NameError: name '_ctx_key' is not defined`）。

    为什么不能用 `parent_id`：那是**文档级**标识（形如 `<doc_id>#doc`），
    用它去重会把同一文档的所有章节折叠成一条，静默丢掉命中章节
    —— 这正是早前 gold 章节"检索不到"的根因之一。
    正确粒度是 `doc_id#chunk_index`（回退：source#chunk_index / 内容前缀）。
    """
    m = c.metadata or {}
    doc_id = str(m.get("doc_id") or c.source or "")
    chunk_index = m.get("chunk_index")
    if doc_id and chunk_index is not None and chunk_index != "":
        return f"{doc_id}#{chunk_index}"
    if doc_id:
        return f"{doc_id}#{c.section or c.content[:100]}"
    return c.content[:100]


class ReactQAAgent:
    """
    ReAct 版问答 Agent。

    对外接口与 `QAAgent.answer(question, acl_scopes)` **完全一致**，
    返回同样的 `QAResult`，因此可在编排层直接替换而无需改动图结构。
    """

    def __init__(
        self,
        vector_store: Any = None,
        max_steps: int = 4,
    ) -> None:
        # 复用 pipeline 版的检索原语与生成链 —— 不重复实现，
        # 也保证引用契约（content / llm_context 三层分离）完全一致。
        self._qa = QAAgent(vector_store=vector_store)
        self.max_steps = max(1, int(max_steps))

        self._tools = _build_tools()

        self._client: AsyncOpenAI | None = None
        # 记录本轮的 ReAct 轨迹（供 reasoning_steps 与调试）
        self._trace: list[str] = []

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                timeout=settings.llm_timeout_seconds,
                max_retries=settings.llm_max_retries,
            )
        return self._client

    # ══════════════════════════════════════════════════════════
    # 对外入口（与 QAAgent.answer 签名一致）
    # ══════════════════════════════════════════════════════════

    async def answer(
        self, question: str, acl_scopes: list[str] | None = None
    ) -> QAResult:
        self._qa._acl_scopes = acl_scopes
        self._trace = []

        # ① 意图只作 hint，不再路由
        intent, conf = await self._qa._classify_intent(question)
        rewrite = await self._qa._rewrite_query(question)

        # ② ReAct 循环：LLM 自主选择工具
        contexts, n_steps = await self._react_loop(question, intent, conf, rewrite)

        # ③ 汇总后走**现有**重排 + 生成链（保留引用契约）
        if contexts:
            contexts = self._qa._hybrid_rerank(contexts)
            contexts = await self._qa._rerank_contexts(question, contexts)

        top = contexts[:8]
        answer_text, reasoning = await self._qa._generate_answer(question, top, intent)

        reasoning.insert(0, f"意图提示: {intent.value} (置信度 {conf:.2f})")
        reasoning.insert(
            1,
            f"ReAct 检索 {n_steps} 轮，工具调用见轨迹，共收集 {len(contexts)} 条上下文",
        )
        reasoning.extend(self._trace)

        return QAResult(
            question=question,
            answer=answer_text,
            contexts=top,
            intent=intent,
            confidence=self._qa._calc_confidence(top),
            reasoning_steps=reasoning,
        )

    # ══════════════════════════════════════════════════════════
    # ReAct 循环
    # ══════════════════════════════════════════════════════════

    @stage("react")
    async def _react_loop(
        self, question: str, intent: QueryIntent, conf: float, rewrite: dict
    ) -> tuple[list[RetrievedContext], int]:
        """
        返回 (累积的检索上下文, 实际轮数)。

        循环终止条件（任一）：
          1. LLM 不再发起工具调用（它认为信息已足够）
          2. 达到 max_steps

        设计要点：**LLM 看到的 Observation 是文本摘要**（用于决策），
        而**最终生成用的是完整 RetrievedContext**（含 llm_context）。
        两条通道分离，避免为了给 LLM 看而破坏引用契约。
        """
        hint = INTENT_HINT_TEMPLATE.format(intent=intent.value, conf=conf)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": REACT_SYSTEM_PROMPT.format(intent_hint=hint)},
            {
                "role": "user",
                "content": (
                    f"用户问题：{question}\n\n"
                    f"（检索线索：{', '.join(rewrite.get('queries') or []) or '无'}；"
                    f"实体：{', '.join(rewrite.get('entities') or []) or '无'}）"
                ),
            },
        ]

        collected: list[RetrievedContext] = []
        seen_keys: set[str] = set()
        steps = 0

        for step in range(1, self.max_steps + 1):
            try:
                resp = await self.client.chat.completions.create(
                    model=settings.openai_model,
                    messages=messages,
                    tools=self._tools,
                    temperature=0,
                )
            except Exception:
                logger.warning("ReAct 决策调用失败，结束循环", exc_info=True)
                break

            # 计量：走原生 client，需显式记录。
            # stage 由 @stage("react") 提供，meter 会合成 react_step1/2/...
            record_openai(resp)

            msg = resp.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None)

            # LLM 不再调用工具 -> 认为信息足够，正常终止
            if not tool_calls:
                self._trace.append(f"[第 {step} 轮] Agent 判断信息已足够，停止检索")
                break

            steps = step
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            })

            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                ctxs, obs_text = await self._exec_tool(name, args)
                self._trace.append(
                    f"[第 {step} 轮] {name}({_fmt_args(args)}) -> {len(ctxs)} 条"
                )

                # 去重：同一 chunk 不重复累积（多个工具可能返回同一条）
                for c in ctxs:
                    key = _ctx_key(c)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        collected.append(c)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": obs_text[:4000] or "（无结果）",
                })

        else:
            # for-else：循环自然走完（达到 max_steps 仍未停）
            self._trace.append(f"[第 {self.max_steps} 轮] 达到步数上限，强制停止")
            steps = self.max_steps

        return collected, steps

    # ══════════════════════════════════════════════════════════
    # 工具执行
    # ══════════════════════════════════════════════════════════

    async def _exec_tool(
        self, name: str, args: dict
    ) -> tuple[list[RetrievedContext], str]:
        """执行一个工具，返回 (完整上下文, 给 LLM 看的文本摘要)"""
        try:
            if name == "search_docs":
                return await self._t_search_docs(args)
            if name == "lookup_facts":
                # 留接口但未注册（事实层仅覆盖 2/55 篇）
                return [], "该工具暂未开放。请用 search_docs 检索具体条款。"
            return [], f"未知工具 {name}"
        except Exception as e:
            logger.warning("工具 %s 执行失败: %s", name, e, exc_info=True)
            return [], f"工具执行失败：{type(e).__name__}"

    async def _t_search_docs(self, args: dict) -> tuple[list[RetrievedContext], str]:
        query = str(args.get("query") or "").strip()
        if not query:
            return [], "缺少 query 参数"
        top_k = args.get("top_k") or 5
        try:
            top_k = max(1, min(8, int(top_k)))
        except (TypeError, ValueError):
            top_k = 5

        vs = self._qa.vector_store
        if vs is None:
            return [], "向量库不可用"

        res = await vs.search(query, top_k=top_k)
        ctxs: list[RetrievedContext] = []
        lines: list[str] = []
        for doc, score in res:
            meta = doc.get("metadata") or {}
            ctxs.append(RetrievedContext(
                content=doc.get("content", ""),
                source=doc.get("source", ""),
                score=float(score),
                retrieval_type="vector",
                metadata=meta,
                llm_context=str(meta.get("parent_content") or ""),
            ))
            title = meta.get("title") or ""
            sec = meta.get("section_title") or ""
            snippet = (doc.get("content") or "").replace("\n", " ")[:220]
            lines.append(f"[{title} > {sec}] {snippet}")

        if not lines:
            return [], "未检索到相关内容，建议更换关键词"
        return ctxs, "\n".join(lines)



