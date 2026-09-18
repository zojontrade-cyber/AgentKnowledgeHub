"""
问答 Prompt 构造 —— 稳定前缀优先

为什么把 Prompt 单独抽成一层
----------------------------
原先 Prompt 是拼在 `QAAgent._generate_answer` 里的内联字符串，导致两个问题：

1. **稳定前缀被动态内容污染**。旧的 system prompt 是
   `ANSWER_PROMPT + "本题意图为「factoid」，直接给出事实要点…"`，
   于是"不同意图的两条问题，system prompt 从 ~149 字符处就分叉"
   （实测见 docs/chat-metrics.md 5.1）。前缀复用被自己的排版打断。
2. **每条上下文标签里嵌了分数**：
   `[来源 1: xxx | 类型: vector | 分数: 0.77]`。
   分数每问必变，在第一个来源处就把前缀打断。

现在按"稳定段 → 动态段"重排，**system prompt 是一个常量，永不插值**：

    ┌─ system（常量，所有问题逐字节相同）──────────────┐
    │ 角色 / 回答规则 / 引用规则 / 输出要求            │
    └──────────────────────────────────────────────┘
    ┌─ user（动态，按固定顺序）───────────────────────┐
    │ <request_metadata> intent=… </request_metadata> │
    │ <retrieved_context> <source …/>… </…>           │
    │ <user_query> … </user_query>                    │
    └──────────────────────────────────────────────┘

动态段的顺序**固定为 metadata → context → query**：变化最大的用户问题放最后，
这样前缀能尽量长。不要改动这个顺序，也不要让任何一段变成可选。

删掉了什么、为什么
------------------
- `分数: {score:.2f}` —— 每问必变，对生成无实质贡献，且破坏前缀复用。
  分数保留在程序内部（`RetrievedContext.score`），只是**不再发给 LLM**。
- `类型: vector` —— 只有一种取值，纯噪音。

保留了什么、为什么
------------------
- **文档名仍随来源发送**（作为 `document` 属性）。
  不能删：system prompt 要求模型用 `[来源: 文档名]` 标注引用，
  删掉文档名会让引用退化成无法核对的下标，直接损害引用准确率。
  它同时也是稳定的（同一文档 → 同一字符串），不破坏前缀复用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage


# ══════════════════════════════════════════════════════════════
# 稳定段：**永远不要在这里插值**
# ══════════════════════════════════════════════════════════════
#
# 任何按问题/意图变化的文字都必须放进动态段（见 build_answer_messages）。
# 一旦这里出现 f-string 或拼接，前缀复用就会在那一行断掉。

ANSWER_SYSTEM_PROMPT = """\
你是企业知识库问答助手。

你的任务是根据提供的知识库内容回答用户问题。

回答规则：
1. 仅基于提供的知识库内容回答。
2. 不得编造知识库不存在的信息。
3. 对事实进行准确引用。
4. 信息不足时明确说明。

引用规则：
- 使用 [来源: 文档名] 标记来源，文档名取 <source> 标签的 document 属性。
- 不得伪造来源。
- 每个关键结论都应有对应依据。

按问题类型组织回答（问题类型见 <request_metadata> 的 intent）：
- factoid：直接给出事实要点，不展开。
- procedural：按步骤或流程顺序组织。
- comparative：明确列出相同点与不同点。
- analytical：先给结论，再分点说明依据。
- exploratory：先给整体概览，再列出关键要点。

输出要求：
- 直接回答问题。
- 不重复用户问题。
- 内容简洁、准确。\
"""


# ══════════════════════════════════════════════════════════════
# 分段记账：让"稳定前缀"可被测量
# ══════════════════════════════════════════════════════════════

@dataclass
class PromptSections:
    """一次调用的 prompt 按"是否稳定"切分的字符数。

    用途：把 Provider 报的 `prompt_cache_hit_tokens` 拆成
      "稳定前缀命中了多少 / 检索上下文命中了多少"，
    否则只能看到一个混合的总命中率（见 usage_meter 的三个率）。
    """

    stable_chars: int = 0       # system prompt（常量）
    metadata_chars: int = 0     # <request_metadata>
    context_chars: int = 0      # <retrieved_context>
    query_chars: int = 0        # <user_query>

    @property
    def dynamic_chars(self) -> int:
        return self.metadata_chars + self.context_chars + self.query_chars

    @property
    def total_chars(self) -> int:
        return self.stable_chars + self.dynamic_chars


@dataclass
class PromptSource:
    """喂给 LLM 的一条来源（已解析好文档名，builder 不碰数据库）"""

    document: str
    body: str


# ══════════════════════════════════════════════════════════════
# 构造
# ══════════════════════════════════════════════════════════════

def format_context(sources: list[PromptSource]) -> str:
    """把来源序列化为**稳定**格式。

    格式稳定性要求：
      - 不带分数 / 类型等每问必变的字段
      - 顺序由调用方固定（按重排得分降序 + 稳定 tie-break），本函数不改顺序
    """
    if not sources:
        return "（未检索到相关上下文）"
    blocks = []
    for i, s in enumerate(sources, 1):
        # document 属性必须转义双引号，否则会破坏 XML 形状
        name = (s.document or "").replace('"', "'").strip() or "未知来源"
        blocks.append(
            f'<source id="{i}" document="{name}">\n{s.body}\n</source>'
        )
    return "\n\n".join(blocks)


def build_answer_messages(
    question: str,
    sources: list[PromptSource],
    intent: str,
) -> tuple[list[Any], PromptSections]:
    """构造 (messages, 分段字符数)。

    system 段是常量；动态段顺序固定：metadata → context → query。
    """
    context_text = format_context(sources)

    metadata = f"<request_metadata>\nintent={intent}\n</request_metadata>"
    context_block = f"<retrieved_context>\n{context_text}\n</retrieved_context>"
    query_block = f"<user_query>\n{question}\n</user_query>"

    user_content = "\n\n".join([metadata, context_block, query_block])

    sections = PromptSections(
        stable_chars=len(ANSWER_SYSTEM_PROMPT),
        # 只计内容本身，标签开销按比例摊（差异在个位数字符，不影响口径）
        metadata_chars=len(metadata),
        context_chars=len(context_block),
        query_chars=len(query_block),
    )

    messages = [
        SystemMessage(content=ANSWER_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]
    return messages, sections


# ══════════════════════════════════════════════════════════════
# 旧结构（仅供 A/B 对照，不要在新代码里使用）
# ══════════════════════════════════════════════════════════════
#
# 保留它的唯一理由是**可复现的对照**：改造生成 Prompt 会影响答案质量，
# 而"改前"的基线必须能重跑。旧评测脚本（eval_rag_v4.py /
# eval_rag_four_metrics.py）都已失效，历史 JSON 也无法确认
# 是否与当前代码同源，因此把旧结构固化在这里，用
# `settings.answer_prompt_mode = "legacy"` 即可回到改前行为。

LEGACY_ANSWER_PROMPT = """\
你是一个专业的企业知识问答助手。根据检索到的上下文信息回答用户问题。

要求：
1. 答案必须基于提供的上下文，不要编造
2. 如果上下文信息不足，明确告知用户
3. 引用信息来源（如 [来源: xxx]）
4. 如果涉及多个信息源，综合分析后给出结论
5. 保持专业、准确、简洁
"""

LEGACY_ANSWER_STYLE_TEXT = {
    "factoid": "直接给出事实要点，不要展开。",
    "analytical": "先给结论，再分点说明推理依据。",
    "comparative": "用对比结构回答，明确指出两者的异同。",
    "procedural": "按步骤或流程顺序组织答案。",
    "exploratory": "先给整体概览，再列出关键要点。",
}


def build_legacy_answer_messages(
    question: str,
    sources: list[PromptSource],
    intent: str,
    types: list[str] | None = None,
    scores: list[float] | None = None,
) -> tuple[list[Any], PromptSections]:
    """改前的结构：意图风格插进 system，上下文标签带类型与分数。"""
    types = types or ["vector"] * len(sources)
    scores = scores or [0.0] * len(sources)

    if sources:
        context_text = "\n\n".join(
            f"[来源 {i+1}: {s.document} | 类型: {types[i] if i < len(types) else 'vector'}"
            f" | 分数: {(scores[i] if i < len(scores) else 0.0):.2f}]\n{s.body}"
            for i, s in enumerate(sources)
        )
    else:
        context_text = "（未检索到相关上下文）"

    style = LEGACY_ANSWER_STYLE_TEXT.get(intent, "")
    system_prompt = LEGACY_ANSWER_PROMPT + (
        f"\n\n本题意图为「{intent}」，{style}" if style else ""
    )
    user_content = f"上下文信息:\n{context_text}\n\n用户问题: {question}"

    sections = PromptSections(
        stable_chars=len(LEGACY_ANSWER_PROMPT),
        metadata_chars=len(system_prompt) - len(LEGACY_ANSWER_PROMPT),
        context_chars=len(f"上下文信息:\n{context_text}"),
        query_chars=len(f"\n\n用户问题: {question}"),
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]
    return messages, sections
