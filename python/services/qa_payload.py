"""
单轮问答的快照序列化 —— **唯一一份**

为什么要有这一层
----------------
"一轮问答"的对外形状（问题 / 答案 / 出处 / 推理步骤 / 用量）此前只存在于
`api/main.py` 的响应构造里。现在它还要被**落库**（对话历史），于是出现
"节点写一份、API 写一份"的风险 —— 两份一旦漂移，就会出现
"实时答对但恢复出来的卡片不一样"这种极难排查的问题。

因此把序列化抽到这里，节点（存储）与 API（传输）都调它。

引用契约（**不要改动**）
------------------------
`sources[].content` 必须是**命中的章节**（`display_content`），
不是 small-to-big 展开后的父文档（`llm_context`）。
早期实现把父文档当引用返回，导致每条引用的开头永远是"## 1. 文档说明"，
用户无法核对出处（实测引用准确率仅 44.4%）。`llm_context` 只喂 LLM，不外传。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# 快照结构版本。改结构时递增，读取端可据此识别旧行。
PAYLOAD_VERSION = 1

# 单条引用的展示截断长度（用户核对用，不需要全文）
SOURCE_EXCERPT_CHARS = 400

# 答案长度上限：模型异常输出时不至于写爆库
ANSWER_MAX_CHARS = 20_000


def serialize_sources(contexts: list[Any]) -> list[dict[str, Any]]:
    """检索上下文 → 前端的出处列表（命中章节，非父文档）"""
    return [
        {
            "title": c.title,
            "section": c.section,
            "content": c.display_content[:SOURCE_EXCERPT_CHARS],
            "source": c.source,
            "score": c.score,
            "type": c.retrieval_type,
            "parent_available": c.parent_available,
        }
        for c in contexts
    ]


def build_turn_payload(
    qa_result: Any,
    *,
    usage: dict[str, Any] | None = None,
    turn: int = 1,
    trace_id: str = "",
    metrics_degraded: bool = False,
) -> dict[str, Any]:
    """把一轮问答打包成可直接返回前端、也可直接落库的形状。

    前端恢复历史时原样喂给同一个渲染函数，因此这里的字段名**就是**前端契约。
    """
    answer = qa_result.answer or ""
    if len(answer) > ANSWER_MAX_CHARS:
        logger.warning(
            "答案超长，落库前截断",
            extra={"extra_fields": {"original_chars": len(answer)}},
        )
        answer = answer[:ANSWER_MAX_CHARS]

    return {
        "payload_version": PAYLOAD_VERSION,
        "question": qa_result.question,
        "answer": answer,
        "confidence": qa_result.confidence,
        "intent": qa_result.intent.value if qa_result.intent else "",
        "sources": serialize_sources(qa_result.contexts),
        "reasoning_steps": qa_result.reasoning_steps,
        "turn": turn,
        "trace_id": trace_id,
        "usage": usage or {},
        "metrics_degraded": metrics_degraded,
    }
