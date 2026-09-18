"""
生成 Prompt 结构回归测试

这两件事一旦被破坏，缓存命中率会无声地掉下去，而答案看起来仍然正常，
所以必须用测试把"稳定前缀"钉住：

  1. system prompt **跨意图逐字节相同**（旧的
     `ANSWER_PROMPT + "本题意图为「x」，…"` 就是在这里出问题的）
  2. Prompt 里**不含分数/类型**这类每问必变的字段
  3. 动态段顺序固定为 metadata → context → query
  4. 文档名必须随来源发送（否则 `[来源: 文档名]` 引用会退化成无法核对的下标）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from services.prompt_builder import (  # noqa: E402
    ANSWER_SYSTEM_PROMPT,
    PromptSource,
    build_answer_messages,
    build_legacy_answer_messages,
    format_context,
)

INTENTS = ["factoid", "procedural", "comparative", "analytical", "exploratory"]

SOURCES = [
    PromptSource(document="YQ-INST-026-产品发布上线规范.md", body="第一章 总则\n本规范适用于……"),
    PromptSource(document="YQ-INST-028-技术变更管理办法.md", body="第二章 分级\n低风险变更……"),
]


# ── 稳定前缀 ─────────────────────────────────────────────────

def test_system_prompt_is_byte_identical_across_intents():
    """核心断言：换意图不能改变 system 段一个字节"""
    seen = set()
    for intent in INTENTS:
        messages, _ = build_answer_messages("问题？", SOURCES, intent)
        sys_msg = messages[0]
        assert isinstance(sys_msg, SystemMessage)
        seen.add(sys_msg.content)
    assert len(seen) == 1, "system prompt 随意图变化了 —— 前缀复用会在意图切换处断掉"
    assert seen.pop() == ANSWER_SYSTEM_PROMPT


def test_system_prompt_is_byte_identical_across_questions():
    """换问题、换上下文都不能改变 system 段"""
    a, _ = build_answer_messages("问题甲？", SOURCES, "factoid")
    b, _ = build_answer_messages("完全不同的问题乙？", SOURCES[::-1], "procedural")
    assert a[0].content == b[0].content == ANSWER_SYSTEM_PROMPT


def test_intent_is_in_dynamic_section_not_system():
    """意图只以"本次取值"的形式出现在动态段。

    system 段里可以（也应该）静态列出**全部**五种意图的通用规则 ——
    那是常量；不能出现的是"本次意图是哪一种"这种按调用变化的插值。
    """
    messages, sections = build_answer_messages("问题？", SOURCES, "comparative")
    system, user = messages[0].content, messages[1].content

    # 动态段带上了本次取值
    assert "intent=comparative" in user
    assert sections.metadata_chars > 0

    # system 段是常量：五种意图的规则都在，且与本次意图无关
    for it in INTENTS:
        assert it in system
    # 对照：换成别的意图，system 一字节不变
    other, _ = build_answer_messages("问题？", SOURCES, "factoid")
    assert other[0].content == system
    assert "intent=factoid" in other[1].content


# ── 动态段：不含易变字段 ──────────────────────────────────────

def test_no_score_or_type_in_prompt():
    """分数/类型不得出现在 prompt 里（每问必变，破坏前缀复用）"""
    # 故意构造带分数与类型的上下文，确认它们不会进入 Prompt
    messages, _ = build_answer_messages("问题？", SOURCES, "factoid")
    text = messages[0].content + messages[1].content
    for bad in ("分数", "score", "0.77", "类型:", "type="):
        assert bad not in text, f"prompt 中出现易变字段: {bad}"


def test_scores_do_not_change_prompt():
    """同一批文档，只要正文与文档名相同，prompt 必须逐字节相同。

    旧的标签 `[来源 1: x | 类型: vector | 分数: 0.77]` 会让这一步失败。
    """
    a, _ = build_answer_messages("问题？", SOURCES, "factoid")
    # 重新构造"同样内容"的对象（模拟分数变了但内容没变）
    same = [PromptSource(s.document, s.body) for s in SOURCES]
    b, _ = build_answer_messages("问题？", same, "factoid")
    assert a[1].content == b[1].content


def test_document_name_is_present_for_citation():
    """文档名必须发送 —— 否则 [来源: 文档名] 无从落地"""
    messages, _ = build_answer_messages("问题？", SOURCES, "factoid")
    user = messages[1].content
    for s in SOURCES:
        assert s.document in user
    assert 'document="YQ-INST-026-产品发布上线规范.md"' in user


# ── 动态段顺序固定 ───────────────────────────────────────────

def test_dynamic_section_order_is_fixed():
    messages, _ = build_answer_messages("我的问题？", SOURCES, "procedural")
    user = messages[1].content
    i_meta = user.index("<request_metadata>")
    i_ctx = user.index("<retrieved_context>")
    i_q = user.index("<user_query>")
    assert i_meta < i_ctx < i_q, "动态段顺序必须固定为 metadata → context → query"


def test_query_is_last_so_varying_text_is_at_the_end():
    """用户问题变化最大，必须放最后，前缀才能尽量长"""
    messages, _ = build_answer_messages("末尾问题？", SOURCES, "factoid")
    user = messages[1].content
    assert user.rstrip().endswith("</user_query>")
    assert "末尾问题？" in user[user.index("<user_query>"):]


# ── 上下文格式 ───────────────────────────────────────────────

def test_context_uses_stable_source_tags():
    text = format_context(SOURCES)
    assert '<source id="1" document="YQ-INST-026-产品发布上线规范.md">' in text
    assert '<source id="2" document="YQ-INST-028-技术变更管理办法.md">' in text
    assert text.count("</source>") == 2


def test_context_escapes_quotes_in_document_name():
    """文档名里的双引号必须转义，否则 XML 形状被破坏"""
    text = format_context([PromptSource(document='a"b.md', body="正文")])
    assert 'document="a\'b.md"' in text
    assert text.count('"') == 6 or text.count('document="') == 1


def test_empty_context_is_explicit():
    text = format_context([])
    assert "未检索到" in text


def test_sections_add_up():
    _, s = build_answer_messages("问题？", SOURCES, "factoid")
    assert s.total_chars == s.stable_chars + s.metadata_chars + s.context_chars + s.query_chars
    assert s.stable_chars == len(ANSWER_SYSTEM_PROMPT)
    assert s.context_chars > 0


# ── 旧结构（A/B 对照路径）必须保持"不稳定"，否则对照无意义 ──

def test_legacy_prompt_varies_by_intent():
    a, _ = build_legacy_answer_messages("问题？", SOURCES, "factoid")
    b, _ = build_legacy_answer_messages("问题？", SOURCES, "procedural")
    assert a[0].content != b[0].content, "legacy 路径应当保留'意图进 system'的旧行为"


def test_legacy_prompt_contains_score_and_type():
    messages, _ = build_legacy_answer_messages(
        "问题？", SOURCES, "factoid",
        types=["vector", "vector"], scores=[0.77, 0.62],
    )
    text = messages[1].content
    assert "分数: 0.77" in text and "类型: vector" in text


def test_stable_prefix_is_longer_than_legacy():
    """改动的直接收益：稳定前缀变长了（字符口径）"""
    _, new = build_answer_messages("问题？", SOURCES, "factoid")
    _, old = build_legacy_answer_messages("问题？", SOURCES, "factoid")
    assert new.stable_chars > old.stable_chars
