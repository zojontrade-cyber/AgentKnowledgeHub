"""
LLM 用量计量回归测试

覆盖 services/usage_meter.py 的核心契约：
  - 两个 SDK（LangChain AIMessage / 原生 Completion）的字段差异被归一化
  - 三条取值路径：DeepSeek 富字段、OpenAI cached_tokens、usage_metadata.cache_read
  - cache_supported（是否可测）与 cache_hit_rate（Token 命中比例）口径分离
  - cache_hit_rate 与 cache_hit_calls 是两个不同概念
  - 无 recorder 时全部 no-op（入库链、脚本不受影响）
  - 嵌套 usage_meter() 恢复的是 previous，而不是置 None
  - stage 归因：react 自动合成 _step{N} 后缀
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from services.usage_meter import (  # noqa: E402
    UsageRecorder,
    current_recorder,
    record_langchain,
    record_openai,
    record_prompt_sections,
    stage,
    usage_meter,
)


# ── 构造假响应：形状照抄 scripts/probe_usage_fields.py 的实测输出 ──

def ai_deepseek(prompt=132, completion=8, hit=0, miss=132):
    """LangChain AIMessage：response_metadata.token_usage 带 DeepSeek 私有字段"""
    return NS(
        response_metadata={
            "token_usage": {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
                "prompt_cache_hit_tokens": hit,
                "prompt_cache_miss_tokens": miss,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
            "model_name": "deepseek-chat",
        },
        usage_metadata={
            "input_tokens": prompt,
            "output_tokens": completion,
            "total_tokens": prompt + completion,
            "input_token_details": {"cache_read": hit},
        },
    )


def ai_usage_metadata_only(prompt=500, completion=5, cache_read=None):
    """只有 usage_metadata 的 provider（无 token_usage）"""
    details = {} if cache_read is None else {"cache_read": cache_read}
    return NS(
        response_metadata={},
        usage_metadata={
            "input_tokens": prompt,
            "output_tokens": completion,
            "total_tokens": prompt + completion,
            "input_token_details": details,
        },
    )


def openai_resp(prompt=1000, completion=20, **usage_extra):
    """原生 OpenAI Completion：resp.usage.model_dump()"""
    base = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "completion_tokens_details": None,
        "prompt_tokens_details": {
            "cached_tokens": 0, "audio_tokens": None,
            "cache_write_tokens": None, "image_tokens": None, "text_tokens": None,
        },
    }
    base.update(usage_extra)
    return NS(usage=NS(model_dump=lambda: base), model="deepseek-chat")


# ── 归一化：三条取值路径 ─────────────────────────────────────

def test_deepseek_rich_fields_win():
    with usage_meter() as rec:
        record_langchain(ai_deepseek(prompt=1200, completion=72, hit=800, miss=400), "generate")
    c = rec.calls[0]
    assert (c.prompt_tokens, c.completion_tokens, c.total_tokens) == (1200, 72, 1272)
    assert (c.cache_hit_tokens, c.cache_miss_tokens) == (800, 400)
    assert c.cache_supported is True
    assert c.model == "deepseek-chat"
    assert c.stage == "generate"


def test_openai_cached_tokens_fallback():
    """无 DeepSeek 私有字段时回退 prompt_tokens_details.cached_tokens"""
    resp = openai_resp(prompt=1000, completion=5,
                       prompt_tokens_details={"cached_tokens": 900})
    with usage_meter() as rec:
        record_openai(resp)
    c = rec.calls[0]
    assert (c.cache_hit_tokens, c.cache_miss_tokens) == (900, 100)
    assert c.cache_supported is True


def test_usage_metadata_cache_read_fallback():
    with usage_meter() as rec:
        record_langchain(ai_usage_metadata_only(prompt=500, completion=1, cache_read=100))
    c = rec.calls[0]
    assert (c.cache_hit_tokens, c.cache_miss_tokens) == (100, 400)
    assert c.cache_supported is True


def test_provider_without_cache_fields_is_not_measured():
    """缺字段 -> 不可测。不能把"没报"混进"报了但为 0"。"""
    with usage_meter() as rec:
        record_langchain(ai_usage_metadata_only(prompt=1000, completion=10))
    c = rec.calls[0]
    assert c.cache_supported is False
    assert (c.cache_hit_tokens, c.cache_miss_tokens) == (0, 0)
    # token 总量仍然照常记录
    assert c.prompt_tokens == 1000


def test_explicit_zero_hit_is_measurable():
    """DeepSeek 明确返回 hit=0 是"可测且未命中"，与"不可测"必须区分"""
    with usage_meter() as rec:
        record_langchain(ai_deepseek(prompt=132, completion=8, hit=0, miss=132))
    c = rec.calls[0]
    assert c.cache_supported is True
    assert (c.cache_hit_tokens, c.cache_miss_tokens) == (0, 132)


def test_missing_usage_is_recorded_as_unparsable():
    """调用发生了但没有 usage：仍记一行，保住 llm_calls 的真实性"""
    with usage_meter() as rec:
        record_openai(NS(usage=None, model="x"), "intent")
    assert len(rec.calls) == 1
    assert rec.calls[0].cache_supported is False
    assert rec.summary()["llm_calls"] == 1


# ── 轮级汇总：两套口径 ───────────────────────────────────────

def test_summary_separates_token_ratio_from_call_ratio():
    """核心用例：0/1000 + 900/1000
       -> Token 命中率 45%，但只有 1/2 次调用命中。
       二者不能混为一谈。"""
    with usage_meter() as rec:
        record_openai(openai_resp(prompt=1000, completion=1,
                                  prompt_cache_hit_tokens=0,
                                  prompt_cache_miss_tokens=1000), "a")
        record_openai(openai_resp(prompt=1000, completion=1,
                                  prompt_cache_hit_tokens=900,
                                  prompt_cache_miss_tokens=100), "b")
    s = rec.summary()
    assert s["cache_hit_tokens"] == 900
    assert s["cache_miss_tokens"] == 1100
    assert s["cache_hit_rate"] == pytest.approx(0.45)
    assert s["cache_hit_calls"] == 1
    assert s["cache_measured_calls"] == 2
    assert s["cache_supported"] is True


def test_unmeasurable_calls_excluded_from_cache_denominator():
    """部分阶段缺 cache 字段，不应把有效命中率一起抹掉"""
    with usage_meter() as rec:
        record_openai(openai_resp(prompt=1000, completion=1,
                                  prompt_cache_hit_tokens=800,
                                  prompt_cache_miss_tokens=200), "generate")
        record_langchain(ai_usage_metadata_only(prompt=1000, completion=10), "rewrite")
    s = rec.summary()
    # token 总量覆盖全部调用
    assert s["prompt_tokens"] == 2000
    assert s["llm_calls"] == 2
    # 缓存口径只覆盖可测的那一次
    assert s["cache_measured_calls"] == 1
    assert s["cache_supported"] is False
    assert s["cache_hit_rate"] == pytest.approx(0.8)


def test_zero_denominator_yields_zero_rate_not_error():
    with usage_meter() as rec:
        record_langchain(ai_usage_metadata_only(prompt=0, completion=0, cache_read=0))
    s = rec.summary()
    assert s["cache_hit_rate"] == 0.0
    assert s["cache_measured_calls"] == 1


def test_no_calls_at_all_is_not_supported():
    rec = UsageRecorder()
    s = rec.summary()
    assert s["llm_calls"] == 0
    assert s["cache_supported"] is False
    assert s["cache_hit_rate"] == 0.0


# ── stage 归因 ───────────────────────────────────────────────

def test_repeatable_stage_gets_step_suffix():
    rec = UsageRecorder()
    for _ in range(3):
        rec.add(stage="react", cache_supported=False)
    rec.add(stage="generate", cache_supported=False)
    assert [c.stage for c in rec.calls] == [
        "react_step1", "react_step2", "react_step3", "generate",
    ]
    assert [c.call_index for c in rec.calls] == [1, 2, 3, 4]


def test_stage_decorator_is_restored_after_exit():
    @stage("intent")
    async def inner():
        return current_recorder()

    async def main():
        with usage_meter() as outer:
            assert await inner() is outer
            # 装饰器退出后不应残留
            with usage_meter() as second:
                assert current_recorder() is second

    asyncio.run(main())


# ── 作用域隔离 ───────────────────────────────────────────────

def test_no_recorder_is_noop():
    """没有 active recorder 时采集是 no-op —— 入库链不受影响"""
    assert current_recorder() is None
    record_langchain(ai_deepseek())
    record_openai(openai_resp())
    assert current_recorder() is None


def test_nested_meter_restores_previous_not_none():
    """内层退出必须恢复外层，而不是置 None —— 否则外层统计被破坏"""
    with usage_meter() as outer:
        record_langchain(ai_deepseek(), "outer_call")
        with usage_meter() as inner:
            record_langchain(ai_deepseek(), "inner_call")
            assert current_recorder() is inner
        # 关键断言：外层仍然有效
        assert current_recorder() is outer
        record_langchain(ai_deepseek(), "outer_call_2")
    assert len(outer.calls) == 2
    assert [c.stage for c in outer.calls] == ["outer_call", "outer_call_2"]


def test_recorder_cleared_after_exit():
    with usage_meter():
        assert current_recorder() is not None
    assert current_recorder() is None


# ── 缓存拆解（① 总命中 / ② 稳定前缀复用 / ③ 检索上下文复用）──

class _Sections:
    """最小化的 PromptSections 替身（避免测试依赖 prompt_builder）"""

    def __init__(self, stable, metadata=0, context=0, query=0):
        self.stable_chars = stable
        self.metadata_chars = metadata
        self.context_chars = context
        self.query_chars = query

    @property
    def dynamic_chars(self):
        return self.metadata_chars + self.context_chars + self.query_chars

    @property
    def total_chars(self):
        return self.stable_chars + self.dynamic_chars


def test_sections_split_tokens_proportionally():
    """区段 token 按字符比例摊分真实 prompt_tokens"""
    with usage_meter() as rec:
        record_prompt_sections(_Sections(stable=100, metadata=0, context=300, query=100))
        # prompt=1000, 总字符 500 -> 稳定 200 / 上下文 600 / 问题 200
        record_langchain(
            ai_deepseek(prompt=1000, completion=1, hit=0, miss=1000), "generate"
        )
    c = rec.calls[0]
    assert (c.stable_prefix_tokens, c.context_tokens, c.query_tokens) == (200, 600, 200)


def test_sections_apply_only_to_the_next_call():
    """区段只对紧接着的一次调用生效，不串到后面的调用上"""
    with usage_meter() as rec:
        record_prompt_sections(_Sections(stable=100, context=100))
        record_langchain(ai_deepseek(prompt=200, completion=1), "generate")
        record_langchain(ai_deepseek(prompt=200, completion=1), "other")
    assert rec.calls[0].stable_prefix_tokens == 100
    assert rec.calls[1].stable_prefix_tokens == 0


def test_summary_decomposes_cache_rates():
    """① 总命中率 / ② 稳定前缀复用率 / ③ 上下文复用率 三者口径分开"""
    with usage_meter() as rec:
        record_prompt_sections(_Sections(stable=200, context=800, query=0))
        # prompt=1000, 字符 1000 -> 稳定 200 / 上下文 800；命中 400
        record_langchain(
            ai_deepseek(prompt=1000, completion=1, hit=400, miss=600), "generate"
        )
    s = rec.summary()
    assert s["cache_hit_rate"] == pytest.approx(0.4)          # ① 400/1000
    assert s["stable_prefix_tokens"] == 200
    assert s["context_tokens"] == 800
    # ② 稳定前缀 200 全部命中 -> 100%
    assert s["stable_prefix_reuse_rate"] == pytest.approx(1.0)
    # ③ 剩余 200 命中算作上下文复用 -> 200/800 = 25%
    assert s["rag_context_reuse_rate"] == pytest.approx(0.25)


def test_context_reuse_cannot_exceed_context_tokens():
    """命中量超过"稳定前缀+上下文"时不得让复用率 > 1"""
    with usage_meter() as rec:
        record_prompt_sections(_Sections(stable=100, context=100))
        record_langchain(
            ai_deepseek(prompt=200, completion=1, hit=200, miss=0), "generate"
        )
    s = rec.summary()
    assert s["stable_prefix_reuse_rate"] == pytest.approx(1.0)
    assert s["rag_context_reuse_rate"] == pytest.approx(1.0)


def test_rates_are_none_when_no_section_data():
    """没有区段信息时，②③ 必须是 None（前端显示 —），不能编造 0"""
    with usage_meter() as rec:
        record_langchain(ai_deepseek(prompt=100, completion=1, hit=50, miss=50), "x")
    s = rec.summary()
    assert s["stable_prefix_reuse_rate"] is None
    assert s["rag_context_reuse_rate"] is None
    assert s["cache_hit_rate"] == pytest.approx(0.5)


def test_cache_creation_is_none_not_zero_for_deepseek():
    """DeepSeek 不提供缓存写入数值 -> None，绝不填 0（0 会被读成"确实没写入"）"""
    with usage_meter() as rec:
        record_langchain(ai_deepseek(prompt=100, completion=1, hit=0, miss=100))
    s = rec.summary()
    assert s["cache_creation_tokens"] is None
    assert rec.calls[0].cache_creation_tokens is None


def test_cache_creation_passthrough_when_provided():
    """Provider 真给了写入数值时要原样带出（Anthropic 风格）"""
    resp = NS(
        response_metadata={},
        usage_metadata={
            "input_tokens": 100, "output_tokens": 5, "total_tokens": 105,
            "input_token_details": {"cache_read": 40, "cache_creation": 60},
        },
    )
    with usage_meter() as rec:
        record_langchain(resp)
    s = rec.summary()
    assert s["cache_creation_tokens"] == 60
