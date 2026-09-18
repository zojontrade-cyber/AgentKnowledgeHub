"""
LLM 用量计量 —— token / 缓存命中的统一采集与归一化

为什么需要这一层
----------------
项目里调用 LLM 有**两条互不相通的路径**，它们返回的用量字段形状完全不同：

  1. LangChain 路径（`LazyLLM.ainvoke`）—— 返回 `AIMessage`
     - `response_metadata["token_usage"]`  富字段，含 DeepSeek 私有的
       `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`
     - `usage_metadata`                    只有 input/output/total
  2. 原生 OpenAI 路径（`_classify_intent`、ReAct 循环）—— 返回 `Completion`
     - `resp.usage.model_dump()`           与 (1) 的 token_usage 同源

如果让这些形状各自泄漏到上层，前端就要认识三套字段名，且任何一处
provider 差异都会变成 UI 上的特例分支。因此**所有差异在本模块收口**，
对外只提供一种形状：

    {"call_index", "stage", "model",
     "prompt_tokens", "completion_tokens", "total_tokens",
     "cache_hit_tokens", "cache_miss_tokens", "cache_supported"}

分层契约
--------
    本模块        —— 归一 + 聚合                （不落库、不返回 HTTP）
    agents/*      —— 只贴 @stage(...) 标签       （不聚合、不算率、不碰 DB）
    database.py   —— 落库 / 取号 / 汇总查询      （不理解 SDK 字段）
    api/main.py   —— 传输
    static/app.js —— 展示

这样 Pipeline / ReAct / 未来任何 Agent 都天然复用同一套设施。

为什么用 ContextVar
-------------------
`LazyLLM` 被 4 个 Agent 共用（其中 3 个属于**入库链路**）。如果用量收集是
全局变量，入库时的 LLM 调用会污染问答的统计。用 ContextVar 做请求级隔离后，
**没有 active recorder 时 record_* 全是 no-op** —— 入库链、脚本、单测
不需要任何额外处理就不会被计量。

已知局限
--------
`cache_hit_rate` 是**可测 Prompt Token 的命中占比**，不是「多少次调用命中了
缓存」。两者是不同的概念，本模块两个都给：

    cache_hit_rate    = hit_tokens / (hit_tokens + miss_tokens)   ← Token 比例
    cache_hit_calls   = 命中(hit>0)的调用次数                      ← 调用次数比例

展示时必须说清用的是哪一个。
"""

from __future__ import annotations

import functools
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

logger = logging.getLogger(__name__)


# 会重复出现的 stage：meter 为其自动合成 `_step{N}` 后缀（如 react_step2）。
# 单次出现的 stage（intent / rewrite / generate）保持原名。
# 未来新增可重复的 stage，在这里登记即可。
_REPEATABLE_STAGES = frozenset({"react"})


# ── 工具 ─────────────────────────────────────────────────────

def _as_int(v: Any) -> int:
    """任意值 → 非负整数。None / 字符串 / 异常一律归 0，绝不抛异常。"""
    if v is None:
        return 0
    try:
        n = int(v)
    except (TypeError, ValueError):
        return 0
    return n if n > 0 else 0


def _as_dict(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _pick_model(resp: Any) -> str:
    """尽力取出模型名，取不到返回空串（不编造）。"""
    meta = _as_dict(getattr(resp, "response_metadata", None))
    for candidate in (meta.get("model_name"), getattr(resp, "model", None)):
        if candidate:
            return str(candidate)
    return ""


# ── 单次调用记录（对外唯一形状）──────────────────────────────

@dataclass
class CallUsage:
    """一次 LLM 调用的归一化用量。字段语义见模块 docstring。"""

    call_index: int = 0
    stage: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    # 该次调用**是否给出了可识别的 cache 字段**。
    # 注意与「是否命中」无关：DeepSeek 会明确返回 hit=0，那是"可测且未命中"，
    # 与「该 provider 根本不报 cache 字段」必须区分开。
    cache_supported: bool = False
    # 缓存"写入"侧。DeepSeek 只提供 `prompt_tokens_details.cache_write_tokens`
    # 且恒为 None —— 没有真实数值时保持 None，**不要填 0**（0 会被误读为"确实没写入"）。
    cache_creation_tokens: int | None = None
    # prompt 按"稳定 / 动态"切分的**估算** token（依据字符数按比例摊分；
    # 精确值是 prompt_tokens，这两个字段只用于把命中量归因到区段）
    stable_prefix_tokens: int = 0
    context_tokens: int = 0
    query_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_index": self.call_index,
            "stage": self.stage,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cache_hit_tokens": self.cache_hit_tokens,
            "cache_miss_tokens": self.cache_miss_tokens,
            "cache_supported": self.cache_supported,
            "cache_creation_tokens": self.cache_creation_tokens,
            "stable_prefix_tokens": self.stable_prefix_tokens,
            "context_tokens": self.context_tokens,
            "query_tokens": self.query_tokens,
        }


@dataclass
class UsageRecorder:
    """一次问答请求内的用量累加器"""

    calls: list[CallUsage] = field(default_factory=list)
    _stage_seq: dict[str, int] = field(default_factory=dict, repr=False)

    # ── 写入 ────────────────────────────────────────────

    def add(
        self,
        *,
        stage: str | None = None,
        model: str = "",
        prompt_tokens: Any = 0,
        completion_tokens: Any = 0,
        total_tokens: Any = 0,
        cache_hit_tokens: Any = 0,
        cache_miss_tokens: Any = None,
        cache_supported: bool = False,
        cache_creation_tokens: Any = None,
        sections: Any = None,
    ) -> CallUsage:
        prompt = _as_int(prompt_tokens)
        completion = _as_int(completion_tokens)
        total = _as_int(total_tokens) or (prompt + completion)

        if cache_supported:
            hit = _as_int(cache_hit_tokens)
            # miss 缺失时用 prompt 反推，保证 hit + miss 的口径可比
            miss = _as_int(cache_miss_tokens) if cache_miss_tokens is not None else max(prompt - hit, 0)
        else:
            # 不支持 cache 的调用，其 hit/miss 一律不计入全局分母，
            # 避免把"没报字段"混进"报了但为 0"里
            hit = 0
            miss = 0

        base = (stage or "unknown").strip() or "unknown"
        seq = self._stage_seq.get(base, 0) + 1
        self._stage_seq[base] = seq
        label = f"{base}_step{seq}" if base in _REPEATABLE_STAGES else base

        # 缓存写入侧：只有 Provider 真的给了数值才记，否则保持 None
        creation = None if cache_creation_tokens is None else _as_int(cache_creation_tokens)

        # 区段 token 估算：按字符数比例摊分真实 prompt_tokens
        stable_t = context_t = query_t = 0
        if sections is not None:
            total_chars = getattr(sections, "total_chars", 0) or 0
            if total_chars > 0 and prompt > 0:
                stable_t = round(prompt * sections.stable_chars / total_chars)
                context_t = round(prompt * sections.context_chars / total_chars)
                query_t = round(prompt * sections.query_chars / total_chars)

        call = CallUsage(
            call_index=len(self.calls) + 1,
            stage=label,
            model=str(model or ""),
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            cache_hit_tokens=hit,
            cache_miss_tokens=miss,
            cache_supported=bool(cache_supported),
            cache_creation_tokens=creation,
            stable_prefix_tokens=stable_t,
            context_tokens=context_t,
            query_tokens=query_t,
        )
        self.calls.append(call)
        return call

    def add_unparsable(self, stage: str | None = None) -> CallUsage:
        """调用确实发生了，但用量字段解析失败 —— 记一条零值行并标记不可测。

        比直接丢弃更诚实：llm_calls 仍反映真实调用次数。
        """
        return self.add(stage=stage, cache_supported=False)

    # ── 读出 ────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        calls = self.calls
        measured = [c for c in calls if c.cache_supported]

        hit = sum(c.cache_hit_tokens for c in measured)
        miss = sum(c.cache_miss_tokens for c in measured)
        denom = hit + miss

        # ── 区段拆解 ─────────────────────────────────────
        # 有区段信息的调用（目前只有 generate 会上报）
        sectioned = [c for c in calls if c.stable_prefix_tokens or c.context_tokens]
        stable_tokens = sum(c.stable_prefix_tokens for c in sectioned)
        context_tokens = sum(c.context_tokens for c in sectioned)
        query_tokens = sum(c.query_tokens for c in sectioned)

        # 区段命中的估算：命中量优先归给稳定前缀，剩余才算检索上下文复用。
        # 这是**保守**归因 —— 稳定前缀在所有问题里逐字节相同，它先命中是合理假设。
        hit_in_sectioned = sum(c.cache_hit_tokens for c in sectioned)
        stable_hit = min(hit_in_sectioned, stable_tokens)
        context_hit = max(0, hit_in_sectioned - stable_tokens)

        return {
            "llm_calls": len(calls),
            # token 总量覆盖**全部**调用（含不可测 cache 的）
            "prompt_tokens": sum(c.prompt_tokens for c in calls),
            "completion_tokens": sum(c.completion_tokens for c in calls),
            "total_tokens": sum(c.total_tokens for c in calls),
            # 缓存口径只覆盖**可测**调用
            "cache_hit_tokens": hit,
            "cache_miss_tokens": miss,
            "cache_measured_calls": len(measured),
            "cache_hit_calls": sum(1 for c in measured if c.cache_hit_tokens > 0),
            # 全部调用都可测，才敢说"本轮的缓存口径是完整的"
            "cache_supported": bool(calls) and len(measured) == len(calls),
            # ① Prompt Cache Hit Rate —— Token 比例，不是调用次数比例
            "cache_hit_rate": round(hit / denom, 6) if denom else 0.0,
            # 缓存"写入"侧：DeepSeek 不提供 -> None（不要当成 0）
            "cache_creation_tokens": (
                sum(c.cache_creation_tokens or 0 for c in calls)
                if any(c.cache_creation_tokens is not None for c in calls)
                else None
            ),
            # ② 稳定前缀复用率：稳定前缀的 token 里有多少命中
            "stable_prefix_tokens": stable_tokens,
            "stable_prefix_reuse_rate": (
                round(stable_hit / stable_tokens, 6) if stable_tokens else None
            ),
            # ③ 检索上下文复用率：检索上下文的 token 里有多少命中
            "context_tokens": context_tokens,
            "rag_context_reuse_rate": (
                round(min(context_hit, context_tokens) / context_tokens, 6)
                if context_tokens else None
            ),
            "query_tokens": query_tokens,
            "per_call": [c.to_dict() for c in calls],
        }


# ── ContextVar：请求级隔离 ───────────────────────────────────

_recorder: ContextVar[UsageRecorder | None] = ContextVar("usage_recorder", default=None)
_stage: ContextVar[str | None] = ContextVar("usage_stage", default=None)
# 下一次调用所属的 prompt 区段组成（由 services/prompt_builder.py 产出）
_sections: ContextVar[Any] = ContextVar("usage_sections", default=None)


def record_prompt_sections(sections: Any) -> None:
    """登记**紧接着那一次** LLM 调用的区段组成。

    用途：把 prompt 拆成"稳定前缀 / 检索上下文 / 用户问题"，
    这样 Provider 报的 hit token 才能归因到区段，
    而不是只能看到一个混合的总命中率。
    """
    _sections.set(sections)


def _take_sections() -> Any:
    """取出并清空 —— 区段只对紧接着的一次调用生效，避免串到后面的调用上"""
    s = _sections.get()
    _sections.set(None)
    return s


@contextmanager
def usage_meter() -> Iterator[UsageRecorder]:
    """开启一个计量作用域。

    退出时**恢复 previous 而不是置 None** —— 嵌套使用时内层退出不会
    把外层 recorder 一起破坏：

        with usage_meter():          # 外层
            ...
            with usage_meter():      # 内层
                ...
            # 这里外层仍然有效
    """
    rec = UsageRecorder()
    previous = _recorder.get()
    _recorder.set(rec)
    try:
        yield rec
    finally:
        _recorder.set(previous)


def current_recorder() -> UsageRecorder | None:
    return _recorder.get()


def current_stage() -> str | None:
    return _stage.get()


def stage(name: str) -> Callable:
    """给 async 方法贴计量标签 —— agent 侧唯一需要写的一行。

    只做标签，不做聚合。meter 据此把调用归因到 intent / rewrite / generate /
    react_step{N}，agent 内部不出现任何统计逻辑。
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            previous = _stage.get()
            _stage.set(name)
            try:
                return await fn(*args, **kwargs)
            finally:
                _stage.set(previous)

        return wrapper

    return decorator


# ── 两个 SDK 的取值适配 ──────────────────────────────────────

def extract_langchain(resp: Any) -> dict[str, Any]:
    """从 LangChain `AIMessage` 提取归一化用量。

    取值优先级（`scripts/probe_usage_fields.py` 实测）：
      1. `response_metadata["token_usage"]` —— 富字段，DeepSeek 的
         `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens` 在这里
      2. `usage_metadata` —— 只有 input/output/total；
         `input_token_details.cache_read` 存在时视作可测
    """
    meta = _as_dict(getattr(resp, "response_metadata", None))
    model = _pick_model(resp)

    tu = _as_dict(meta.get("token_usage"))
    if tu:
        return _from_openai_style(tu, model)

    um = _as_dict(getattr(resp, "usage_metadata", None))
    if um:
        prompt = _as_int(um.get("input_tokens"))
        completion = _as_int(um.get("output_tokens"))
        details = _as_dict(um.get("input_token_details"))
        # 仅在字段确实存在时才算"可测"
        if "cache_read" in details:
            hit = _as_int(details.get("cache_read"))
            return {
                "model": model,
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": _as_int(um.get("total_tokens")),
                "cache_hit_tokens": hit,
                "cache_miss_tokens": max(prompt - hit, 0),
                "cache_supported": True,
                # Anthropic 风格的 cache_creation；DeepSeek 不提供
                "cache_creation_tokens": details.get("cache_creation"),
            }
        return {
            "model": model,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": _as_int(um.get("total_tokens")),
            "cache_supported": False,
            "cache_creation_tokens": details.get("cache_creation"),
        }

    return {"model": model, "cache_supported": False}


def extract_openai(resp: Any) -> dict[str, Any]:
    """从原生 OpenAI `Completion` 提取归一化用量（`resp.usage`）。"""
    model = _pick_model(resp)
    usage = getattr(resp, "usage", None)
    if usage is None:
        return {"model": model, "cache_supported": False}

    if hasattr(usage, "model_dump"):
        data = _as_dict(usage.model_dump())
    else:  # pragma: no cover - 兜底：极端情况下退化为属性读取
        data = {
            k: getattr(usage, k, None)
            for k in ("prompt_tokens", "completion_tokens", "total_tokens",
                      "prompt_cache_hit_tokens", "prompt_cache_miss_tokens",
                      "prompt_tokens_details")
        }
    return _from_openai_style(data, model)


def _from_openai_style(data: dict[str, Any], model: str) -> dict[str, Any]:
    """OpenAI / DeepSeek 风格字段 → 归一化字典（两个 SDK 共用）。"""
    prompt = _as_int(data.get("prompt_tokens"))
    completion = _as_int(data.get("completion_tokens"))
    total = _as_int(data.get("total_tokens"))
    hit_raw = data.get("prompt_cache_hit_tokens")
    miss_raw = data.get("prompt_cache_miss_tokens")
    details = _as_dict(data.get("prompt_tokens_details"))
    # 缓存"写入"侧：DeepSeek 有这个字段但恒为 None -> 保持 None，不填 0
    creation = details.get("cache_write_tokens")

    if hit_raw is None and miss_raw is None:
        # 回退：标准 OpenAI 风格的 prompt_tokens_details.cached_tokens
        if "cached_tokens" in details and details.get("cached_tokens") is not None:
            hit = _as_int(details.get("cached_tokens"))
            return {
                "model": model, "prompt_tokens": prompt,
                "completion_tokens": completion, "total_tokens": total,
                "cache_hit_tokens": hit,
                "cache_miss_tokens": max(prompt - hit, 0),
                "cache_supported": True,
                "cache_creation_tokens": creation,
            }
        return {
            "model": model, "prompt_tokens": prompt,
            "completion_tokens": completion, "total_tokens": total,
            "cache_supported": False,
            "cache_creation_tokens": creation,
        }

    hit = _as_int(hit_raw)
    miss = _as_int(miss_raw) if miss_raw is not None else max(prompt - hit, 0)
    return {
        "model": model, "prompt_tokens": prompt,
        "completion_tokens": completion, "total_tokens": total,
        "cache_hit_tokens": hit, "cache_miss_tokens": miss,
        "cache_supported": True,
        "cache_creation_tokens": creation,
    }


# ── 采集入口（无 recorder 时全部 no-op）──────────────────────

def record_langchain(resp: Any, stage_name: str | None = None) -> None:
    """记录一次 LangChain 调用。解析失败也记零值行（保留调用次数）。"""
    rec = _recorder.get()
    if rec is None:
        return
    label = stage_name or _stage.get()
    try:
        rec.add(stage=label, sections=_take_sections(), **extract_langchain(resp))
    except Exception:
        logger.warning("LangChain 用量解析失败，记为零值", exc_info=True)
        rec.add_unparsable(label)


def record_openai(resp: Any, stage_name: str | None = None) -> None:
    """记录一次原生 OpenAI 调用。解析失败也记零值行。"""
    rec = _recorder.get()
    if rec is None:
        return
    label = stage_name or _stage.get()
    try:
        rec.add(stage=label, sections=_take_sections(), **extract_openai(resp))
    except Exception:
        logger.warning("OpenAI 用量解析失败，记为零值", exc_info=True)
        rec.add_unparsable(label)
