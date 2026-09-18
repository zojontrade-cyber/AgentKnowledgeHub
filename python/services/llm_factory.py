"""
LLM 客户端工厂 —— 统一的惰性构造与超时/重试配置

为什么需要这一层
----------------
原实现里 4 个 Agent 都在 `__init__` 中立即构造 `ChatOpenAI(...)`：

    class DocParserAgent:
        def __init__(self):
            self.llm = ChatOpenAI(model=..., api_key=settings.openai_api_key, ...)

`ChatOpenAI` 在**构造时**就要求 api_key 非空，否则抛
`openai.OpenAIError: Missing credentials`。后果：
  - 没配 Key 时，服务连启动都做不到 —— 哪怕只是想看一眼健康检查或路由表
  - 存储层（不需要 LLM）也被一起拖死
  - 单元测试无法在无凭据环境下导入这些模块

改为惰性构造：只在真正要调用模型时才创建客户端，让「未配置 Key」的报错
发生在使用点，而不是导入/启动点。
"""

from __future__ import annotations

import logging
from typing import Any

from config import settings
from services.usage_meter import record_langchain

logger = logging.getLogger(__name__)


class MissingCredentialsError(RuntimeError):
    """未配置 LLM 凭据时抛出 —— 带明确可操作的提示"""


def build_chat_llm(temperature: float = 0) -> Any:
    """
    构造 ChatOpenAI 客户端（惰性调用）。

    仅在使用点调用本函数，不要在 __init__ 里调用。
    """
    if not settings.openai_api_key:
        raise MissingCredentialsError(
            "未配置 OPENAI_API_KEY，无法调用 LLM。\n"
            "请在 python/.env 中填写 OPENAI_API_KEY 后重启服务。\n"
            "（若使用兼容接口，同时设置 OPENAI_BASE_URL 与 OPENAI_MODEL）"
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=temperature,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )


class LazyLLM:
    """
    惰性 LLM 包装器：首次访问时才真正创建客户端。

    用法：
        self.llm = LazyLLM()          # __init__ 中安全，不做任何校验
        resp = await self.llm.ainvoke(messages)   # 此处才需要凭据
        structured = self.llm.client.with_structured_output(Schema)
    """

    def __init__(self, temperature: float = 0) -> None:
        self._temperature = temperature
        self._client: Any = None

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = build_chat_llm(self._temperature)
        return self._client

    @property
    def configured(self) -> bool:
        """是否已配置凭据（不触发构造）"""
        return bool(settings.openai_api_key)

    async def ainvoke(self, messages: Any, **kwargs: Any) -> Any:
        resp = await self.client.ainvoke(messages, **kwargs)
        # 计量：无 active recorder 时是 no-op（入库链、脚本、单测不受影响）。
        # stage 由上层 @stage(...) 装饰器通过 ContextVar 提供，本层不做归因。
        record_langchain(resp)
        return resp

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        resp = self.client.invoke(messages, **kwargs)
        record_langchain(resp)
        return resp

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        return self.client.with_structured_output(schema, **kwargs)
