"""探测：LLM 响应中能拿到哪些 usage / 缓存字段。只读，不改代码。"""
import asyncio
import os
import sys
from pathlib import Path

_PY = Path(Path(__file__).resolve().parent.parent / "python")
os.chdir(_PY)
sys.path.insert(0, str(_PY))
sys.stdout.reconfigure(encoding="utf-8")

from langchain_core.messages import HumanMessage, SystemMessage

from config import settings
from services.llm_factory import LazyLLM


async def main():
    print("=" * 78)
    print("1. LangChain AIMessage（qa_agent 走这条）")
    print("=" * 78)
    llm = LazyLLM(temperature=0)
    # 连续两次相同前缀，触发缓存
    for i in (1, 2):
        msgs = [
            SystemMessage(content="你是企业知识问答助手。" * 20),
            HumanMessage(content=f"回复OK（第{i}次）"),
        ]
        resp = await llm.ainvoke(msgs)
        print(f"\n  第 {i} 次调用：")
        print(f"    type                = {type(resp).__name__}")
        um = getattr(resp, "usage_metadata", None)
        print(f"    usage_metadata      = {um}")
        rm = getattr(resp, "response_metadata", None) or {}
        print(f"    response_metadata keys = {list(rm.keys())}")
        # 深挖 token_usage
        tu = rm.get("token_usage") or rm.get("usage") or {}
        print(f"    token_usage         = {tu}")

    print()
    print("=" * 78)
    print("2. 原生 AsyncOpenAI（_classify_intent 走这条）")
    print("=" * 78)
    from openai import AsyncOpenAI

    cli = AsyncOpenAI(api_key=settings.openai_api_key,
                      base_url=settings.openai_base_url, timeout=60)
    for i in (1, 2):
        r = await cli.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": "你是查询意图分类器。" * 20},
                {"role": "user", "content": f"请假流程（第{i}次）"},
            ],
            temperature=0,
        )
        u = r.usage
        print(f"\n  第 {i} 次调用：")
        print(f"    usage 对象类型   = {type(u).__name__}")
        print(f"    prompt_tokens    = {getattr(u, 'prompt_tokens', None)}")
        print(f"    completion_tokens= {getattr(u, 'completion_tokens', None)}")
        print(f"    total_tokens     = {getattr(u, 'total_tokens', None)}")
        # DeepSeek 扩展字段
        dump = u.model_dump() if hasattr(u, "model_dump") else {}
        print(f"    model_dump()     = {dump}")


asyncio.run(main())
