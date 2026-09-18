"""实测 LLM 与 embedding 配置是否可用"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
sys.stdout.reconfigure(encoding="utf-8")

from config import settings

print("=== 当前配置 ===")
print(f"  对话端点:   {settings.openai_base_url}")
print(f"  对话模型:   {settings.openai_model}")
print(f"  Embedding端点: {settings.effective_embedding_base_url}")
print(f"  Embedding模型: {settings.embedding_model}")
print(f"  分离配置:   {settings.embedding_uses_separate_endpoint}")
print()

# ── 1. 测试对话模型 ─────────────────────────────────────────
print("=== 1. 测试对话模型 ===")
try:
    from openai import OpenAI

    client = OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        timeout=30,
    )
    resp = client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": "回复两个字：成功"}],
        max_tokens=20,
    )
    print(f"  [OK] 对话可用，回复: {resp.choices[0].message.content!r}")
except Exception as e:
    print(f"  [FAIL] {type(e).__name__}: {str(e)[:200]}")

print()

# ── 2. 测试意图识别（需要 function calling）──────────────────
print("=== 2. 测试结构化输出（意图识别依赖）===")
try:
    from langchain_openai import ChatOpenAI
    from pydantic import BaseModel, Field

    class Intent(BaseModel):
        intent: str = Field(description="意图类别")

    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=0,
        timeout=30,
    )
    structured = llm.with_structured_output(Intent)
    r = structured.invoke("张三和李四有什么区别？")
    print(f"  [OK] 结构化输出可用: {r}")
except Exception as e:
    print(f"  [FAIL] {type(e).__name__}: {str(e)[:200]}")
    print("         -> 意图识别会降级为规则匹配")

print()

# ── 3. 测试 embedding（用 effective_* 分离配置）──────────────
print("=== 3. 测试 embedding ===")
try:
    from openai import OpenAI

    client = OpenAI(
        api_key=settings.effective_embedding_api_key,
        base_url=settings.effective_embedding_base_url,
        timeout=30,
    )
    resp = client.embeddings.create(
        model=settings.embedding_model,
        input="测试文本",
    )
    dim = len(resp.data[0].embedding)
    print(f"  [OK] embedding 可用，模型 {settings.embedding_model}，维度: {dim}")
except Exception as e:
    print(f"  [FAIL] {type(e).__name__}: {str(e)[:200]}")
    print("         -> 向量检索不可用（无法入库/检索）")
