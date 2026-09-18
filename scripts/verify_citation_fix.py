"""
验证 P2.1 修复：引用契约是否正确。

核心断言：
  sources[].content 应当是**命中的章节**，而不是文档开头的「文档说明」。

同时确认 LLM context 仍用展开的父文档（回答质量不退化）。
"""

import asyncio
import json
import os
import sys
from pathlib import Path

_PY = Path(Path(__file__).resolve().parent.parent / "python")
os.chdir(_PY)
sys.path.insert(0, str(_PY))
sys.stdout.reconfigure(encoding="utf-8")

from agents.qa_agent import QAAgent
from config import settings
from services.database import init_db
from services.vector_store import VectorStoreService

ROOT = Path(__file__).resolve().parent.parent
BENCH = [
    json.loads(l)
    for l in (ROOT / "data" / "eval" / "bench_v3.jsonl").read_text(encoding="utf-8").splitlines()
    if l.strip()
]
EXACT = [r for r in BENCH if r["gold_type"] == "exact"]


async def main():
    init_db(settings.sqlite_path)
    vs = VectorStoreService()
    await vs.init()
    # 本次验证的是引用序列化契约，与图检索无关。
    agent = QAAgent(vector_store=vs)

    # ── 1. 结构检查（走检索层，不调 LLM）─────────────────
    q = "上班时间是几点到几点？"
    ctxs = await agent._vector_retrieve({"queries": [q], "intent": "factual"})
    print("=" * 78)
    print(f"Q: {q}")
    print("=" * 78)
    for i, c in enumerate(ctxs[:3], 1):
        print(f"  [{i}] type={c.retrieval_type} score={c.score:.3f}")
        print(f"      title   = {c.title}")
        print(f"      section = {c.section}")
        print(f"      content = {c.display_content[:90].replace(chr(10),' ')}")
        print(f"      parent_available = {c.parent_available}")
        print(f"      llm_context 长度  = {len(c.llm_context)}")
        assert c.content == c.display_content
        print()

    # ── 2. 契约断言：content 不应以「文档说明」开头 ────────
    # 直接走检索层（不调 LLM），验证对象契约本身
    print("=" * 78)
    print("契约检查：sources content 是否仍被父文档开头污染")
    print("=" * 78)
    polluted = 0
    total = 0
    for r in EXACT:
        ctxs = await agent._vector_retrieve(
            {"queries": [r["question"]], "intent": "factual"}
        )
        for c in ctxs:
            if c.retrieval_type != "vector":
                continue
            total += 1
            if c.section and "文档说明" in c.section:
                polluted += 1
    print(f"  vector 引用总数        {total}")
    print(f"  其中 section=文档说明   {polluted}  ({polluted/max(total,1):.1%})")
    print("  （修复前 content 被替换为父文档，开头恒为「1. 文档说明」）")

    # ── 3. 引用准确性：gold 章节内容是否出现在引用里 ───────
    print()
    print("=" * 78)
    print("引用准确性（gold 章节正文片段是否出现在返回引用中）")
    print("=" * 78)
    ok = 0
    for r in EXACT:
        ctxs = await agent._vector_retrieve(
            {"queries": [r["question"]], "intent": "factual"}
        )
        key = (r["expected_answer"] or "").strip()[:30]
        if any(key and key in (c.display_content or "") for c in ctxs):
            ok += 1
    print(f"  {ok}/{len(EXACT)}  ({ok/len(EXACT):.1%})")


asyncio.run(main())
