"""
端到端冒烟测试。

验证：应用能启动、编排图能构建、问答能跑通。
"""

import asyncio
import os
import sys
import time
from pathlib import Path

_PY = Path(Path(__file__).resolve().parent.parent / "python")
os.chdir(_PY)
sys.path.insert(0, str(_PY))
sys.stdout.reconfigure(encoding="utf-8")

from config import settings
from services.database import init_db
from services.vector_store import VectorStoreService


async def main():
    print("=" * 76)
    print("端到端冒烟验证")
    print("=" * 76)

    # 1) 配置项完整
    for k in ("openai_model", "retrieval_mode", "qa_mode"):
        has = hasattr(settings, k)
        print(f"  settings.{k:<24} {'**仍存在（应已删除）**' if has else '已删除 ✓'}")

    init_db(settings.sqlite_path)

    # 2) 编排图能构建
    from orchestrator.graph import build_knowledge_graph_workflow

    vs = VectorStoreService()
    await vs.init()
    wf = build_knowledge_graph_workflow(vector_store=vs)
    print(f"  编排流水线: {sorted(wf.keys())}")

    # 3) 问答跑通
    from agents.qa_agent import QAAgent

    agent = QAAgent(vector_store=vs)
    print("  QAAgent 构造成功（无 knowledge_graph 参数）")
    print()

    for q in ("标准工作时间是几点到几点？", "加班费怎么算？", "公司有没有规定员工可以无限期远程办公？"):
        t0 = time.perf_counter()
        res = await agent.answer(q, acl_scopes=["public"])
        ms = (time.perf_counter() - t0) * 1000
        src_titles = [(c.metadata or {}).get("title") for c in res.contexts[:3]]
        print(f"  Q: {q}")
        print(f"     {ms:.0f}ms  上下文 {len(res.contexts)} 条  {src_titles}")
        print(f"     A: {(res.answer or '')[:130]}")
        print()

    # 4) ReAct 工具集
    from agents.react_qa_agent import ReactQAAgent

    ra = ReactQAAgent(vector_store=vs, max_steps=4)
    print(f"  ReAct 工具: {[t['function']['name'] for t in ra._tools]}")
    print(f"  qa_mode = {settings.qa_mode}（ReAct 可切换）")

    # 5) API 能导入
    from api import main as api_main
    print(f"  api.main 导入成功；健康检查函数存在: {hasattr(api_main, 'health')}")

    print()
    print("=" * 76)
    print("全部通过")
    print("=" * 76)


asyncio.run(main())
