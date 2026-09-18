"""探测 reranker 可用性（不修改任何生产数据）。"""
import asyncio
import os
import sys
from pathlib import Path

_PY = Path(Path(__file__).resolve().parent.parent / "python")
os.chdir(_PY)
sys.path.insert(0, str(_PY))
sys.stdout.reconfigure(encoding="utf-8")

from config import settings
from services.reranker import reranker


async def main():
    print("rerank_model =", settings.rerank_model)
    print("base_url     =", settings.effective_embedding_base_url)
    print("enabled      =", reranker.enabled)
    print()

    q = "年假有多少天？"
    docs = [
        "年休假管理办法\n章节：3. 年假天数\n累计工作年限 | 每年年假天数 满 1 年不满 10 年 5 天 满 10 年不满 20 年 10 天",
        "印章与证照管理制度\n章节：2. 印章范围\n公司印章包括公章、合同专用章、财务专用章",
        "员工考勤管理制度\n章节：2. 工作时间\n标准工作时间为周一至周五 9:00 至 18:00",
    ]
    try:
        res = await reranker.rerank(q, docs, top_k=3)
        print("rerank 返回:", res)
        for idx, sc in res:
            print(f"   [{idx}] {sc:.4f}  {docs[idx][:40]}")
    except Exception as e:
        print("rerank 失败:", type(e).__name__, e)
    finally:
        await reranker.close()


asyncio.run(main())
