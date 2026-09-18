"""测试硅基流动的 rerank 端点是否可用，并找出正确的模型名"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
sys.stdout.reconfigure(encoding="utf-8")

import httpx

from config import settings

BASE = settings.effective_embedding_base_url
KEY = settings.effective_embedding_api_key

print(f"=== 端点: {BASE} ===")

MODELS = [
    "BAAI/bge-reranker-v2-m3",
    "BAAI/bge-reranker-large",
    "netease-youdao/bce-reranker-base_v1",
    "Qwen/Qwen3-Reranker-8B",
]

QUERY = "员工申诉流程是什么样的"
DOCS = [
    "【员工申诉与沟通制度】申诉人应在知悉结果后 10 个工作日内提交书面申诉及证据。HR 组织调查。",
    "【固定资产管理制度】达到使用年限且无法继续使用的资产，由使用人提出报废申请。",
    "【薪酬福利管理办法】薪酬数据仅限本人和相关管理人员查看，禁止打听或传播。",
    "【员工考勤管理制度】上班时间后 5 分钟内到岗视为迟到。",
]

with httpx.Client(base_url=BASE, timeout=40,
                  headers={"Authorization": f"Bearer {KEY}",
                           "Content-Type": "application/json"}) as c:
    for m in MODELS:
        for path in ("/rerank", "/v1/rerank"):
            try:
                r = c.post(path, json={
                    "model": m, "query": QUERY, "documents": DOCS,
                    "top_n": 3, "return_documents": False,
                })
                if r.status_code == 200:
                    data = r.json()
                    res = data.get("results") or data.get("data") or []
                    print(f"\n  [OK] {path}  model={m}")
                    for item in res:
                        idx = item.get("index")
                        sc = item.get("relevance_score", item.get("score"))
                        print(f"        idx={idx} score={sc:.4f}  {DOCS[idx][:40]}")
                    sys.exit(0)
                else:
                    code = r.status_code
                    msg = r.text[:90].replace("\n", " ")
                    print(f"  [{code}] {path} {m}: {msg}")
            except Exception as e:
                print(f"  [ERR] {path} {m}: {str(e)[:70]}")

print("\n所有组合均失败")
