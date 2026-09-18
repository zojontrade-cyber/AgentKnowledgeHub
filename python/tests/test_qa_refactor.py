"""
pytest 收集入口 —— 复用 test_refactor_offline 的断言逻辑

运行: cd python && pytest tests/ -v
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from agents.qa_agent import (  # noqa: E402
    INTENT_RETRIEVAL_PLAN,
    RetrievedContext,
    QAAgent,
    QueryIntent,
)


# ── 意图 ─────────────────────────────────────────────────────

@pytest.mark.parametrize("intent", list(QueryIntent))
def test_every_intent_has_plan(intent):
    assert intent in INTENT_RETRIEVAL_PLAN


def test_intent_plans_currently_share_one_retriever():
    """现状：五种意图的检索动作**完全相同**（只有 BM25 一路）。

    这条断言刻意记录一个**已知局限**：意图目前只影响答案组织方式，
    不影响检索路径。将来若真的为不同意图接上不同检索器，本测试会失败 ——
    那正是提示"该认知该更新了"的信号，请同步更新它。
    """
    plans = {json.dumps(INTENT_RETRIEVAL_PLAN[i], sort_keys=True)
             for i in QueryIntent}
    assert len(plans) == 1


@pytest.mark.parametrize(
    "question,expected",
    [
        ("张三和李四有什么区别", QueryIntent.COMPARATIVE),
        ("这个流程怎么做", QueryIntent.PROCEDURAL),
        ("系统里有哪些模块", QueryIntent.EXPLORATORY),
        ("为什么会这样", QueryIntent.ANALYTICAL),
        ("张三的职位是什么", QueryIntent.FACTOID),
    ],
)
def test_fallback_intent_classification(question, expected):
    assert QAAgent._classify_intent_fallback(question) == expected


# ── 重排序 ────────────────────────────────────────────────

def test_hybrid_rerank_dedupes_same_content():
    """同内容去重应生效"""
    ctxs = [
        RetrievedContext("v", "s", 0.8, "vector"),
        RetrievedContext("v", "s", 0.5, "vector"),  # 重复内容
        RetrievedContext("w", "s", 0.3, "vector"),
    ]
    ranked = QAAgent._hybrid_rerank(ctxs)
    assert len(ranked) == 2
    assert ranked[0].content == "v"
    assert ranked[0].score == pytest.approx(0.8)

def test_hybrid_rerank_sorts_desc():
    ctxs = [
        RetrievedContext("a", "s", 0.2, "vector"),
        RetrievedContext("b", "s", 0.9, "vector"),
    ]
    ranked = QAAgent._hybrid_rerank(ctxs)
    assert [c.content for c in ranked] == ["b", "a"]
