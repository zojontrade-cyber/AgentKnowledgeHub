"""
对话历史（展示保留）回归测试

覆盖三件事：
  1. services/qa_payload.py 的快照构造 —— 尤其**引用契约**（必须存命中章节，
     不能存 small-to-big 展开后的父文档，否则恢复出来的引用无法核对）
  2. services/database.py 的 qa_messages 读写 —— 顺序、actor 隔离、limit、坏数据
  3. TTL 清理的边界 —— `days<=0` 必须是"永久保留"而不是"删光"
"""

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from services.database import Database  # noqa: E402
from services.qa_payload import (  # noqa: E402
    ANSWER_MAX_CHARS,
    PAYLOAD_VERSION,
    build_turn_payload,
    serialize_sources,
)


# ── 夹具 ─────────────────────────────────────────────────────

@pytest.fixture()
def db(tmp_path):
    return Database(str(tmp_path / "history.db"))


class _Ctx:
    """RetrievedContext 的最小替身（只带 payload 需要的字段）"""

    def __init__(self, content, llm_context="", title="T", section="S",
                 source="/d/a.md", score=0.9, retrieval_type="vector"):
        self.content = content
        self.llm_context = llm_context
        self.metadata = {"title": title, "section_title": section}
        self.source = source
        self.score = score
        self.retrieval_type = retrieval_type

    @property
    def display_content(self):
        return self.content

    @property
    def title(self):
        return self.metadata["title"]

    @property
    def section(self):
        return self.metadata["section_title"]

    @property
    def parent_available(self):
        return bool(self.llm_context)


def make_result(*, answer="答案", question="问题？", intent="factoid",
                contexts=None):
    return NS(
        question=question,
        answer=answer,
        confidence=0.5,
        intent=NS(value=intent),
        contexts=contexts if contexts is not None else [_Ctx("命中章节正文")],
        reasoning_steps=["step1"],
    )


def sample_usage():
    return {
        "llm_calls": 3, "prompt_tokens": 100, "completion_tokens": 10,
        "total_tokens": 110, "cache_hit_tokens": 40, "cache_miss_tokens": 60,
        "cache_measured_calls": 3, "cache_hit_calls": 1, "cache_supported": True,
        "cache_hit_rate": 0.4, "stable_prefix_reuse_rate": 1.0,
        "rag_context_reuse_rate": 0.2, "latency_ms": 1234.5,
        "per_call": [{"call_index": 1, "stage": "intent", "prompt_tokens": 100,
                      "completion_tokens": 10, "total_tokens": 110,
                      "cache_hit_tokens": 40, "cache_miss_tokens": 60,
                      "cache_supported": True, "model": "m",
                      "cache_creation_tokens": None,
                      "stable_prefix_tokens": 0, "context_tokens": 0,
                      "query_tokens": 0}],
    }


# ── 1. 快照构造 ──────────────────────────────────────────────

def test_payload_has_required_shape():
    p = build_turn_payload(make_result(), usage=sample_usage(), turn=2,
                           trace_id="t1", metrics_degraded=False)
    for key in ("question", "answer", "confidence", "intent", "sources",
                "reasoning_steps", "turn", "trace_id", "usage",
                "metrics_degraded", "payload_version"):
        assert key in p, f"缺少字段 {key}"
    assert p["turn"] == 2 and p["trace_id"] == "t1"
    assert p["payload_version"] == PAYLOAD_VERSION
    assert p["intent"] == "factoid"


def test_payload_is_json_serializable():
    p = build_turn_payload(make_result(), usage=sample_usage(), turn=1)
    again = json.loads(json.dumps(p, ensure_ascii=False))
    assert again == p


def test_sources_use_hit_section_not_parent_document():
    """**引用契约**：存下来供展示/核对的必须是命中章节。

    若误存 llm_context（父文档全文），恢复出来的每条引用都会以
    "## 1. 文档说明" 开头，用户无法核对出处
    —— 这正是早期引用准确率只有 44.4% 的原因。
    """
    ctx = _Ctx("命中的那一小节", llm_context="## 1. 文档说明\n父文档全文……")
    p = build_turn_payload(make_result(contexts=[ctx]), turn=1)
    assert p["sources"][0]["content"] == "命中的那一小节"
    assert "文档说明" not in p["sources"][0]["content"]


def test_source_excerpt_is_truncated():
    long_ctx = _Ctx("x" * 2000)
    p = build_turn_payload(make_result(contexts=[long_ctx]), turn=1)
    assert len(p["sources"][0]["content"]) == 400


def test_source_carries_parent_available_flag():
    a = _Ctx("小章节", llm_context="父文档")
    b = _Ctx("小章节2", llm_context="")
    p = build_turn_payload(make_result(contexts=[a, b]), turn=1)
    assert p["sources"][0]["parent_available"] is True
    assert p["sources"][1]["parent_available"] is False


def test_overlong_answer_truncated():
    p = build_turn_payload(make_result(answer="y" * (ANSWER_MAX_CHARS + 500)),
                           turn=1)
    assert len(p["answer"]) == ANSWER_MAX_CHARS


def test_serialize_sources_handles_empty():
    assert serialize_sources([]) == []


# ── 2. 读写 ─────────────────────────────────────────────────

def record(db, session_id="s1", actor="k1", turn=1, question="q"):
    payload = build_turn_payload(make_result(question=question),
                                usage=sample_usage(), turn=turn)
    return db.record_qa_message(actor=actor, session_id=session_id, turn=turn,
                               payload=payload, trace_id=f"t{turn}")


def test_roundtrip_preserves_payload(db):
    record(db, question="第一问", turn=1)
    turns, total = db.list_qa_messages("s1", "k1")
    assert total == 1 and len(turns) == 1
    assert turns[0]["question"] == "第一问"
    assert turns[0]["answer"] == "答案"
    assert turns[0]["usage"]["total_tokens"] == 110
    assert turns[0]["sources"][0]["content"] == "命中章节正文"


def test_turns_returned_in_ascending_order(db):
    for i in (1, 2, 3):
        record(db, turn=i, question=f"q{i}")
    turns, total = db.list_qa_messages("s1", "k1")
    assert [t["turn"] for t in turns] == [1, 2, 3]
    assert total == 3


def test_limit_returns_most_recent_but_ascending(db):
    for i in range(1, 6):
        record(db, turn=i, question=f"q{i}")
    turns, total = db.list_qa_messages("s1", "k1", limit=2)
    assert [t["turn"] for t in turns] == [4, 5]   # 最近的 N 轮，仍按升序
    assert total == 5                              # total 用于提示"仅显示最近 N 轮"


def test_isolated_by_actor(db):
    """只按 session_id 查等于"猜到随机 id 就能读到别人的问答内容与出处" """
    record(db, session_id="shared", actor="k1")
    turns, total = db.list_qa_messages("shared", "k2")
    assert turns == [] and total == 0
    assert db.list_qa_messages("shared", "k1")[1] == 1


def test_isolated_by_session(db):
    record(db, session_id="a", actor="k1")
    assert db.list_qa_messages("b", "k1")[1] == 0


def test_corrupt_payload_is_skipped(db):
    record(db, turn=1, question="good1")
    # 手工插入一行坏 JSON
    with db.tx() as c:
        c.execute(
            "INSERT INTO qa_messages (ts, trace_id, actor, session_id, turn,"
            " payload_version, payload) VALUES (?,?,?,?,?,?,?)",
            (time.time(), "tbad", "k1", "s1", 2, 1, "{不是合法 JSON"),
        )
    record(db, turn=3, question="good3")
    turns, total = db.list_qa_messages("s1", "k1")
    assert [t["question"] for t in turns] == ["good1", "good3"]
    assert total == 3   # total 仍是真实行数（含坏行），前端只提示截断


def test_limit_is_clamped(db):
    record(db)
    turns, _ = db.list_qa_messages("s1", "k1", limit=0)
    assert len(turns) == 1          # 至少 1
    turns, _ = db.list_qa_messages("s1", "k1", limit=10_000)
    assert len(turns) == 1          # 上限不做奇怪的事


# ── 3. TTL 清理 ─────────────────────────────────────────────

def _age_last_row(db, seconds: float) -> None:
    with db.tx() as c:
        c.execute("UPDATE qa_messages SET ts = ?", (time.time() - seconds,))


def test_purge_zero_days_keeps_everything(db):
    """0 = 永久保留。绝不能被解释成"删光"。"""
    record(db)
    _age_last_row(db, 10 * 86400)
    assert db.purge_qa_messages(0) == 0
    assert db.list_qa_messages("s1", "k1")[1] == 1


def test_purge_negative_days_keeps_everything(db):
    record(db)
    _age_last_row(db, 10 * 86400)
    assert db.purge_qa_messages(-1) == 0
    assert db.list_qa_messages("s1", "k1")[1] == 1


def test_purge_removes_only_expired(db):
    record(db, turn=1, question="old")
    _age_last_row(db, 40 * 86400)       # 40 天前
    record(db, turn=2, question="new")

    removed = db.purge_qa_messages(30)
    assert removed == 1
    turns, total = db.list_qa_messages("s1", "k1")
    assert total == 1
    assert turns[0]["question"] == "new"


def test_purge_boundary_keeps_recent(db):
    record(db)
    _age_last_row(db, 29 * 86400)       # 29 天 < 30 天，应保留
    assert db.purge_qa_messages(30) == 0
    assert db.list_qa_messages("s1", "k1")[1] == 1


def test_purge_does_not_touch_metrics(db):
    """清理只作用于内容表；用量行很小且是成本趋势数据，另有保留策略"""
    record(db)
    db.record_qa_metrics(
        actor="k1", session_id="s1", question="q", intent="factoid",
        qa_mode="pipeline", usage=sample_usage(), trace_id="t1",
    )
    _age_last_row(db, 100 * 86400)
    db.purge_qa_messages(30)
    assert db.qa_session_summary("s1", "k1")["turns"] == 1


def test_history_stats(db):
    record(db, turn=1)
    record(db, turn=2)
    st = db.qa_history_stats()
    assert st["rows"] == 2
    assert st["payload_bytes"] > 0
    assert db.qa_history_stats("k1")["rows"] == 2
    assert db.qa_history_stats("other")["rows"] == 0
