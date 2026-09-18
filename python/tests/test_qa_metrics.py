"""
问答用量持久化回归测试

覆盖 services/database.py 的 qa_metrics / qa_metrics_calls：
  - 轮次由服务端按 session_id + actor 递增取号（客户端不上报轮次）
  - 会话累计的 token 与缓存口径
  - **按 actor 隔离**：不同 Key 读不到彼此的会话与 trace
  - trace_id 下钻到单次调用，且按 actor 隔离
  - 两张表在一次事务内写入（轮表 + 调用表）
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from services.database import Database  # noqa: E402


# ── 夹具 ─────────────────────────────────────────────────────

@pytest.fixture()
def db(tmp_path):
    return Database(str(tmp_path / "metrics.db"))


def make_usage(total=1000, hit=400, miss=600, calls=2):
    """构造 usage_meter.summary() 形状的字典"""
    return {
        "llm_calls": calls,
        "prompt_tokens": total,
        "completion_tokens": 50,
        "total_tokens": total + 50,
        "cache_hit_tokens": hit,
        "cache_miss_tokens": miss,
        "cache_measured_calls": calls,
        "cache_hit_calls": 1 if hit else 0,
        "cache_supported": True,
        "cache_hit_rate": round(hit / (hit + miss), 6) if (hit + miss) else 0.0,
        "per_call": [
            {
                "call_index": i + 1, "stage": f"stage{i + 1}", "model": "deepseek-chat",
                "prompt_tokens": 500, "completion_tokens": 25, "total_tokens": 525,
                "cache_hit_tokens": 200, "cache_miss_tokens": 300,
                "cache_supported": True,
            }
            for i in range(calls)
        ],
    }


def record(db, session_id="sess-1", actor="key_a", usage=None, trace_id="t1"):
    return db.record_qa_metrics(
        actor=actor, session_id=session_id, question="q?", intent="factoid",
        qa_mode="pipeline", usage=usage or make_usage(),
        latency_ms=1234.5, trace_id=trace_id,
    )


# ── 轮次取号 ─────────────────────────────────────────────────

def test_turn_increments_per_session(db):
    assert record(db)[1] == 1
    assert record(db)[1] == 2
    assert record(db)[1] == 3


def test_turn_is_scoped_by_session(db):
    """不同会话各自从 1 开始"""
    assert record(db, session_id="sess-A")[1] == 1
    assert record(db, session_id="sess-B")[1] == 1
    assert record(db, session_id="sess-A")[1] == 2


def test_turn_is_scoped_by_actor(db):
    """同一 session_id 被不同 actor 使用时不共享轮次（防串号）"""
    assert record(db, session_id="shared", actor="key_a")[1] == 1
    assert record(db, session_id="shared", actor="key_b")[1] == 1
    assert record(db, session_id="shared", actor="key_a")[1] == 2


# ── 会话累计 ─────────────────────────────────────────────────

def test_session_summary_sums_tokens(db):
    record(db, usage=make_usage(total=1000, hit=400, miss=600))
    record(db, usage=make_usage(total=2000, hit=100, miss=1900))
    s = db.qa_session_summary("sess-1", "key_a")
    assert s["turns"] == 2
    assert s["total_tokens"] == (1050 + 2050)
    assert s["cache_hit_tokens"] == 500
    assert s["cache_miss_tokens"] == 2500
    assert s["cache_hit_rate"] == pytest.approx(500 / 3000, abs=1e-5)


def test_session_summary_isolated_by_actor(db):
    """只按 session_id 查询等于"猜到 id 就能读到别人的用量"，必须同时按 actor 过滤"""
    record(db, session_id="shared", actor="key_a", usage=make_usage(total=9999))
    s_other = db.qa_session_summary("shared", "key_b")
    assert s_other["turns"] == 0
    assert s_other["total_tokens"] == 0
    assert db.qa_session_summary("shared", "key_a")["turns"] == 1


def test_session_turns_are_ordered_and_limited(db):
    for _ in range(5):
        record(db)
    rows = db.qa_session_turns("sess-1", "key_a", limit=3)
    assert [r["turn"] for r in rows] == [3, 4, 5]   # 取最近 3 轮，时间正序返回
    assert db.qa_session_summary("sess-1", "key_a")["turns"] == 5


# ── trace 下钻 ───────────────────────────────────────────────

def test_calls_written_and_drillable_by_trace(db):
    record(db, usage=make_usage(calls=3), trace_id="trace-xyz")
    calls = db.qa_trace_calls("trace-xyz", "key_a")
    assert [c["call_index"] for c in calls] == [1, 2, 3]
    assert [c["stage"] for c in calls] == ["stage1", "stage2", "stage3"]
    assert calls[0]["model"] == "deepseek-chat"


def test_trace_drilldown_isolated_by_actor(db):
    """qa_metrics_calls 自带 actor，下钻查询自身即可隔离"""
    record(db, actor="key_a", trace_id="trace-shared")
    assert len(db.qa_trace_calls("trace-shared", "key_a")) == 2
    assert db.qa_trace_calls("trace-shared", "key_b") == []


def test_unknown_trace_returns_empty(db):
    assert db.qa_trace_calls("nope", "key_a") == []


# ── 全局累计 ─────────────────────────────────────────────────

def test_global_summary_across_sessions(db):
    record(db, session_id="s1", actor="key_a", usage=make_usage(total=100, hit=10, miss=90))
    record(db, session_id="s2", actor="key_b", usage=make_usage(total=200, hit=20, miss=180))
    g = db.qa_metrics_global()
    assert g["turns"] == 2
    assert g["total_tokens"] == (150 + 250)
    assert g["cache_hit_rate"] == pytest.approx(30 / 300)


# ── 表结构 ───────────────────────────────────────────────────

def test_schema_created_on_existing_db(tmp_path):
    """旧库重启时应自动补表（CREATE TABLE IF NOT EXISTS）"""
    p = tmp_path / "legacy.db"
    Database(str(p))                      # 第一次建库
    again = Database(str(p))              # 模拟重启
    again.record_qa_metrics(
        actor="k", session_id="s", question="q", intent="factoid",
        qa_mode="pipeline", usage=make_usage(calls=1), trace_id="t",
    )
    assert again.qa_session_summary("s", "k")["turns"] == 1


def test_anon_session_still_recorded(db):
    """匿名单次问答（未传 session_id）仍落库供成本核算，turn 恒为 1"""
    assert record(db, session_id="anon-abc123")[1] == 1
