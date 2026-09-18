"""
LangGraph 编排引擎 — 三条流水线

本次重构的核心变化
------------------
**入库流水线从「直接双写」改为「状态机 + 发件箱」**

    旧结构（无事务、无补偿、无状态）:
        extract ──> store_vectors ──> END
        问题：中途失败 -> 数据状态不可知，且无人知道卡在哪

    新结构（可追踪、可重放）:
        parse -> extract -> persist_vectors -> commit
                     ↑________ 失败可重放 ________↓
        每一步都更新 documents.status，失败可定位到具体阶段并单独重放

为什么保留分步而不是合并成一步：
  重新解析 + LLM 抽取是**昂贵操作**（每 chunk 一次 LLM 调用）。
  分步记录状态后，失败只需重放失败的那一步，
  不必重新解析（省费用，也缩短不一致窗口）。
"""

from __future__ import annotations

import logging
import operator
import time
import uuid
from enum import Enum
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph

from agents.doc_parser_agent import DocParserAgent, DocumentChunk
from agents.knowledge_extract_agent import ExtractionResult, KnowledgeExtractAgent
from agents.knowledge_update_agent import (
    ChangeType,
    DocumentChange,
    KnowledgeUpdateAgent,
    UpdateResult,
)
from agents.qa_agent import QAAgent, QAResult
from agents.react_qa_agent import ReactQAAgent
from config import settings
from services.database import DocStatus, get_db
from services.qa_payload import build_turn_payload
from services.usage_meter import usage_meter
from services.vector_store import VectorStoreService
from utils.logging_config import get_request_id

logger = logging.getLogger(__name__)


class WorkflowType(str, Enum):
    INGEST = "ingest"
    QA = "qa"
    UPDATE = "update"


# _deprecated/services/entity_canonicalizer.py。


# ── State Schemas ────────────────────────────────────────────

class IngestState(TypedDict, total=False):
    """文档入库流程状态"""
    doc_id: str
    file_paths: list[str]
    chunks: list[DocumentChunk]
    extractions: list[ExtractionResult]
    vectors_stored: Annotated[int, operator.add]
    entities_stored: Annotated[int, operator.add]
    relations_stored: Annotated[int, operator.add]
    error: str


class QAState(TypedDict, total=False):
    question: str
    result: QAResult | None
    actor: str
    acl_scopes: list[str]
    # ── 用量计量（由 services/usage_meter.py 采集、database 持久化）──
    session_id: str
    turn: int
    usage: dict
    session_stats: dict
    trace_id: str
    # 统计库写入失败时置 True：本轮数字仅本请求有效，前端不应把它当真值展示
    metrics_degraded: bool


class UpdateState(TypedDict, total=False):
    changes: list[DocumentChange]
    results: list[UpdateResult]


# ── Workflow Builder ─────────────────────────────────────────

def build_knowledge_graph_workflow(
    vector_store: VectorStoreService | None = None,
) -> dict[str, Any]:
    """
    构建编排流水线，返回 {"ingest": graph, "qa": graph, "update": graph}

    """
    doc_parser = DocParserAgent()
    extractor = KnowledgeExtractAgent()

    # QA Agent 按 qa_mode 选择实现。
    #
    # 两个实现的对外接口完全一致（`answer(question, acl_scopes) -> QAResult`），
    # 因此编排图结构无需任何改动 —— 只换实例。
    #   pipeline = 入口意图路由 + 静态检索计划（现状，默认）
    #   react    = LLM 逐步选择工具与停止时机
    if settings.qa_mode == "react":
        qa_agent: Any = ReactQAAgent(
            vector_store=vector_store,
            max_steps=settings.qa_max_steps,
        )
        logger.info(
            "QA 使用 ReAct 模式",
            extra={"extra_fields": {"max_steps": settings.qa_max_steps}},
        )
    else:
        qa_agent = QAAgent(vector_store=vector_store)

    update_agent = KnowledgeUpdateAgent(
        doc_parser=doc_parser,
        knowledge_extractor=extractor,
        vector_store=vector_store,
    )

    return {
        "ingest": _build_ingest_graph(doc_parser, extractor, vector_store),
        "qa": _build_qa_graph(qa_agent),
        "update": _build_update_graph(update_agent),
    }


# ── Ingest Pipeline ─────────────────────────────────────────

def _build_ingest_graph(
    doc_parser: DocParserAgent,
    extractor: KnowledgeExtractAgent,
    vector_store: VectorStoreService | None,
) -> StateGraph:
    """
    入库流水线：串行 + 状态机。

    拓扑：
        parse → extract → vectors → commit

    为什么要串行（放弃并行）：
      并行双写是「数据不一致」的根源——两个分支各自成功/失败无法协调。
      改为串行后，每一步的成败都能精确记录到 documents.status，
      失败时从断点重放而不是整篇重来。

    `extract` 节点保留 —— 抽取结果仍写入 documents 的计数字段，
    且是最昂贵的一步（每 chunk 一次 LLM），失败时应从断点重放。
    """

    async def parse_documents(state: dict) -> dict:
        doc_id = state.get("doc_id", "")
        file_paths = state.get("file_paths", [])
        try:
            chunks = await doc_parser.parse_batch(file_paths)
            if doc_id:
                get_db().set_status(
                    doc_id, DocStatus.PROCESSING, chunks_count=len(chunks)
                )
            return {"chunks": chunks}
        except Exception as e:
            if doc_id:
                get_db().set_status(doc_id, DocStatus.FAILED, error=str(e)[:500])
            raise

    async def extract_knowledge(state: dict) -> dict:
        """
        LLM 抽取实体与关系。

        这是最昂贵的一步（每 chunk 一次 LLM 调用），因此单独成节点：
        后续步骤失败时只需重放后续部分，不必重复这一阶段。
        """
        chunks = state.get("chunks", [])
        extractions = await extractor.extract(chunks)
        return {"extractions": extractions}

    async def persist_vectors(state: dict) -> dict:
        """写入向量库，成功后把状态推进到 VECTOR_DONE"""
        doc_id = state.get("doc_id", "")
        chunks = state.get("chunks", [])
        count = 0
        if vector_store and chunks:
            count = await vector_store.add_chunks(chunks)
        if doc_id:
            get_db().set_status(doc_id, DocStatus.VECTOR_DONE)
        return {"vectors_stored": count}

    async def commit(state: dict) -> dict:
        """全部成功，标记 COMMITTED 并登记文件哈希"""
        doc_id = state.get("doc_id", "")
        if not doc_id:
            return {}
        db = get_db()
        doc = db.get_document(doc_id)
        file_paths = state.get("file_paths", [])
        chunks = state.get("chunks", [])

        db.set_status(
            doc_id,
            DocStatus.COMMITTED,
            chunks_count=len(chunks),
            error="",
        )
        # 登记文件哈希（取代进程内存态 _file_hashes）
        if doc and doc.get("content_hash") and file_paths:
            db.set_file_hash(file_paths[0], doc["content_hash"])
        logger.info(
            "文档入库完成",
            extra={"extra_fields": {
                "doc_id": doc_id,
                "chunks": len(chunks),
                "vectors": state.get("vectors_stored", 0),
            }},
        )
        return {}

    graph = StateGraph(IngestState)
    graph.add_node("parse", parse_documents)
    graph.add_node("extract", extract_knowledge)
    graph.add_node("vectors", persist_vectors)
    graph.add_node("commit", commit)

    graph.set_entry_point("parse")
    graph.add_edge("parse", "extract")
    graph.add_edge("extract", "vectors")
    graph.add_edge("vectors", "commit")
    graph.add_edge("commit", END)

    return graph.compile()


# ── Replay（断点重放）────────────────────────────────────────

async def replay_document(
    doc_id: str,
    vector_store: VectorStoreService | None = None,
) -> dict[str, Any]:
    """
    从失败断点重放一篇文档 —— 一致性的恢复机制。

    依据 documents.status 决定从哪一步继续：
        VECTOR_DONE -> 向量已写好，只需补 commit
        其他/FAILED -> 整篇重做

    这是「可恢复」的具体实现：向量已经写好的文档不必重新
    解析和 LLM 抽取（那是最贵的部分）。

    注：`VECTOR_DONE` 直接补 commit。
    """
    db = get_db()
    doc = db.get_document(doc_id)
    if not doc:
        return {"ok": False, "error": "文档不存在"}

    status = doc["status"]
    source_path = doc["source_path"]

    if status == DocStatus.COMMITTED:
        return {"ok": True, "action": "already_committed"}

    parser = DocParserAgent()
    extractor = KnowledgeExtractAgent()

    try:
        if status == DocStatus.VECTOR_DONE:
            # 向量已就绪，无需重跑解析与 LLM 抽取，直接补 commit
            db.set_status(doc_id, DocStatus.COMMITTED)
            return {"ok": True, "action": "committed_from_vectors"}

        # 其他情况：整篇重做
        g = _build_ingest_graph(parser, extractor, vector_store)
        await g.ainvoke({
            "doc_id": doc_id,
            "file_paths": [source_path],
            "vectors_stored": 0,
            "entities_stored": 0,
            "relations_stored": 0,
        })
        return {"ok": True, "action": "full_replay"}

    except Exception as e:
        db.bump_retry(doc_id)
        retries = db.get_document(doc_id)
        if retries and int(retries.get("retry_count", 0)) >= 5:
            db.set_status(doc_id, DocStatus.FAILED, error=str(e)[:500])
        logger.error("重放失败", exc_info=True, extra={"extra_fields": {"doc_id": doc_id}})
        return {"ok": False, "error": str(e)[:300]}


# ── QA Pipeline ──────────────────────────────────────────────

def _build_qa_graph(qa_agent: QAAgent) -> StateGraph:

    async def process_question(state: dict) -> dict:
        question = state.get("question", "")
        actor = state.get("actor", "")
        session_id = state.get("session_id") or ""
        db = get_db()

        # trace_id 复用日志中间件已分配的 request_id，使用量与日志同源可对。
        # 脱离 HTTP 调用（脚本 / 单测）时为默认值 "-"，此时另生成一个。
        trace_id = get_request_id()
        if not trace_id or trace_id == "-":
            trace_id = uuid.uuid4().hex[:12]

        # 计量作用域：包住**含 TypeError 兼容分支**的整个调用。
        # record_* 由 services/usage_meter.py 在 LLM 调用点采集，
        # 此处只负责开域、计时、聚合。
        with usage_meter() as rec:
            started = time.perf_counter()
            try:
                result = await qa_agent.answer(question, acl_scopes=state.get("acl_scopes"))
            except TypeError:
                # 兼容未支持 acl_scopes 的调用签名
                result = await qa_agent.answer(question)
            latency_ms = (time.perf_counter() - started) * 1000
            usage = rec.summary()

        usage["latency_ms"] = round(latency_ms, 1)
        intent_value = result.intent.value if result else None

        # ── 持久化 + 审计：失败**不能**影响问答本身 ──────────
        turn = 1
        session_stats: dict[str, Any] = {}
        degraded = False
        try:
            _, turn = db.record_qa_metrics(
                actor=actor, session_id=session_id, question=question,
                intent=intent_value or "", qa_mode=settings.qa_mode,
                usage=usage, latency_ms=latency_ms, trace_id=trace_id,
                # 记录用的 Prompt 结构，便于把命中率按结构对比
                answer_prompt_mode=settings.answer_prompt_mode,
            )
            session_stats = db.qa_session_summary(session_id, actor)

            # 对话历史：仅供**展示**恢复，不注入 prompt（问答仍是单轮无记忆）。
            #
            # 匿名请求不写：服务端生成的 `anon-*` 永不交给客户端
            #（见 /api/qa/ask 的注释），写了也没人能读回来，只是白占空间。
            #
            # 与 metrics 非同一事务：metrics 成功而这里失败时，历史缺一轮，
            # 但 turn 不会错位（turn 的权威在 qa_metrics）。历史是展示数据，
            # 有意接受这种非原子性，以免改动已验证过的 record_qa_metrics。
            if result and session_id and not session_id.startswith("anon-"):
                db.record_qa_message(
                    actor=actor, session_id=session_id, turn=turn,
                    trace_id=trace_id,
                    payload=build_turn_payload(
                        result, usage=usage, turn=turn,
                        trace_id=trace_id, metrics_degraded=False,
                    ),
                )
        except Exception:
            # 统计库故障时的 fallback：不再提"退化为前端传入值"
            #（请求里根本没有该字段），而是就地合成，保证响应永远合法。
            degraded = True
            logger.warning("问答用量写入失败，返回降级统计", exc_info=True)
            session_stats = {
                "session_id": session_id,
                "turns": 1,
                "total_tokens": usage["total_tokens"],
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "cache_hit_tokens": usage["cache_hit_tokens"],
                "cache_miss_tokens": usage["cache_miss_tokens"],
                "cache_hit_rate": usage["cache_hit_rate"],
            }

        try:
            db.audit(
                "qa.ask", actor=actor, resource=question[:120],
                detail={
                    "intent": intent_value,
                    "session_id": session_id,
                    "turn": turn,
                    "total_tokens": usage["total_tokens"],
                },
                result="ok",
            )
        except Exception:
            pass

        return {
            "result": result,
            "usage": usage,
            "turn": turn,
            "session_stats": session_stats,
            "metrics_degraded": degraded,
            "trace_id": trace_id,
        }

    graph = StateGraph(QAState)
    graph.add_node("answer", process_question)
    graph.set_entry_point("answer")
    graph.add_edge("answer", END)

    return graph.compile()


# ── Update Pipeline ──────────────────────────────────────────

def _build_update_graph(update_agent: KnowledgeUpdateAgent) -> StateGraph:
    """文档更新流水线：带**有界重试 + 指数退避**（原实现只重试一次且不复查）

    注：这是**整篇替换**（按 doc_id 删旧 + 重新解析写入）。
    且当前无自动触发入口，只有 POST /api/admin/update 手动调用。
    """

    MAX_ATTEMPTS = 3
    BASE_DELAY = 1.0
    MAX_DELAY = 8.0

    async def process_updates(state: dict) -> dict:
        changes = state.get("changes", [])
        results = await update_agent.process_batch(changes)
        return {"results": results, "_attempt": 0}

    def should_retry(state: dict) -> str:
        results = state.get("results", [])
        attempt = state.get("_attempt", 0)
        failed = [r for r in results if not r.success]
        if failed and attempt < MAX_ATTEMPTS:
            return "retry"
        return "done"

    async def retry_failed(state: dict) -> dict:
        import asyncio

        results = state.get("results", [])
        attempt = state.get("_attempt", 0) + 1
        failed_changes = [r.change for r in results if not r.success]

        # 指数退避，避免失败瞬间重试再次撞上同一问题
        delay = min(BASE_DELAY * (2 ** (attempt - 1)), MAX_DELAY)
        await asyncio.sleep(delay)

        retried = await update_agent.process_batch(failed_changes)
        logger.info(
            "更新重试",
            extra={"extra_fields": {
                "attempt": attempt, "count": len(failed_changes), "delay": delay,
            }},
        )
        return {"results": [r for r in results if r.success] + retried, "_attempt": attempt}

    graph = StateGraph(dict)
    graph.add_node("process", process_updates)
    graph.add_node("retry", retry_failed)
    graph.set_entry_point("process")
    graph.add_conditional_edges(
        "process", should_retry, {"retry": "retry", "done": END}
    )
    # 重试后**回到判断**，形成有界循环（原实现 retry->END 直接结束，
    # 第二次失败被静默接受）
    graph.add_conditional_edges(
        "retry", should_retry, {"retry": "retry", "done": END}
    )
    return graph.compile()
