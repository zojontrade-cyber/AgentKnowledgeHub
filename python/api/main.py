"""
FastAPI 入口 — 企业知识管理系统 REST API

提供四组接口:
  1. /api/ingest   — 文档上传（异步）& 入库
  2. /api/qa       — 智能问答（带数据级 ACL）
  3. /api/admin    — 管理（统计、更新触发）
  4. /api/jobs     — 后台任务查询

本次重构要点
------------
1. **异步入库**：上传立即返回 task_id（202），后台 worker 处理。
   原实现是同步阻塞 —— 大文档必然 HTTP 超时。
2. **状态机 + Outbox**：documents 表记录 PENDING→…→COMMITTED，
   失败可从断点重放，不会留下「写了一半」的不可知状态。
3. **状态可恢复**：向量已写入但未提交的文档可直接补 commit。
4. **数据级 ACL**：API Key 携带可见范围，检索时按文档 acl_scope 过滤。
5. **审计日志**：问答、上传等敏感操作落库，可追溯。
6. 启动 fail-fast、真实健康检查、鉴权限流、安全上传（沿用前次修复）
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agents.knowledge_update_agent import ChangeType, DocumentChange
from api.security import (
    enforce_rate_limit,
    key_fingerprint,
    require_admin_key,
    require_api_key,
    require_api_key_with_scope,
)
from api.upload_handler import resolve_managed_path, save_upload_safely
from config import settings
from orchestrator.graph import build_knowledge_graph_workflow
from services.database import DocStatus, get_db, init_db
from services.ingest_worker import IngestWorker
from services.qa_payload import build_turn_payload
from services.reranker import reranker
from services.vector_store import VectorStoreService
from utils.logging_config import configure_logging, set_request_id

configure_logging()
logger = logging.getLogger(__name__)

vector_store = VectorStoreService()
workflows: dict[str, Any] = {}
worker: IngestWorker | None = None


# ── 生命周期 ─────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    启动校验 + 依赖初始化。

    与原实现的关键差别：初始化失败**不再被吞掉**。要么启动成功且依赖可用，
    要么启动失败并给出明确原因 —— 避免"健康绿灯 + 功能静默残缺"。
    """
    global worker
    os.makedirs(settings.upload_dir, exist_ok=True)

    # SQLite 持久化层（零外部依赖）
    init_db(settings.sqlite_path)
    logger.info(
        "持久化层已就绪",
        extra={"extra_fields": {"path": settings.sqlite_path}},
    )

    logger.info(
        "服务启动中",
        extra={"extra_fields": {
            "environment": settings.environment,
            "vector_backend": settings.vector_store_type,
            "auth_enabled": settings.auth_enabled,
            "worker_enabled": settings.worker_enabled,
        }},
    )

    failures: list[str] = []

    try:
        await vector_store.init()
        logger.info("向量库初始化成功", extra={"extra_fields": {"backend": settings.vector_store_type}})
    except Exception as e:
        failures.append(f"向量库不可用: {e}")
        logger.error("向量库初始化失败", exc_info=True)

    if failures:
        msg = "依赖初始化失败:\n  - " + "\n  - ".join(failures)
        if settings.is_prod:
            logger.critical(msg + "\n生产环境要求依赖全部可用，拒绝启动。")
            raise RuntimeError(msg)
        # dev 模式：明确是「降级启动」，避免日志自相矛盾
        logger.warning(
            msg + "\nDEV 模式：降级启动，上述依赖相关功能不可用。",
            extra={"extra_fields": {"degraded": True}},
        )

    if not settings.openai_api_key:
        logger.warning(
            "未配置 OPENAI_API_KEY：服务可启动，但文档解析与问答在调用 LLM 时会报错。"
            "请在 python/.env 中填写后重启。"
        )

    workflows.update(build_knowledge_graph_workflow(vector_store=vector_store))
    logger.info("编排流水线就绪", extra={"extra_fields": {"pipelines": sorted(workflows)}})

    # 启动后台 worker：消费 outbox，处理异步入库任务
    if settings.worker_enabled:
        worker = IngestWorker(
            workflows=workflows,
            vector_store=vector_store,
            poll_interval=settings.worker_poll_interval,
            max_retries=settings.max_retries,
        )
        worker.start()

    # 启动时报告遗留任务，避免"静默积压"
    try:
        db = get_db()
        pending = db.outbox_stats()
        stuck = [d for d in db.list_documents(limit=100)
                 if d["status"] not in (DocStatus.COMMITTED, DocStatus.FAILED)]
        if stuck:
            logger.warning(
                "存在未完成的文档，worker 将继续处理",
                extra={"extra_fields": {
                    "count": len(stuck),
                    "outbox": pending,
                }},
            )
    except Exception:
        logger.warning("检查遗留任务失败", exc_info=True)

    # ── 对话历史 TTL 清理 ────────────────────────────────
    # retention_days <= 0 表示永久保留 —— 此时**不启动**任务，
    # 而不是"按 0 天清理"（那会把历史全删掉）。
    purge_task: asyncio.Task | None = None
    if settings.qa_history_retention_days > 0:
        try:
            removed = get_db().purge_qa_messages(settings.qa_history_retention_days)
            if removed:
                logger.info(
                    "启动清理过期对话历史",
                    extra={"extra_fields": {
                        "removed": removed,
                        "retention_days": settings.qa_history_retention_days,
                    }},
                )
        except Exception:
            logger.warning("启动清理对话历史失败", exc_info=True)

        async def _purge_loop() -> None:
            interval = max(1, settings.qa_history_purge_interval_hours) * 3600
            while True:
                await asyncio.sleep(interval)
                try:
                    n = get_db().purge_qa_messages(settings.qa_history_retention_days)
                    if n:
                        logger.info(
                            "周期清理过期对话历史",
                            extra={"extra_fields": {"removed": n}},
                        )
                except Exception:
                    logger.warning("周期清理对话历史失败", exc_info=True)

        purge_task = asyncio.create_task(_purge_loop())
        logger.info(
            "对话历史清理任务已启动",
            extra={"extra_fields": {
                "retention_days": settings.qa_history_retention_days,
                "interval_hours": settings.qa_history_purge_interval_hours,
            }},
        )

    yield

    # 优雅关闭：先停清理任务，再停 worker
    if purge_task:
        purge_task.cancel()
    if worker:
        await worker.stop()
    logger.info("服务已关闭")


app = FastAPI(
    title="AgentKnowledgeHub — 多Agent企业知识管理系统",
    description="企业知识库智能问答 API（BM25 召回 + cross-encoder 精排）",
    version="1.0.0",
    lifespan=lifespan,
)

# Web UI 路由（文档列表 / 概览）
from api import ui_routes  # noqa: E402

ui_routes.bind(vector_store)
app.include_router(ui_routes.router)

# 静态页面：访问 http://127.0.0.1:8080/ 直达 Web UI
_STATIC_DIR = Path(__file__).resolve().parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        """根路径返回单页应用"""
        return FileResponse(str(_STATIC_DIR / "index.html"))


# ── 请求日志中间件 ───────────────────────────────────────────

@app.middleware("http")
async def request_logging(request: Request, call_next):
    """为每个请求分配 request_id，记录耗时与状态"""
    rid = set_request_id(request.headers.get("X-Request-ID"))
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        elapsed = (time.perf_counter() - start) * 1000
        logger.exception(
            "请求处理异常",
            extra={"extra_fields": {
                "path": request.url.path, "method": request.method,
                "latency_ms": round(elapsed, 1),
            }},
        )
        raise
    elapsed = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = rid
    logger.info(
        "请求完成",
        extra={"extra_fields": {
            "path": request.url.path, "method": request.method,
            "status": response.status_code, "latency_ms": round(elapsed, 1),
        }},
    )
    return response


# ── 请求/响应模型 ────────────────────────────────────────────

# 客户端自有的会话 id 形状约束（见 QuestionRequest.session_id）
_SESSION_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")

class QuestionRequest(BaseModel):
    question: str
    # 会话标识由**客户端自有**（前端存在 sessionStorage）。
    # 不传 = 匿名单次问答：服务端为本次请求生成临时 id，turn=1，不形成累计会话，
    # 且**不会把这个 id 交回给客户端**，杜绝"服务端生成的 id 被复用而变成真 session"。
    #
    # 注意：这里**没有** turn 字段 —— 轮次由服务端按 DB 记录递增取号。
    session_id: str | None = None


class QaCallUsage(BaseModel):
    """单次 LLM 调用的用量（归一化后，前端只认这一种形状）"""

    call_index: int = 0
    stage: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    cache_supported: bool = False
    # 缓存"写入"侧：本 Provider 不提供 -> None（**不是 0**）
    cache_creation_tokens: int | None = None
    # prompt 区段拆分（估算 token，见 services/usage_meter.py）
    stable_prefix_tokens: int = 0
    context_tokens: int = 0
    query_tokens: int = 0


class QaUsage(BaseModel):
    """本轮问答的用量汇总（含缓存拆解）"""

    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # 缓存口径只覆盖"可测"的调用（提供可识别 cache 字段的那些）
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    cache_measured_calls: int = 0
    cache_hit_calls: int = 0
    cache_supported: bool = False
    cache_creation_tokens: int | None = None
    # ① Prompt Cache Hit Rate —— **可测 Prompt Token 的命中占比**，
    #    不是"多少次调用命中了缓存"（后者是 cache_hit_calls）
    cache_hit_rate: float = 0.0
    # ② 稳定前缀复用率：system 段（逐字节相同的那部分）有多少被复用
    stable_prefix_tokens: int = 0
    stable_prefix_reuse_rate: float | None = None
    # ③ 检索上下文复用率：RAG 上下文有多少被复用
    context_tokens: int = 0
    rag_context_reuse_rate: float | None = None
    query_tokens: int = 0
    latency_ms: float = 0.0
    per_call: list[QaCallUsage] = Field(default_factory=list)


class QaSessionStats(BaseModel):
    """会话累计（按 session_id + actor 隔离）"""

    session_id: str = ""
    turns: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    cache_hit_rate: float = 0.0


class QuestionResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    intent: str
    sources: list[dict[str, Any]]
    reasoning_steps: list[str]
    # ── 计量字段 ────────────────────────────────────────
    trace_id: str = ""
    # 未传 session_id 时为 None（服务端不交出匿名 id）
    session_id: str | None = None
    turn: int = 1
    usage: QaUsage = Field(default_factory=QaUsage)
    session: QaSessionStats = Field(default_factory=QaSessionStats)
    # 统计库写入失败时为 True —— 前端应显示"统计不可用"，
    # 而不是把 fallback 的 turn=1 当真值展示
    metrics_degraded: bool = False


class IngestResponse(BaseModel):
    file_name: str
    stored_as: str
    chunks_count: int
    entities_count: int
    relations_count: int
    status: str


class StatsResponse(BaseModel):
    vector_store: dict[str, Any]


class UpdateRequest(BaseModel):
    file_path: str
    change_type: str = "modified"


class UpdateResponse(BaseModel):
    file_path: str
    vectors_added: int
    vectors_deleted: int
    entities_added: int
    relations_added: int
    success: bool
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    service: str
    dependencies: dict[str, str]


# ── 异步任务模型 ─────────────────────────────────────────────

class JobAcceptedResponse(BaseModel):
    """上传后的即时响应（202）"""
    doc_id: str
    file_name: str
    status: str
    duplicate: bool = False
    message: str = ""


class JobStatusResponse(BaseModel):
    """任务进度"""
    doc_id: str
    status: str
    progress: int = 0
    file_name: str = ""
    chunks_count: int = 0
    entities_count: int = 0
    relations_count: int = 0
    retry_count: int = 0
    error: str = ""
    acl_scope: str = "public"
    created_at: float = 0
    updated_at: float = 0


def _key_fingerprint(key: str) -> str:
    """兼容旧调用点：实现已移到 api/security.py（用量隔离也要用）"""
    return key_fingerprint(key)


# ── 文档入库（异步）──────────────────────────────────────────

@app.post(
    "/api/ingest/upload",
    response_model=JobAcceptedResponse,
    status_code=202,
    tags=["文档入库"],
)
async def upload_document(
    file: UploadFile = File(...),
    key: str = Depends(require_api_key),
    _rl: None = Depends(enforce_rate_limit),
):
    """
    上传文档（异步）。

    立即返回 202 + doc_id，**不再同步等待**解析与 LLM 抽取。
    原因：一份 50 页 PDF 的完整链路需要数分钟，同步必然 HTTP 超时。

    流程：安全落盘 -> 登记 documents 表（PENDING）-> 后台 worker 处理
    进度查询：GET /api/jobs/{doc_id}
    """
    save_path, original_name, content_hash = await save_upload_safely(file)

    db = get_db()
    doc_id = db.create_document(
        source_path=save_path,
        original_name=original_name,
        content_hash=content_hash,
        acl_scope=settings.default_acl_scope,
        owner=_key_fingerprint(key),
    )

    doc = db.get_document(doc_id)
    already = doc and doc["status"] == DocStatus.COMMITTED

    db.audit(
        "doc.upload", actor=_key_fingerprint(key), resource=original_name,
        detail={"doc_id": doc_id, "hash": content_hash[:16], "duplicate": already},
    )

    logger.info(
        "文档已接收，等待后台处理",
        extra={"extra_fields": {
            "doc_id": doc_id, "file": original_name,
            "bytes": os.path.getsize(save_path), "duplicate": already,
        }},
    )

    return JobAcceptedResponse(
        doc_id=doc_id,
        file_name=original_name,
        status=doc["status"] if doc else DocStatus.PENDING,
        duplicate=bool(already),
        message=(
            "内容已存在，无需重复入库" if already
            else "已接收，正在后台处理，请通过 GET /api/jobs/{doc_id} 查询进度"
        ),
    )


@app.post(
    "/api/ingest/batch",
    response_model=list[JobAcceptedResponse],
    status_code=202,
    tags=["文档入库"],
)
async def upload_batch(
    files: list[UploadFile] = File(...),
    key: str = Depends(require_api_key),
    _rl: None = Depends(enforce_rate_limit),
):
    """批量上传（全部异步）"""
    results = []
    for file in files:
        results.append(await upload_document(file, key, _rl))
    return results


# ── 任务查询 ─────────────────────────────────────────────────

@app.get("/api/jobs/{doc_id}", response_model=JobStatusResponse, tags=["任务查询"])
async def get_job(doc_id: str, _key: str = Depends(require_api_key)):
    """查询文档处理进度"""
    doc = get_db().get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="任务不存在")

    # 计算进度百分比，便于前端展示
    progress = {
        DocStatus.PENDING: 5,
        DocStatus.PROCESSING: 35,
        DocStatus.VECTOR_DONE: 65,
        DocStatus.GRAPH_DONE: 90,
        DocStatus.COMMITTED: 100,
        DocStatus.FAILED: 100,
    }.get(doc["status"], 0)

    return JobStatusResponse(
        doc_id=doc_id,
        status=doc["status"],
        progress=progress,
        file_name=doc.get("original_name") or "",
        chunks_count=doc.get("chunks_count") or 0,
        entities_count=doc.get("entities_count") or 0,
        relations_count=doc.get("relations_count") or 0,
        retry_count=doc.get("retry_count") or 0,
        error=doc.get("error") or "",
        acl_scope=doc.get("acl_scope") or "public",
        created_at=doc.get("created_at") or 0,
        updated_at=doc.get("updated_at") or 0,
    )


@app.get("/api/jobs", response_model=list[JobStatusResponse], tags=["任务查询"])
async def list_jobs(_key: str = Depends(require_admin_key), limit: int = 50):
    """列出全部任务（管理员）"""
    out = []
    for doc in get_db().list_documents(limit=limit):
        out.append(JobStatusResponse(
            doc_id=doc["id"],
            status=doc["status"],
            progress=100 if doc["status"] in (
                DocStatus.COMMITTED, DocStatus.FAILED) else 50,
            file_name=doc.get("original_name") or "",
            chunks_count=doc.get("chunks_count") or 0,
            entities_count=doc.get("entities_count") or 0,
            relations_count=doc.get("relations_count") or 0,
            retry_count=doc.get("retry_count") or 0,
            error=doc.get("error") or "",
            acl_scope=doc.get("acl_scope") or "public",
            created_at=doc.get("created_at") or 0,
            updated_at=doc.get("updated_at") or 0,
        ))
    return out


@app.post("/api/jobs/{doc_id}/retry", tags=["任务查询"])
async def retry_job(doc_id: str, _key: str = Depends(require_admin_key)):
    """
    手动重放失败的文档（从断点继续，不必重新解析）。

    这是「一致性可恢复」的对外入口：向量已写好的文档只补提交，
    不会重复消耗 LLM 额度。
    """
    from orchestrator.graph import replay_document

    r = await replay_document(doc_id, vector_store)
    get_db().audit(
        "doc.retry", actor=_key_fingerprint(_key), resource=doc_id,
        detail=r, result="ok" if r.get("ok") else "error",
    )
    if not r.get("ok"):
        raise HTTPException(status_code=500, detail=str(r.get("error", "重放失败")))
    return r


# ── 智能问答 ─────────────────────────────────────────────────

@app.post("/api/qa/ask", response_model=QuestionResponse, tags=["智能问答"])
async def ask_question(
    req: QuestionRequest,
    creds: tuple[str, list[str]] = Depends(require_api_key_with_scope),
    _rl: None = Depends(enforce_rate_limit),
):
    """
    智能问答 — 意图驱动的混合检索

    数据级 ACL：根据调用方 Key 的 scope 过滤检索结果，
    保证不同角色的用户只能看到自己有权访问的知识。
    """
    key, scopes = creds
    qa_wf = workflows.get("qa")
    if not qa_wf:
        raise HTTPException(status_code=503, detail="问答流水线未就绪")

    # "*" 表示不限制（管理员）
    acl = None if "*" in scopes else scopes

    # session_id 是客户端自有的随机串，会进入 SQL 之外的索引与聚合，
    # 因此做白名单校验（参数化查询本身已防注入，这里是防脏数据污染统计）。
    session_id = (req.session_id or "").strip()
    client_owned = bool(_SESSION_ID_RE.fullmatch(session_id))
    if client_owned:
        session_id_used = session_id
    else:
        # 未提供（或非法）= 匿名单次问答：仅为本次请求生成 id，
        # turn 恒为 1，不形成累计会话。响应里 session_id 返回 None。
        session_id_used = f"anon-{uuid.uuid4().hex[:12]}"

    result = await qa_wf.ainvoke({
        "question": req.question,
        "actor": _key_fingerprint(key),
        "acl_scopes": acl,
        "session_id": session_id_used,
    })
    qa_result = result.get("result")
    if not qa_result:
        raise HTTPException(status_code=500, detail="问答处理失败")

    usage = result.get("usage") or {}
    session_stats = result.get("session_stats") or {}
    if not client_owned:
        # 匿名单次问答：**任何地方都不交出服务端生成的 id**。
        # session_stats 里也带着它，若原样返回，客户端就能读到并复用，
        # 于是匿名单次会悄悄变成真会话 —— 这正是要避免的。
        session_stats = {**session_stats, "session_id": ""}

    # 用与"落库历史"完全相同的那一份序列化来构造响应
    #（services/qa_payload.py）。两边各写一遍的话，一旦漂移就会出现
    # "实时答得对、刷新后恢复出来的卡片不一样"这种极难排查的问题。
    payload = build_turn_payload(
        qa_result,
        usage=usage,
        turn=int(result.get("turn") or 1),
        trace_id=result.get("trace_id") or "",
        metrics_degraded=bool(result.get("metrics_degraded")),
    )

    return QuestionResponse(
        question=payload["question"],
        answer=payload["answer"],
        confidence=payload["confidence"],
        intent=payload["intent"],
        # 引用契约见 services/qa_payload.serialize_sources：
        # **展示命中的章节**，而不是展开后的父文档。
        #
        # 早期实现直接序列化 c.content，而 _expand_small_to_big 当时把
        # content 替换成了父文档全文，于是每个引用的开头永远是
        # "## 1. 文档说明" —— 答案可能对，但用户无法核对出处
        # （实测引用准确率仅 44.4%）。
        sources=payload["sources"],
        reasoning_steps=payload["reasoning_steps"],
        trace_id=payload["trace_id"],
        # 只有客户端自己声明的会话才回传；匿名单次问答不回传 id
        session_id=session_id if client_owned else None,
        turn=payload["turn"],
        usage=QaUsage(**usage),
        session=QaSessionStats(**session_stats),
        metrics_degraded=payload["metrics_degraded"],
    )


# ── 管理接口（需管理员 Key）──────────────────────────────────

@app.get("/api/admin/stats", response_model=StatsResponse, tags=["系统管理"])
async def get_stats(_key: str = Depends(require_admin_key)):
    """获取系统统计信息"""
    vs_stats = await vector_store.get_stats()
    return StatsResponse(vector_store=vs_stats)


@app.post("/api/admin/update", response_model=UpdateResponse, tags=["系统管理"])
async def trigger_update(req: UpdateRequest, _key: str = Depends(require_admin_key)):
    """
    手动触发知识更新。

    安全约束（原实现缺失）：
      - 需要管理员 Key（否则这是第二条未授权删库路径）
      - file_path 必须落在受管 uploads 目录内
    """
    managed = resolve_managed_path(req.file_path)

    update_wf = workflows.get("update")
    if not update_wf:
        raise HTTPException(status_code=503, detail="更新流水线未就绪")

    try:
        change_type = ChangeType(req.change_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"非法 change_type: {req.change_type}；"
                   f"可选 {[c.value for c in ChangeType]}",
        ) from None

    change = DocumentChange(file_path=str(managed), change_type=change_type)
    result = await update_wf.ainvoke({"changes": [change]})
    results = result.get("results", [])
    if not results:
        raise HTTPException(status_code=500, detail="更新处理失败")

    r = results[0]
    logger.info(
        "手动更新完成",
        extra={"extra_fields": {
            "file": str(managed), "change_type": change_type.value,
            "success": r.success,
        }},
    )
    return UpdateResponse(
        file_path=r.change.file_path,
        vectors_added=r.vectors_added,
        vectors_deleted=r.vectors_deleted,
        entities_added=r.entities_added,
        relations_added=r.relations_added,
        success=r.success,
        processing_time_ms=r.processing_time_ms,
    )


# ── 健康检查（真实探测）──────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse, tags=["系统管理"])
async def health():
    """
    真实探测依赖 —— 不再是硬编码 {"status": "ok"}。

    原实现只返回常量，导致依赖挂掉时编排系统永远认为服务健康、
    永不重启坏实例。现在依赖不可用会如实上报。

    依赖清单
    ----------------------
      vector_store   必需 —— QA 主路径（BM25）与入库都依赖它
      reranker       可选 —— 失败时回退 BM25 原序（质量降级而非不可用）

    """
    deps: dict[str, str] = {}

    try:
        await vector_store.get_stats()
        deps["vector_store"] = "ok"
    except Exception as e:
        deps["vector_store"] = f"error: {type(e).__name__}"

    # Reranker 状态可见化。
    #
    # 为什么必须暴露：reranker 失败时**主动回退**到 BM25 原序（这是
    # 正确的可用性设计），但排序质量会静默退回基线（Chunk@1
    # 86.8% -> 76.3%）。若不暴露，运维只看到 status=ok，无法察觉
    # 质量已降级。
    if not reranker.enabled:
        deps["reranker"] = "disabled"
    elif reranker.fail_streak >= 3:
        deps["reranker"] = f"degraded: {reranker.fail_streak} consecutive failures"
    else:
        deps["reranker"] = "ok"

    # 只有必需依赖决定整体健康：vector_store 恒必需。
    required = [deps["vector_store"]]

    healthy = all(v == "ok" for v in required)
    payload = HealthResponse(
        status="ok" if healthy else "degraded",
        service="AgentKnowledgeHub",
        dependencies=deps,
    )
    if not healthy:
        return JSONResponse(status_code=503, content=payload.model_dump())
    return payload


# ── 启动前端口检查 ───────────────────────────────────────────

def _port_owner_pid(port: int) -> str:
    """尽力查出占用端口的进程 PID（Windows netstat）。查不到返回空串。"""
    import subprocess

    try:
        out = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except Exception:
        return ""
    for line in out.splitlines():
        parts = line.split()
        # 形如: TCP  127.0.0.1:8080  0.0.0.0:0  LISTENING  25104
        if len(parts) >= 5 and parts[3].upper() == "LISTENING":
            if parts[1].endswith(f":{port}"):
                return parts[4]
    return ""


def _probe_own_service(host: str, port: int) -> bool:
    """端口上是否**已经是我们自己的服务**（看 /api/health 的 service 字段）。

    用空 ProxyHandler 建 opener：本机注册表配了系统代理，
    走默认 opener 时 127.0.0.1 的请求也可能被代理而超时。
    """
    import json as _json
    import urllib.request

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"http://{host}:{port}/api/health", timeout=3) as resp:
            data = _json.loads(resp.read().decode("utf-8", "replace"))
        return data.get("service") == "AgentKnowledgeHub"
    except Exception:
        return False


def _preflight_port(host: str, port: int) -> None:
    """启动前检查端口占用，避免"初始化到一半才失败"。

    为什么必须有这一步
    ------------------
    uvicorn 的顺序是：**先跑完 lifespan（打开 Chroma、启动后台 worker 与
    对话历史清理任务），再 bind 端口**。端口被占时那些资源已经初始化过了，
    日志里会出现"后台 worker 已启动 → 服务已关闭"再抛
    `[Errno 10048]`，看起来像"启动失败"却看不出原因，
    而且**两个进程会短暂同时打开同一个 Chroma 目录**。

    在这里提前拦下：给出可操作的原因，并且不去初始化任何东西。
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        if s.connect_ex((host, port)) != 0:
            return  # 端口空闲

    if _probe_own_service(host, port):
        # 目标状态已达成 —— 这不算失败，退出码给 0
        print(
            f"\n端口 {port} 上已有本服务在运行：http://{host}:{port}\n"
            f"无需重复启动。若要重启：先运行 stop-api.bat，再启动。\n"
        )
        raise SystemExit(0)

    pid = _port_owner_pid(port)
    who = f"（PID {pid}）" if pid else ""
    print(
        f"\n端口 {port} 已被其它程序占用{who}，服务无法启动。\n\n"
        f"  查看占用：netstat -ano | findstr :{port}\n"
        f"  结束进程：taskkill /PID <pid> /F"
        f"   （若为提权进程，需以管理员身份执行）\n"
        f"  换端口  ：set API_PORT=8090 && python -m api.main\n"
        f"  停已有实例：stop-api.bat\n"
    )
    raise SystemExit(1)


if __name__ == "__main__":
    import uvicorn

    _preflight_port(settings.api_host, settings.api_port)

    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
    )
