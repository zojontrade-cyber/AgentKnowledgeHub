"""
Web UI 专用接口 —— 文档列表、概览、问答会话与历史

这些接口服务于前端单页应用（api/static/index.html），不属于核心业务 API。
设计与业务接口保持一致：需要鉴权、使用结构化日志、参数化查询。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from api.security import require_api_key
from config import settings
from services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ui", tags=["Web UI"])


# 通过依赖注入由 main.py 设置
_vector_store: VectorStoreService | None = None


def bind(vector_store: VectorStoreService) -> None:
    """由 main.py 在启动时注入服务实例"""
    global _vector_store
    _vector_store = vector_store


def _vs() -> VectorStoreService:
    if _vector_store is None:
        raise HTTPException(status_code=503, detail="向量库未就绪")
    return _vector_store


# ── 概览 ─────────────────────────────────────────────────────

@router.get("/overview", summary="系统概览（供仪表盘）")
async def overview(_key: str = Depends(require_api_key)) -> dict[str, Any]:
    """返回向量库统计信息，供前端仪表盘展示"""
    vs = _vs()

    try:
        vs_stats = await vs.get_stats()
    except Exception:
        logger.warning("读取向量库统计失败", exc_info=True)
        vs_stats = {"backend": "unknown", "total_vectors": 0}

    # 上传目录里的文件数
    upload_dir = Path(settings.upload_dir)
    doc_count = len(list(upload_dir.glob("*"))) if upload_dir.exists() else 0

    return {
        "vector_store": vs_stats,
        "uploaded_files": doc_count,
        "config": {
            "chat_model": settings.openai_model,
            "embedding_model": settings.embedding_model,
            "chroma_mode": settings.chroma_mode,
            "retrieval_mode": settings.retrieval_mode,
            # 生成 Prompt 结构（stable / legacy），便于对照命中率
            "answer_prompt_mode": settings.answer_prompt_mode,
            "environment": settings.environment,
        },
    }


# ── 文档列表 ─────────────────────────────────────────────────

@router.get("/documents", summary="已入库文档列表")
async def list_documents(_key: str = Depends(require_api_key)) -> dict[str, Any]:
    """
    列出已入库文档。

    为什么改成读 documents 表（而不再扫目录）
    -----------------------------------------
    原实现遍历上传目录并返回 `p.name` —— 而落盘文件名是**服务端生成的
    UUID**（上传接口出于防路径穿越的考虑刻意不采用客户端文件名）。
    结果是 UI 上满屏 `156fc719bb654112a177c5a2b9cd2933.md`，用户无法辨认。

    现在改为读 `documents` 表：
      - 用 `original_name` 作为 `display_name` 展示（上传时记录的中文名）
      - 同时返回真实入库状态（COMMITTED/PENDING/...），
        比"目录里有文件"更准确 —— 目录里有 ≠ 已索引成功
      - `source_path` 保持为**唯一底层真实路径**，仅作内部标识，不用于展示

    兼容兜底：`original_name` 缺失时回退为文件名，避免 UI 出现空名称。
    """
    from services.database import get_db

    try:
        rows = get_db().list_documents(limit=500, acl_scopes=None)
    except Exception:
        logger.warning("读取文档表失败", exc_info=True)
        return {"total": 0, "documents": []}

    upload_dir = Path(settings.upload_dir)
    docs = []
    for r in rows:
        source_path = str(r.get("source_path") or "")
        fname = Path(source_path).name
        # 兜底：original_name 缺失时用文件名，避免空名称
        display_name = (r.get("original_name") or "").strip() or fname

        # 文件大小仍从磁盘取（表里没存）；文件可能已被移动，故容错
        size = 0
        modified = int(r.get("updated_at") or 0)
        try:
            p = upload_dir / fname
            if p.exists():
                st = p.stat()
                size = st.st_size
                modified = int(st.st_mtime)
        except Exception:
            pass

        docs.append({
            # 展示名（中文）—— UI 优先用它
            "display_name": display_name,
            # 兼容旧的 name 字段（前端可能仍在读）
            "name": display_name,
            # 内部真实路径标识（不用于展示）
            "source_path": source_path,
            "status": r.get("status") or "",
            "chunks_count": int(r.get("chunks_count") or 0),
            "acl_scope": r.get("acl_scope") or "",
            "size": size,
            "size_human": _human_size(size),
            "modified": modified,
        })

    return {"total": len(docs), "documents": docs[:200]}


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} TB"


# ── 问答会话用量 ─────────────────────────────────────────────

@router.get("/qa-session", summary="问答会话用量（供侧栏回填）")
async def qa_session(
    session_id: str = Query(..., min_length=1, max_length=64),
    limit: int = Query(10, ge=1, le=50),
    key: str = Depends(require_api_key),
) -> dict[str, Any]:
    """返回某个会话的累计用量与最近若干轮。

    为什么需要这个接口：指标持久化到 SQLite 后，刷新页面必须能把侧栏数字
    读回来，否则"持久化"只是名义上的。

    隔离：session_id 是客户端自有的随机串，单靠它查询等于"猜到 id 就能读到
    别人的用量"，因此查询**同时按 actor（当前 Key 指纹）过滤**。
    """
    from api.security import key_fingerprint
    from services.database import get_db

    actor = key_fingerprint(key)
    try:
        db = get_db()
        summary = db.qa_session_summary(session_id, actor)
        turns = db.qa_session_turns(session_id, actor, limit=limit)
    except Exception:
        logger.warning("读取问答会话用量失败", exc_info=True)
        raise HTTPException(status_code=503, detail="用量统计不可用")

    return {"summary": summary, "turns": turns}


# ── 对话历史（展示保留）──────────────────────────────────────

@router.get("/qa-history", summary="会话历史（刷新后恢复界面）")
async def qa_history(
    session_id: str = Query(..., min_length=1, max_length=64),
    limit: int = Query(50, ge=1, le=200),
    key: str = Depends(require_api_key),
) -> dict[str, Any]:
    """读回某会话的历史问答快照。

    存的是**当轮响应快照**（问题 / 答案 / 出处 / 推理步骤 / 用量），
    前端恢复时直接喂给与实时回答**同一个**渲染函数，因此不可能出现
    "恢复出来的卡片和当时看到的不一样"。

    隔离：`session_id` 是客户端自有的随机串，只按它查询等于
    "猜到 id 就能读到别人的问答内容与出处"，因此**同时按 actor（当前 Key 指纹）过滤**。

    注意：历史**只用于展示**，不会注入 prompt —— 问答链路仍是单轮无记忆。
    详见 docs/conversation-history.md
    """
    from api.security import key_fingerprint
    from services.database import get_db

    actor = key_fingerprint(key)
    try:
        turns, total = get_db().list_qa_messages(session_id, actor, limit=limit)
    except Exception:
        logger.warning("读取对话历史失败", exc_info=True)
        raise HTTPException(status_code=503, detail="对话历史不可用")

    return {
        "session_id": session_id,
        "total": total,
        "returned": len(turns),
        # 有截断时前端提示"仅显示最近 N 轮"
        "truncated": total > len(turns),
        "turns": turns,
    }


