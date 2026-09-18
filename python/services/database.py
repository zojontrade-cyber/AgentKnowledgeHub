"""
持久化层 —— SQLite 实现文档状态机与 Outbox

解决什么问题
------------
原实现在编排层对向量库与派生索引**直接双写**：

    extract ──┬──> store_vectors ──> END
              └──> store_graph   ──> END

两个分支各走各的，没有事务、没有补偿、没有状态记录。于是：
  - vector 成功 + graph 失败 -> 数据永久不一致，且**无人知道**
  - 失败后无重放机制，只能重新上传（重复消耗 LLM 额度）
  - _file_hashes / _version_counter 是进程内存态，重启失忆、多副本损坏

本模块引入「单一事实源 + 发件箱」模式（SQLite 版，零外部依赖）：

    documents（唯一事实源，记录状态机）
        ↓ 状态流转
    outbox（待处理的派生任务，可重放）
        ↓ 由 worker 消费
    vector_store（可重建的派生索引）

状态机
------
    PENDING -> PROCESSING -> VECTOR_DONE -> COMMITTED
                          ↖______ 失败可重试 ______↙
    FAILED（超过重试上限）

为什么要状态机而不是简单的 success/fail：
  入库是分阶段的，任何一步都可能失败。分开记录才能知道卡在哪一步，
  从而只重放缺失的那一步，而不是整篇重新解析
  （省 LLM 费用，也避免不一致窗口）。

注：状态机里另有 GRAPH_DONE 一环，已不再被写入；
常量保留仅为兼容旧库中可能存在的历史记录。

迁移到 Postgres
---------------
本模块把 SQL 集中在一处，接口保持通用（execute/fetchone/fetchall）。
迁移时只需替换 _connect 与少量方言（AUTOINCREMENT -> SERIAL、
? -> %s），业务代码不用动。
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


# ── 状态常量 ─────────────────────────────────────────────────

class DocStatus:
    PENDING = "PENDING"            # 已落盘，等待处理
    PROCESSING = "PROCESSING"      # 正在解析
    VECTOR_DONE = "VECTOR_DONE"    # 向量已写入
    # 已废弃：保留以免旧库中可能存在的历史记录无法识别
    GRAPH_DONE = "GRAPH_DONE"
    COMMITTED = "COMMITTED"        # 全部完成
    FAILED = "FAILED"              # 超过重试上限


# 允许的重试流转：从某状态失败后，下次应从哪里继续
RESUME_FROM: dict[str, str] = {
    DocStatus.VECTOR_DONE: DocStatus.VECTOR_DONE,   # 向量已写入，直接推进到 COMMITTED
    DocStatus.GRAPH_DONE: DocStatus.GRAPH_DONE,     # 已废弃状态
}


class Database:
    """SQLite 封装：线程安全、支持事务、SQL 方言集中管理"""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # SQLite 默认禁止跨线程使用连接；服务端是多线程的，
        # 因此用 check_same_thread=False + 自己的锁保证安全。
        self._local = threading.local()
        self._write_lock = threading.RLock()
        self.init_schema()

    # ── 连接管理 ─────────────────────────────────────────

    @property
    def conn(self) -> sqlite3.Connection:
        """每线程一个连接（SQLite 的最佳实践）"""
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(
                str(self.path),
                timeout=30.0,
                isolation_level=None,  # 手动控制事务
            )
            c.row_factory = sqlite3.Row
            # WAL 模式：读写并发更好，是 SQLite 用于服务端的推荐配置
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            c.execute("PRAGMA foreign_keys=ON")
            self._local.conn = c
        return c

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        """事务上下文：异常回滚"""
        with self._write_lock:
            c = self.conn
            c.execute("BEGIN IMMEDIATE")
            try:
                yield c
                c.execute("COMMIT")
            except Exception:
                c.execute("ROLLBACK")
                raise

    # ── 建表 ─────────────────────────────────────────────

    def init_schema(self) -> None:
        """
        建表。

        注意：这里**不能**用 tx() 包住 executescript()。
        sqlite3 的 executescript 会先隐式 COMMIT 当前事务，导致外层
        BEGIN 失效，随后 COMMIT/ROLLBACK 报
        "cannot commit - no transaction is active"。
        建表本身是幂等的（IF NOT EXISTS），直接执行即可。
        """
        c = self.conn
        c.executescript(
            """
                -- ── 文档表：唯一事实源 ─────────────────────
                CREATE TABLE IF NOT EXISTS documents (
                    id            TEXT PRIMARY KEY,   -- uuid
                    source_path   TEXT NOT NULL,      -- 磁盘路径
                    original_name TEXT,               -- 用户上传时的文件名
                    content_hash  TEXT,               -- sha256，用于幂等
                    status        TEXT NOT NULL DEFAULT 'PENDING',
                    version       INTEGER NOT NULL DEFAULT 0,
                    chunks_count  INTEGER DEFAULT 0,
                    entities_count INTEGER DEFAULT 0,
                    relations_count INTEGER DEFAULT 0,
                    retry_count   INTEGER DEFAULT 0,
                    error         TEXT,
                    -- 数据级 ACL：可见范围（逗号分隔的角色/用户）
                    acl_scope     TEXT NOT NULL DEFAULT 'public',
                    owner         TEXT DEFAULT '',
                    created_at    REAL NOT NULL,
                    updated_at    REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_doc_status ON documents(status);
                CREATE INDEX IF NOT EXISTS idx_doc_hash   ON documents(content_hash);

                -- ── 发件箱：可重放的派生任务 ────────────────
                CREATE TABLE IF NOT EXISTS outbox (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id      TEXT NOT NULL,
                    task_type   TEXT NOT NULL,   -- vector | graph
                    payload     TEXT NOT NULL,   -- JSON
                    status      TEXT NOT NULL DEFAULT 'PENDING',
                    retry_count INTEGER DEFAULT 0,
                    error       TEXT,
                    created_at  REAL NOT NULL,
                    processed_at REAL,
                    FOREIGN KEY(doc_id) REFERENCES documents(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_outbox_status
                    ON outbox(status, created_at);

                -- ── 变更追踪：取代进程内存态 ────────────────
                CREATE TABLE IF NOT EXISTS file_versions (
                    source_path  TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    version      INTEGER NOT NULL DEFAULT 1,
                    updated_at   REAL NOT NULL
                );

                -- ── 实体消歧表 ──────────────────────────────
                CREATE TABLE IF NOT EXISTS entity_aliases (
                    canonical_id   TEXT NOT NULL,
                    canonical_name TEXT NOT NULL,
                    alias          TEXT NOT NULL,
                    entity_type    TEXT NOT NULL DEFAULT '',
                    created_at     REAL NOT NULL,
                    PRIMARY KEY (canonical_id, alias)
                );
                CREATE INDEX IF NOT EXISTS idx_alias_alias
                    ON entity_aliases(alias);

                -- ── 审计日志（问答/上传等敏感操作）──────────
                CREATE TABLE IF NOT EXISTS audit_log (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts         REAL NOT NULL,
                    actor      TEXT,               -- API Key 指纹
                    action     TEXT NOT NULL,      -- qa.ask / graph.query / doc.upload ...
                    resource   TEXT,               -- 对象标识
                    detail     TEXT,               -- JSON
                    result     TEXT                -- ok / denied / error
                );
                CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(ts);

                -- ── 问答用量：每轮一行 ──────────────────────
                -- 与 audit_log 分开的理由：审计关心"谁做了什么"，
                -- 用量关心"花了多少"，两者的保留期与查询模式不同。
                CREATE TABLE IF NOT EXISTS qa_metrics (
                    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts                   REAL NOT NULL,
                    -- 复用日志中间件的 request_id，用于从轮汇总下钻到单次调用
                    trace_id             TEXT,
                    actor                TEXT,
                    session_id           TEXT NOT NULL,
                    -- 服务端维护的 session 内递增轮次（客户端不上报轮次）
                    turn                 INTEGER NOT NULL,
                    question             TEXT,
                    intent               TEXT,
                    qa_mode              TEXT,
                    prompt_tokens        INTEGER DEFAULT 0,
                    completion_tokens    INTEGER DEFAULT 0,
                    total_tokens         INTEGER DEFAULT 0,
                    -- 缓存口径只统计"可测"的调用（见 services/usage_meter.py）
                    cache_hit_tokens     INTEGER DEFAULT 0,
                    cache_miss_tokens    INTEGER DEFAULT 0,
                    cache_measured_calls INTEGER DEFAULT 0,
                    cache_hit_calls      INTEGER DEFAULT 0,
                    cache_supported      INTEGER DEFAULT 1,
                    llm_calls            INTEGER DEFAULT 0,
                    latency_ms           REAL DEFAULT 0,
                    result               TEXT,
                    error                TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_qa_metrics_session
                    ON qa_metrics(session_id, ts);
                CREATE INDEX IF NOT EXISTS idx_qa_metrics_trace
                    ON qa_metrics(trace_id);

                -- ── 问答用量：每次 LLM 调用一行 ─────────────
                -- 拆表的理由：轮表只能回答"花了多少"，回答不了
                -- "哪一步花的"。按 stage 展开后，token 异常可直接定位到
                -- intent / rewrite / react_step2 / generate。
                CREATE TABLE IF NOT EXISTS qa_metrics_calls (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id          TEXT NOT NULL,
                    -- actor 冗余在此表：下钻查询自身即可做数据隔离，
                    -- 不必依赖调用方先查轮表验证归属
                    actor             TEXT,
                    session_id        TEXT NOT NULL,
                    turn              INTEGER NOT NULL,
                    call_index        INTEGER NOT NULL,
                    stage             TEXT,
                    model             TEXT,
                    prompt_tokens     INTEGER DEFAULT 0,
                    completion_tokens INTEGER DEFAULT 0,
                    total_tokens      INTEGER DEFAULT 0,
                    cache_hit_tokens  INTEGER DEFAULT 0,
                    cache_miss_tokens INTEGER DEFAULT 0,
                    cache_supported   INTEGER DEFAULT 1
                );
                CREATE INDEX IF NOT EXISTS idx_qmc_trace
                    ON qa_metrics_calls(actor, trace_id, call_index);

                -- ── 对话历史：每轮一行（内容快照）──────────────
                -- 与 qa_metrics 分开的理由：
                --   qa_metrics = "花了多少"（小、留久、可聚合）
                --   qa_messages = "内容"（约 5KB/轮、30 天 TTL、只按会话读）
                -- 混在一张表里，以后想只清内容就清不动了。
                CREATE TABLE IF NOT EXISTS qa_messages (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts              REAL NOT NULL,
                    trace_id        TEXT,
                    actor           TEXT,
                    session_id      TEXT NOT NULL,
                    -- 与 qa_metrics.turn 同值（轮次的权威仍在 qa_metrics）
                    turn            INTEGER NOT NULL,
                    payload_version INTEGER NOT NULL DEFAULT 1,
                    -- 当轮响应快照 JSON。存快照而不是拆列，是为了让前端恢复
                    -- 复用同一条渲染路径，不可能出现"恢复的卡片 ≠ 实时渲染"。
                    payload         TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_qam_session
                    ON qa_messages(actor, session_id, turn);
                CREATE INDEX IF NOT EXISTS idx_qam_ts ON qa_messages(ts);
                """
        )
        self._migrate_qa_metrics()
        logger.info("SQLite 持久化层已就绪", extra={"extra_fields": {"path": str(self.path)}})

    # 问答用量表的增量列（老库需要补，CREATE TABLE IF NOT EXISTS 不会加列）
    _QA_METRICS_NEW_COLUMNS: tuple[tuple[str, str], ...] = (
        ("cache_creation_tokens", "INTEGER"),
        ("stable_prefix_tokens", "INTEGER DEFAULT 0"),
        ("context_tokens", "INTEGER DEFAULT 0"),
        ("query_tokens", "INTEGER DEFAULT 0"),
        ("stable_prefix_reuse_rate", "REAL"),
        ("rag_context_reuse_rate", "REAL"),
        ("answer_prompt_mode", "TEXT DEFAULT ''"),
    )
    _QA_CALLS_NEW_COLUMNS: tuple[tuple[str, str], ...] = (
        ("cache_creation_tokens", "INTEGER"),
        ("stable_prefix_tokens", "INTEGER DEFAULT 0"),
        ("context_tokens", "INTEGER DEFAULT 0"),
        ("query_tokens", "INTEGER DEFAULT 0"),
    )

    def _migrate_qa_metrics(self) -> None:
        """给已存在的 qa_metrics / qa_metrics_calls 补新增列。

        `CREATE TABLE IF NOT EXISTS` 对**已存在**的表是空操作，所以新增列
        必须显式 ALTER。没有这一步，老库上会报 "no such column"。
        """
        for table, columns in (
            ("qa_metrics", self._QA_METRICS_NEW_COLUMNS),
            ("qa_metrics_calls", self._QA_CALLS_NEW_COLUMNS),
        ):
            try:
                existing = {
                    str(r[1]) for r in self.conn.execute(f"PRAGMA table_info({table})")
                }
            except Exception:
                logger.warning("读取表结构失败: %s", table, exc_info=True)
                continue
            if not existing:
                continue
            for name, decl in columns:
                if name in existing:
                    continue
                try:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
                    logger.info(
                        "已补充列", extra={"extra_fields": {"table": table, "column": name}}
                    )
                except Exception:
                    logger.warning(
                        "补充列失败: %s.%s", table, name, exc_info=True
                    )

    # ── 文档 CRUD ────────────────────────────────────────

    def create_document(
        self,
        source_path: str,
        original_name: str = "",
        content_hash: str = "",
        acl_scope: str = "public",
        owner: str = "",
    ) -> str:
        """登记一篇文档，返回 doc_id（uuid）。

        幂等：同一 content_hash 已存在且已提交时，直接返回原 doc_id，
        避免重复消耗 LLM 额度。
        """
        now = time.time()
        if content_hash:
            row = self.fetchone(
                "SELECT id, status FROM documents WHERE content_hash = ? "
                "AND status = ? LIMIT 1",
                (content_hash, DocStatus.COMMITTED),
            )
            if row:
                logger.info(
                    "文档内容已存在，跳过重复入库",
                    extra={"extra_fields": {"doc_id": row["id"]}},
                )
                return row["id"]

        doc_id = uuid.uuid4().hex
        with self.tx() as c:
            c.execute(
                "INSERT INTO documents (id, source_path, original_name, "
                "content_hash, status, acl_scope, owner, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (doc_id, source_path, original_name, content_hash,
                 DocStatus.PENDING, acl_scope, owner, now, now),
            )
        return doc_id

    def set_status(
        self,
        doc_id: str,
        status: str,
        error: str | None = None,
        **fields: Any,
    ) -> None:
        """更新文档状态（状态机流转的唯一入口）"""
        sets = ["status = ?", "updated_at = ?"]
        vals: list[Any] = [status, time.time()]
        if error is not None:
            sets.append("error = ?")
            vals.append(error)
        for k, v in fields.items():
            sets.append(f"{k} = ?")
            vals.append(v)
        vals.append(doc_id)
        with self.tx() as c:
            c.execute(
                f"UPDATE documents SET {', '.join(sets)} WHERE id = ?", vals
            )

    def bump_retry(self, doc_id: str) -> int:
        with self.tx() as c:
            c.execute(
                "UPDATE documents SET retry_count = retry_count + 1, "
                "updated_at = ? WHERE id = ?",
                (time.time(), doc_id),
            )
        row = self.fetchone("SELECT retry_count FROM documents WHERE id = ?", (doc_id,))
        return int(row["retry_count"]) if row else 0

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        row = self.fetchone("SELECT * FROM documents WHERE id = ?", (doc_id,))
        return dict(row) if row else None

    def list_documents(self, limit: int = 200, acl_scopes: list[str] | None = None) -> list[dict]:
        """列出文档，可选按 ACL 过滤"""
        if acl_scopes:
            ph = ",".join("?" * len(acl_scopes))
            rows = self.fetchall(
                f"SELECT * FROM documents WHERE acl_scope IN ({ph}) "
                "ORDER BY created_at DESC LIMIT ?",
                (*acl_scopes, limit),
            )
        else:
            rows = self.fetchall(
                "SELECT * FROM documents ORDER BY created_at DESC LIMIT ?", (limit,)
            )
        return [dict(r) for r in rows]

    def doc_count(self) -> int:
        row = self.fetchone("SELECT COUNT(*) AS c FROM documents")
        return int(row["c"]) if row else 0

    # ── 入库队列（worker 消费）───────────────────────────

    def list_pending_documents(self, limit: int = 5) -> list[dict]:
        """
        取出待处理的文档 —— **在 SQL 层过滤，按 FIFO 排序**。

        为什么必须这样写（这是一次真实的生产事故修复）
        ------------------------------------------------
        原实现是：
            docs = [d for d in db.list_documents(limit=5)
                    if d["status"] == PENDING]
        而 list_documents 是 `ORDER BY created_at DESC LIMIT 5` ——
        **先取最新 5 行，再筛 PENDING**。

        后果（实测）：最新 5 行是 4 个 COMMITTED + 1 个 PROCESSING，
        筛完 PENDING 数为 0；而 20 个真正的 PENDING 全部排在第 6 名之后，
        worker **永远看不到它们** —— 队列确定性死锁，不会自愈。

        为什么用 ASC（FIFO）而不是 DESC：
            DESC 会让刚上传的任务不断插队，早先的任务可能永远饿死。
            ASC 保证先到先处理，与"上传顺序"一致，行为可预期。

        注意：这里**不做** limit 之后再过滤 —— 过滤条件进 SQL，
        否则同样的截断问题会以另一种形式重现。
        """
        rows = self.fetchall(
            "SELECT * FROM documents WHERE status = ? "
            "ORDER BY created_at ASC LIMIT ?",
            (DocStatus.PENDING, limit),
        )
        return [dict(r) for r in rows]

    def reclaim_stuck_documents(self, timeout_seconds: float = 1800.0) -> int:
        """
        回收卡死的 PROCESSING 文档 —— 复位为 PENDING 等待重试。

        为什么需要
        ----------
        `_process_document` 先置 PROCESSING 再跑流水线。若进程在处理中途
        被杀（重启、崩溃、OOM），**不会走 except 分支**，状态就永久停在
        PROCESSING。由于 worker 只取 PENDING，该文档再也不会被处理，
        同时它还会占住"最新 5 行"的窗口（见 list_pending_documents 的说明）。

        超时阈值默认 30 分钟：远大于单篇文档的正常处理时间（约 2-3 分钟），
        避免误伤正在处理的任务。

        返回复位的行数。
        """
        cutoff = time.time() - timeout_seconds
        with self.tx() as c:
            cur = c.execute(
                "UPDATE documents SET status = ?, updated_at = ?, "
                "error = COALESCE(error, '') || '[reclaimed: PROCESSING 超时]' "
                "WHERE status = ? AND updated_at < ?",
                (DocStatus.PENDING, time.time(), DocStatus.PROCESSING, cutoff),
            )
            return int(cur.rowcount or 0)

    def fail_document(self, doc_id: str, error: str) -> None:
        """标记文档为 FAILED（超过重试上限时使用）"""
        self.set_status(doc_id, DocStatus.FAILED, error=error[:500])

    # ── Outbox ───────────────────────────────────────────

    def enqueue(self, doc_id: str, task_type: str, payload: dict) -> int:
        """投递一个派生任务（vector / graph）"""
        with self.tx() as c:
            cur = c.execute(
                "INSERT INTO outbox (doc_id, task_type, payload, status, created_at) "
                "VALUES (?,?,?,?,?)",
                (doc_id, task_type, json.dumps(payload, ensure_ascii=False),
                 "PENDING", time.time()),
            )
            return int(cur.lastrowid or 0)

    def claim_tasks(self, limit: int = 10, task_types: tuple[str, ...] | None = None) -> list[dict]:
        """取出待处理任务（并标记为 PROCESSING，防止重复消费）"""
        types = task_types or ("vector", "graph")
        ph = ",".join("?" * len(types))
        with self.tx() as c:
            rows = c.execute(
                f"SELECT * FROM outbox WHERE status = 'PENDING' AND task_type IN ({ph}) "
                "ORDER BY created_at LIMIT ?",
                (*types, limit),
            ).fetchall()
            ids = [r["id"] for r in rows]
            if ids:
                qs = ",".join("?" * len(ids))
                c.execute(
                    f"UPDATE outbox SET status = 'PROCESSING' WHERE id IN ({qs})", ids
                )
        # 注意：必须在 UPDATE 之后再反映新状态。
        # 之前直接返回 SELECT 到的行，里面还是 PENDING —— 调用方若据此
        # 判断会误认为任务未被占用。
        out = []
        for r in rows:
            d = dict(r)
            d["status"] = "PROCESSING"
            out.append(d)
        return out

    def finish_task(self, task_id: int, ok: bool, error: str = "") -> None:
        with self.tx() as c:
            if ok:
                c.execute(
                    "UPDATE outbox SET status = 'DONE', processed_at = ? WHERE id = ?",
                    (time.time(), task_id),
                )
            else:
                c.execute(
                    "UPDATE outbox SET status = 'PENDING', retry_count = retry_count + 1, "
                    "error = ? WHERE id = ?",
                    (error[:500], task_id),
                )

    def outbox_stats(self) -> dict[str, int]:
        rows = self.fetchall(
            "SELECT status, COUNT(*) AS c FROM outbox GROUP BY status"
        )
        return {r["status"]: int(r["c"]) for r in rows}

    def pending_tasks(self, limit: int = 50) -> list[dict]:
        rows = self.fetchall(
            "SELECT * FROM outbox WHERE status = 'PENDING' ORDER BY created_at LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in rows]

    # ── 文件版本（取代内存态 _file_hashes）───────────────

    def get_file_hash(self, source_path: str) -> str | None:
        row = self.fetchone(
            "SELECT content_hash FROM file_versions WHERE source_path = ?",
            (source_path,),
        )
        return row["content_hash"] if row else None

    def set_file_hash(self, source_path: str, content_hash: str) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT INTO file_versions (source_path, content_hash, version, updated_at) "
                "VALUES (?,?,1,?) "
                "ON CONFLICT(source_path) DO UPDATE SET "
                "content_hash = excluded.content_hash, "
                "version = file_versions.version + 1, "
                "updated_at = excluded.updated_at",
                (source_path, content_hash, time.time()),
            )

    # ── 实体别名（消歧）──────────────────────────────────

    def register_alias(
        self, canonical_id: str, canonical_name: str, alias: str, entity_type: str = ""
    ) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT OR IGNORE INTO entity_aliases "
                "(canonical_id, canonical_name, alias, entity_type, created_at) "
                "VALUES (?,?,?,?,?)",
                (canonical_id, canonical_name, alias, entity_type, time.time()),
            )

    def lookup_alias(self, alias: str) -> dict | None:
        row = self.fetchone(
            "SELECT * FROM entity_aliases WHERE alias = ? LIMIT 1", (alias,)
        )
        return dict(row) if row else None

    def all_aliases(self, limit: int = 5000) -> list[dict]:
        rows = self.fetchall(
            "SELECT * FROM entity_aliases LIMIT ?", (limit,)
        )
        return [dict(r) for r in rows]

    # ── 审计 ─────────────────────────────────────────────

    def audit(
        self,
        action: str,
        actor: str = "",
        resource: str = "",
        detail: dict | None = None,
        result: str = "ok",
    ) -> None:
        try:
            with self.tx() as c:
                c.execute(
                    "INSERT INTO audit_log (ts, actor, action, resource, detail, result) "
                    "VALUES (?,?,?,?,?,?)",
                    (time.time(), actor, action, resource,
                     json.dumps(detail or {}, ensure_ascii=False), result),
                )
        except Exception:
            logger.warning("审计日志写入失败", exc_info=True)

    def recent_audit(self, limit: int = 50) -> list[dict]:
        rows = self.fetchall(
            "SELECT * FROM audit_log ORDER BY ts DESC LIMIT ?", (limit,)
        )
        return [dict(r) for r in rows]

    # ── 问答用量 ─────────────────────────────────────────

    def record_qa_metrics(
        self,
        *,
        actor: str,
        session_id: str,
        question: str,
        intent: str,
        qa_mode: str,
        usage: dict,
        latency_ms: float = 0.0,
        trace_id: str = "",
        result: str = "ok",
        error: str = "",
        answer_prompt_mode: str = "",
    ) -> tuple[int, int]:
        """写入一轮问答的用量，返回 (row_id, turn)。

        `turn` 由**服务端**按 `session_id + actor` 的记录数递增取号 ——
        客户端不上报轮次，避免"客户端说第 3 轮 / DB 算出第 4 轮"的歧义。

        并发前提：取号与插入在同一个事务（复用 `_write_lock`）内完成，
        因此**单进程内**不会重号。多进程 / 多实例部署下 SQLite 的行锁
        无法跨进程串行化，此处不再保证唯一 —— 届时应改为数据库原子序列
        或唯一约束。详见 docs/chat-metrics.md。
        """
        calls = usage.get("per_call") or []
        with self.tx() as c:
            row = c.execute(
                "SELECT COUNT(*) FROM qa_metrics WHERE session_id = ? AND actor = ?",
                (session_id, actor),
            ).fetchone()
            turn = int(row[0] if row else 0) + 1

            cur = c.execute(
                "INSERT INTO qa_metrics ("
                " ts, trace_id, actor, session_id, turn, question, intent, qa_mode,"
                " prompt_tokens, completion_tokens, total_tokens,"
                " cache_hit_tokens, cache_miss_tokens, cache_measured_calls,"
                " cache_hit_calls, cache_supported, llm_calls, latency_ms, result, error,"
                " cache_creation_tokens, stable_prefix_tokens, context_tokens,"
                " query_tokens, stable_prefix_reuse_rate, rag_context_reuse_rate,"
                " answer_prompt_mode"
                ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    time.time(), trace_id, actor, session_id, turn,
                    (question or "")[:500], intent, qa_mode,
                    int(usage.get("prompt_tokens") or 0),
                    int(usage.get("completion_tokens") or 0),
                    int(usage.get("total_tokens") or 0),
                    int(usage.get("cache_hit_tokens") or 0),
                    int(usage.get("cache_miss_tokens") or 0),
                    int(usage.get("cache_measured_calls") or 0),
                    int(usage.get("cache_hit_calls") or 0),
                    1 if usage.get("cache_supported") else 0,
                    int(usage.get("llm_calls") or 0),
                    float(latency_ms or 0.0),
                    result, (error or "")[:500],
                    # 写入侧：Provider 不给就是 NULL，不要写成 0
                    (None if usage.get("cache_creation_tokens") is None
                     else int(usage.get("cache_creation_tokens") or 0)),
                    int(usage.get("stable_prefix_tokens") or 0),
                    int(usage.get("context_tokens") or 0),
                    int(usage.get("query_tokens") or 0),
                    (None if usage.get("stable_prefix_reuse_rate") is None
                     else float(usage["stable_prefix_reuse_rate"])),
                    (None if usage.get("rag_context_reuse_rate") is None
                     else float(usage["rag_context_reuse_rate"])),
                    str(answer_prompt_mode or ""),
                ),
            )
            row_id = int(cur.lastrowid or 0)

            for call in calls:
                c.execute(
                    "INSERT INTO qa_metrics_calls ("
                    " trace_id, actor, session_id, turn, call_index, stage, model,"
                    " prompt_tokens, completion_tokens, total_tokens,"
                    " cache_hit_tokens, cache_miss_tokens, cache_supported,"
                    " cache_creation_tokens, stable_prefix_tokens, context_tokens,"
                    " query_tokens"
                    ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        trace_id, actor, session_id, turn,
                        int(call.get("call_index") or 0),
                        str(call.get("stage") or ""),
                        str(call.get("model") or ""),
                        int(call.get("prompt_tokens") or 0),
                        int(call.get("completion_tokens") or 0),
                        int(call.get("total_tokens") or 0),
                        int(call.get("cache_hit_tokens") or 0),
                        int(call.get("cache_miss_tokens") or 0),
                        1 if call.get("cache_supported") else 0,
                        (None if call.get("cache_creation_tokens") is None
                         else int(call.get("cache_creation_tokens") or 0)),
                        int(call.get("stable_prefix_tokens") or 0),
                        int(call.get("context_tokens") or 0),
                        int(call.get("query_tokens") or 0),
                    ),
                )

        return row_id, turn

    def qa_session_summary(self, session_id: str, actor: str) -> dict:
        """会话累计。

        **必须同时按 session_id 与 actor 过滤** —— session_id 是客户端自有的
        随机串，只按它查询等于"猜到 id 就能读到别人的用量"。
        """
        row = self.fetchone(
            "SELECT COUNT(*) AS turns,"
            "       COALESCE(SUM(total_tokens), 0)      AS total_tokens,"
            "       COALESCE(SUM(prompt_tokens), 0)     AS prompt_tokens,"
            "       COALESCE(SUM(completion_tokens), 0) AS completion_tokens,"
            "       COALESCE(SUM(cache_hit_tokens), 0)  AS cache_hit_tokens,"
            "       COALESCE(SUM(cache_miss_tokens), 0) AS cache_miss_tokens"
            "  FROM qa_metrics WHERE session_id = ? AND actor = ?",
            (session_id, actor),
        )
        d = dict(row) if row else {}
        hit = int(d.get("cache_hit_tokens") or 0)
        miss = int(d.get("cache_miss_tokens") or 0)
        denom = hit + miss
        return {
            "session_id": session_id,
            "turns": int(d.get("turns") or 0),
            "total_tokens": int(d.get("total_tokens") or 0),
            "prompt_tokens": int(d.get("prompt_tokens") or 0),
            "completion_tokens": int(d.get("completion_tokens") or 0),
            "cache_hit_tokens": hit,
            "cache_miss_tokens": miss,
            "cache_hit_rate": round(hit / denom, 6) if denom else 0.0,
        }

    def qa_session_turns(self, session_id: str, actor: str, limit: int = 10) -> list[dict]:
        """会话内最近若干轮（时间正序，便于直接渲染）"""
        rows = self.fetchall(
            "SELECT turn, ts, question, intent, total_tokens,"
            "       cache_hit_tokens, cache_miss_tokens, cache_measured_calls,"
            "       llm_calls, latency_ms"
            "  FROM qa_metrics WHERE session_id = ? AND actor = ?"
            " ORDER BY turn DESC LIMIT ?",
            (session_id, actor, max(1, min(int(limit), 50))),
        )
        return [dict(r) for r in reversed(rows)]

    def qa_trace_calls(self, trace_id: str, actor: str) -> list[dict]:
        """按 trace_id 下钻到单次 LLM 调用（**同时按 actor 隔离**）"""
        rows = self.fetchall(
            "SELECT call_index, stage, model, prompt_tokens, completion_tokens,"
            "       total_tokens, cache_hit_tokens, cache_miss_tokens, cache_supported"
            "  FROM qa_metrics_calls WHERE trace_id = ? AND actor = ?"
            " ORDER BY call_index ASC",
            (trace_id, actor),
        )
        return [dict(r) for r in rows]

    def qa_metrics_global(self, since: float | None = None) -> dict:
        """全局累计（为成本看板预留；本期不暴露 HTTP 接口）"""
        sql = (
            "SELECT COUNT(*) AS turns,"
            "       COALESCE(SUM(total_tokens), 0)      AS total_tokens,"
            "       COALESCE(SUM(cache_hit_tokens), 0)  AS cache_hit_tokens,"
            "       COALESCE(SUM(cache_miss_tokens), 0) AS cache_miss_tokens"
            "  FROM qa_metrics"
        )
        params: tuple = ()
        if since is not None:
            sql += " WHERE ts >= ?"
            params = (float(since),)
        d = dict(self.fetchone(sql, params) or {})
        hit = int(d.get("cache_hit_tokens") or 0)
        miss = int(d.get("cache_miss_tokens") or 0)
        denom = hit + miss
        return {
            "turns": int(d.get("turns") or 0),
            "total_tokens": int(d.get("total_tokens") or 0),
            "cache_hit_tokens": hit,
            "cache_miss_tokens": miss,
            "cache_hit_rate": round(hit / denom, 6) if denom else 0.0,
        }

    # ── 对话历史（内容快照）──────────────────────────────

    def record_qa_message(
        self,
        *,
        actor: str,
        session_id: str,
        turn: int,
        payload: dict,
        trace_id: str = "",
        payload_version: int = 1,
    ) -> int:
        """写入一轮问答的内容快照，返回 row_id。

        `turn` 由调用方从 `record_qa_metrics` 的返回值传入 —— 轮次的权威
        在 qa_metrics，这里只沿用，不重新取号（避免两张表各算一次而错位）。
        """
        with self.tx() as c:
            cur = c.execute(
                "INSERT INTO qa_messages ("
                " ts, trace_id, actor, session_id, turn, payload_version, payload"
                ") VALUES (?,?,?,?,?,?,?)",
                (
                    time.time(), trace_id, actor, session_id, int(turn),
                    int(payload_version),
                    json.dumps(payload, ensure_ascii=False),
                ),
            )
        return int(cur.lastrowid or 0)

    def list_qa_messages(
        self, session_id: str, actor: str, limit: int = 50
    ) -> tuple[list[dict], int]:
        """读回会话历史，返回 (按 turn 升序的最近 N 轮, 会话内历史总轮数)。

        **必须同时按 session_id 与 actor 过滤** —— 只按 session_id 查询等于
        "猜到随机 id 就能读到别人的问答内容与出处"。

        损坏的 payload 行会被跳过并告警，不让一行坏数据打挂整段历史。
        """
        n = max(1, min(int(limit), 200))
        total_row = self.fetchone(
            "SELECT COUNT(*) FROM qa_messages WHERE session_id = ? AND actor = ?",
            (session_id, actor),
        )
        total = int(total_row[0] if total_row else 0)

        rows = self.fetchall(
            "SELECT turn, ts, trace_id, payload_version, payload"
            "  FROM qa_messages WHERE session_id = ? AND actor = ?"
            " ORDER BY turn DESC LIMIT ?",
            (session_id, actor, n),
        )

        turns: list[dict] = []
        for r in reversed(rows):
            try:
                data = json.loads(r["payload"])
                if not isinstance(data, dict):
                    raise ValueError("payload 不是对象")
            except Exception:
                logger.warning(
                    "历史 payload 解析失败，跳过该轮",
                    extra={"extra_fields": {"turn": r["turn"], "session_id": session_id}},
                )
                continue
            data.setdefault("turn", int(r["turn"] or 0))
            data.setdefault("trace_id", str(r["trace_id"] or ""))
            data["payload_version"] = int(r["payload_version"] or 1)
            turns.append(data)

        return turns, total

    def purge_qa_messages(self, days: int) -> int:
        """清理超过保留期的历史，返回删除行数。

        `days <= 0` 表示永久保留 —— 直接返回 0，**绝不**把 0 解释成"删光"。
        """
        if int(days) <= 0:
            return 0
        cutoff = time.time() - int(days) * 86400
        with self.tx() as c:
            cur = c.execute("DELETE FROM qa_messages WHERE ts < ?", (cutoff,))
        return int(cur.rowcount or 0)

    def qa_history_stats(self, actor: str = "") -> dict:
        """全局历史规模（诊断用：行数与占用字节）"""
        sql = "SELECT COUNT(*), COALESCE(SUM(LENGTH(payload)), 0) FROM qa_messages"
        params: tuple = ()
        if actor:
            sql += " WHERE actor = ?"
            params = (actor,)
        row = self.fetchone(sql, params)
        return {
            "rows": int(row[0] or 0) if row else 0,
            "payload_bytes": int(row[1] or 0) if row else 0,
        }

    # ── 通用查询 ─────────────────────────────────────────

    def fetchone(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        return self.conn.execute(sql, params).fetchone()

    def fetchall(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchall()

    def close(self) -> None:
        c = getattr(self._local, "conn", None)
        if c is not None:
            c.close()
            self._local.conn = None


# 模块级单例（由 api/main.py 在启动时初始化）
_db: Database | None = None


def init_db(path: str) -> Database:
    global _db
    _db = Database(path)
    return _db


def get_db() -> Database:
    if _db is None:
        raise RuntimeError("数据库未初始化，请先调用 init_db()")
    return _db
