"""
后台 Worker —— 消费 Outbox 任务

为什么需要它
------------
原实现上传接口是**同步阻塞**的：

    POST /upload -> 解析 -> LLM 抽取 -> 双写 -> 返回

一份 50 页 PDF 的完整链路（OCR/视觉兜底 + 每 chunk 一次 LLM 抽取）
需要几分钟，HTTP 层必然超时。

本模块把处理搬到后台：
    POST /upload  -> 落盘 + 登记状态 + 返回 task_id（立即返回）
    Worker        -> 从 outbox 取任务，异步处理
    GET /jobs/{id} -> 查询进度

同时它也是「一致性可恢复」的执行者：
    失败的任务会回到 PENDING 等待重放，超过上限则标记 FAILED。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from services.database import DocStatus, get_db

logger = logging.getLogger(__name__)


class IngestWorker:
    """
    单个后台 worker 循环。

    注意：单机部署时随 API 进程启动；多副本部署应独立成进程，
    并依赖 outbox 的 status 字段做互斥（claim_tasks 会把任务
    标记为 PROCESSING，避免多实例重复消费）。
    """

    def __init__(
        self,
        workflows: dict[str, Any],
        vector_store: Any = None,
        poll_interval: float = 2.0,
        max_retries: int = 3,
        stuck_timeout_seconds: float = 1800.0,
    ) -> None:
        self.workflows = workflows
        self.vector_store = vector_store
        self.poll_interval = poll_interval
        self.max_retries = max_retries
        # PROCESSING 超过该时长视为卡死，启动时复位为 PENDING。
        # 默认 30 分钟，远大于单篇正常处理时间（2-3 分钟），避免误伤。
        self.stuck_timeout_seconds = stuck_timeout_seconds
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self.processed = 0
        self.failed = 0

    # ── 生命周期 ─────────────────────────────────────────

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._stop.clear()
            # 启动时先回收卡死的 PROCESSING（进程重启/崩溃遗留）。
            # 必须在循环开始前做，否则这些文档既不被处理、又占住取数窗口。
            try:
                n = get_db().reclaim_stuck_documents(self.stuck_timeout_seconds)
                if n:
                    logger.warning(
                        "回收了卡死的 PROCESSING 文档，已复位为 PENDING",
                        extra={"extra_fields": {"count": n}},
                    )
            except Exception:
                logger.warning("回收卡死文档失败", exc_info=True)

            self._task = asyncio.create_task(self._loop(), name="ingest-worker")
            logger.info("后台 worker 已启动")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
            logger.info(
                "后台 worker 已停止",
                extra={"extra_fields": {
                    "processed": self.processed, "failed": self.failed,
                }},
            )

    # ── 主循环 ───────────────────────────────────────────

    async def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                did_work = await self._tick()
                if not did_work:
                    await asyncio.sleep(self.poll_interval)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.error("worker 循环异常", exc_info=True)
                await asyncio.sleep(self.poll_interval)

    async def _tick(self) -> bool:
        """处理一轮任务，返回是否有任务被处理"""
        db = get_db()

        # 1) 先处理「文档级」任务：PENDING 的文档走完整入库流水线
        #
        # 修复（生产事故）：原实现是
        #     docs = [d for d in db.list_documents(limit=5)
        #             if d["status"] == PENDING]
        # 即**先取最新 5 行、再筛 PENDING**。当最新 5 行恰好都是
        # COMMITTED/PROCESSING 时，筛完为空，而真正的 PENDING 排在第 6 名
        # 之后 —— worker 永远看不到，队列确定性死锁（实测 20 篇文档卡死）。
        #
        # 现在过滤下推到 SQL，且按 created_at ASC（FIFO），
        # 避免新任务不断插队导致旧任务饿死。
        docs = db.list_pending_documents(limit=5)
        if docs:
            for doc in docs:
                await self._process_document(doc)
            return True

        # 2) 再处理 outbox 里遗留的派生任务（失败重放走这里）
        tasks = db.claim_tasks(limit=5)
        if tasks:
            for t in tasks:
                await self._process_task(t)
            return True

        return False

    async def _process_document(self, doc: dict) -> None:
        """
        走完整入库流水线。

        状态机契约（**必须保证不会永久停在 PROCESSING**）
        -------------------------------------------------
        PENDING -> PROCESSING -> (VECTOR_DONE -> GRAPH_DONE -> COMMITTED)
                              \\-> 失败：回 PENDING 重试，或 FAILED

        历史问题：原实现只在 `except` 里复位状态。若进程在处理中途被杀
        （重启/崩溃），不走 except，状态永久停在 PROCESSING，
        该文档再也不被处理，并占住 worker 的取数窗口（造成队列死锁）。

        `finally` 分支的作用不是"处理成功"，而是**兜底**：
        无论正常结束、抛异常、还是被取消，都检查一次状态，
        确保没有留在 PROCESSING。
        """
        db = get_db()
        doc_id = doc["id"]
        try:
            db.set_status(doc_id, DocStatus.PROCESSING)
            wf = self.workflows.get("ingest")
            if not wf:
                raise RuntimeError("入库流水线未就绪")

            await wf.ainvoke({
                "doc_id": doc_id,
                "file_paths": [doc["source_path"]],
                "vectors_stored": 0,
                "entities_stored": 0,
                "relations_stored": 0,
            })
            self.processed += 1
        except asyncio.CancelledError:
            # 被取消（如进程关闭）：立即复位，避免留下卡死状态
            try:
                db.set_status(doc_id, DocStatus.PENDING, error="cancelled")
            except Exception:
                pass
            raise
        except Exception as e:
            retries = db.bump_retry(doc_id)
            if retries >= self.max_retries:
                db.fail_document(doc_id, str(e))
                self.failed += 1
                logger.error(
                    "文档处理失败且已达重试上限",
                    extra={"extra_fields": {"doc_id": doc_id, "retries": retries}},
                )
            else:
                # 回到 PENDING 等待下一轮重试
                db.set_status(doc_id, DocStatus.PENDING, error=str(e)[:500])
                logger.warning(
                    "文档处理失败，将重试",
                    extra={"extra_fields": {"doc_id": doc_id, "retry": retries}},
                )
        finally:
            # 兜底：若因任何未预期原因仍停在 PROCESSING，复位为 PENDING。
            # 正常成功时状态已是 COMMITTED，这里不会改动它。
            try:
                cur = db.get_document(doc_id)
                if cur and cur.get("status") == DocStatus.PROCESSING:
                    db.set_status(
                        doc_id, DocStatus.PENDING,
                        error="worker 未能推进状态，已复位待重试",
                    )
                    logger.warning(
                        "文档状态未推进，已复位于 PENDING",
                        extra={"extra_fields": {"doc_id": doc_id}},
                    )
            except Exception:
                logger.warning("finally 复位检查失败", exc_info=True)

    async def _process_task(self, task: dict) -> None:
        """处理 outbox 派生任务（失败重放）"""
        from orchestrator.graph import replay_document

        db = get_db()
        try:
            r = await replay_document(task["doc_id"], self.vector_store)
            ok = bool(r.get("ok"))
            db.finish_task(task["id"], ok, error="" if ok else str(r.get("error", ""))[:300])
            if ok:
                self.processed += 1
            else:
                self.failed += 1
        except Exception as e:
            db.finish_task(task["id"], False, str(e)[:300])
            self.failed += 1

    # ── 状态 ─────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        return {
            "running": self._task is not None and not self._task.done(),
            "processed": self.processed,
            "failed": self.failed,
        }
