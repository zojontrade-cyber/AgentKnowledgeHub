"""
知识更新 Agent — 检测文档变更并更新向量库

⚠️ 现状说明
----------
本模块的 detect_changes() / commit_hash() **没有任何调用方**。

**当前真正生效的“更新”是重新上传**（POST /api/ingest/upload）：
  - 内容哈希已存在且 COMMITTED -> 幂等跳过，返回 duplicate
  - 内容变了                  -> 新建 doc_id，整篇重新解析+抽取+建索引

即整篇替换。
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from config import settings

logger = logging.getLogger(__name__)


class ChangeType(str, Enum):
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"


@dataclass
class DocumentChange:
    file_path: str
    change_type: ChangeType
    timestamp: float = field(default_factory=time.time)
    old_hash: str = ""
    new_hash: str = ""
    diff_chunks: list[str] = field(default_factory=list)


@dataclass
class UpdateResult:
    change: DocumentChange
    vectors_added: int = 0
    vectors_deleted: int = 0
    entities_added: int = 0
    entities_updated: int = 0
    relations_added: int = 0
    success: bool = True
    error: str = ""
    processing_time_ms: float = 0


class KnowledgeUpdateAgent:
    """
    知识更新 Agent

    工作流（process_change）:
      CREATED  -> 解析 + 写入向量
      MODIFIED -> 按 doc_id 删除旧向量 + 重新解析写入
      DELETED  -> 按 doc_id 删除向量

    注意：这里的“更新”是**整篇替换**，不做块级 diff。
    """

    def __init__(
        self,
        doc_parser: Any = None,
        knowledge_extractor: Any = None,
        vector_store: Any = None,
    ) -> None:
        self.doc_parser = doc_parser
        self.knowledge_extractor = knowledge_extractor
        self.vector_store = vector_store
        self._file_hashes: dict[str, str] = {}
        self._version_counter: dict[str, int] = {}

    # ── public API ───────────────────────────────────────────

    async def process_change(self, change: DocumentChange) -> UpdateResult:
        """处理单个文档变更"""
        start = time.time()
        result = UpdateResult(change=change)

        logger.info(
            "开始处理文档变更",
            extra={"extra_fields": {
                "file": change.file_path,
                "change_type": change.change_type.value,
            }},
        )

        try:
            if change.change_type == ChangeType.DELETED:
                await self._handle_delete(change, result)
            elif change.change_type == ChangeType.CREATED:
                await self._handle_create(change, result)
            elif change.change_type == ChangeType.MODIFIED:
                await self._handle_modify(change, result)
        except Exception as e:
            result.success = False
            result.error = str(e)
            # 原实现只把错误塞进 result 就返回，无人读取 → 失败无痕
            logger.error(
                "文档变更处理失败",
                exc_info=True,
                extra={"extra_fields": {
                    "file": change.file_path,
                    "change_type": change.change_type.value,
                }},
            )

        result.processing_time_ms = (time.time() - start) * 1000
        logger.info(
            "文档变更处理完成",
            extra={"extra_fields": {
                "file": change.file_path,
                "success": result.success,
                "vectors_added": result.vectors_added,
                "entities_added": result.entities_added,
                "relations_added": result.relations_added,
                "elapsed_ms": round(result.processing_time_ms, 1),
            }},
        )
        return result

    async def process_batch(self, changes: list[DocumentChange]) -> list[UpdateResult]:
        """批量处理文档变更"""
        results: list[UpdateResult] = []
        for change in changes:
            results.append(await self.process_change(change))
        return results

    def detect_changes(self, file_paths: list[str]) -> list[DocumentChange]:
        """
        扫描文件列表，检测变更。

        修复要点：原实现在检测阶段就把新哈希写入 `_file_hashes`（乐观记账，
        见原 :164），导致若随后处理失败，该文件**永远不会被再次检测为变更**。
        现在检测只读取哈希，处理成功后由 commit_hash() 显式提交。
        """
        changes: list[DocumentChange] = []
        current_files = set(file_paths)

        for fp in current_files:
            new_hash = self._compute_hash(fp)
            old_hash = self._file_hashes.get(fp, "")

            if not old_hash:
                changes.append(DocumentChange(
                    file_path=fp,
                    change_type=ChangeType.CREATED,
                    new_hash=new_hash,
                ))
            elif new_hash != old_hash:
                changes.append(DocumentChange(
                    file_path=fp,
                    change_type=ChangeType.MODIFIED,
                    old_hash=old_hash,
                    new_hash=new_hash,
                ))
            # 注意：此处不再写 _file_hashes，改由 commit_hash 在处理成功后提交

        for fp in set(self._file_hashes) - current_files:
            changes.append(DocumentChange(
                file_path=fp,
                change_type=ChangeType.DELETED,
                old_hash=self._file_hashes[fp],
            ))

        return changes

    def commit_hash(self, file_path: str) -> None:
        """处理成功后提交哈希，使其不再被判定为变更"""
        self._file_hashes[file_path] = self._compute_hash(file_path)

    def forget_hash(self, file_path: str) -> None:
        """文件已删除时清除哈希记录"""
        self._file_hashes.pop(file_path, None)

    # ── internal handlers ────────────────────────────────────

    async def _handle_create(self, change: DocumentChange, result: UpdateResult) -> None:
        if not self.doc_parser:
            return
        chunks = await self.doc_parser.parse(change.file_path)

        if self.vector_store:
            await self.vector_store.add_chunks(chunks)
            result.vectors_added = len(chunks)


    async def _handle_modify(self, change: DocumentChange, result: UpdateResult) -> None:
        doc_id = hashlib.sha256(change.file_path.encode()).hexdigest()[:16]

        if self.vector_store:
            deleted = await self.vector_store.delete_by_doc_id(doc_id)
            result.vectors_deleted = deleted

        await self._handle_create(change, result)

    async def _handle_delete(self, change: DocumentChange, result: UpdateResult) -> None:
        doc_id = hashlib.sha256(change.file_path.encode()).hexdigest()[:16]

        if self.vector_store:
            deleted = await self.vector_store.delete_by_doc_id(doc_id)
            result.vectors_deleted = deleted


    # ── utilities ────────────────────────────────────────────

    @staticmethod
    def _compute_hash(file_path: str) -> str:
        try:
            with open(file_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except FileNotFoundError:
            return ""

    def _bump_version(self, entity_name: str) -> int:
        ver = self._version_counter.get(entity_name, 0) + 1
        self._version_counter[entity_name] = ver
        return ver
