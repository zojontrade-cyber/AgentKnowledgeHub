"""
知识抽取 Agent — 从文档块中提取实体、关系、事件

⚠️ 当前**没有消费者**：抽取结果原本用于写入图数据库，
但入库流水线当前并不使用它的输出。
入库流水线现在只做「解析 → 建向量索引」，不调用本 Agent。

核心能力:
  1. 命名实体识别 (NER)
  2. 关系抽取 (RE)
  3. 事件抽取
  4. 三元组生成（当前无消费者，见模块头说明）
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agents.doc_parser_agent import DocumentChunk
from config import settings
from services.llm_factory import LazyLLM

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """\
你是一个专业的知识抽取引擎。给定一段文本，请提取其中的：
1. **实体 (entities)**：人名、组织、地点、产品、技术、概念等
2. **关系 (relations)**：实体之间的关系，用三元组 (头实体, 关系, 尾实体) 表示
3. **事件 (events)**：文本中提到的事件，包含触发词和参与者

请严格按照以下 JSON 格式返回：
{
  "entities": [
    {"name": "实体名", "type": "实体类型", "description": "简短描述"}
  ],
  "relations": [
    {"head": "头实体", "relation": "关系类型", "tail": "尾实体", "confidence": 0.95}
  ],
  "events": [
    {"trigger": "触发词", "type": "事件类型", "participants": ["参与者1"]}
  ]
}

注意:
- 实体类型包括: Person, Organization, Location, Product, Technology, Concept, Event, Time
- 关系类型包括: belongs_to, works_at, located_in, developed_by, related_to, part_of, uses, depends_on
- confidence 为 0-1 之间的浮点数
- 只返回 JSON，不要包含其他文字
"""


@dataclass
class Entity:
    name: str
    type: str
    description: str = ""
    properties: dict[str, Any] = field(default_factory=dict)

    @property
    def node_label(self) -> str:
        return self.type.replace(" ", "_")


@dataclass
class Relation:
    head: str
    relation: str
    tail: str
    confidence: float = 0.0
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeEvent:
    trigger: str
    type: str
    participants: list[str] = field(default_factory=list)


@dataclass
class ExtractionResult:
    entities: list[Entity]
    relations: list[Relation]
    events: list[KnowledgeEvent]
    source_chunk_id: str = ""


class KnowledgeExtractAgent:
    """
    知识抽取 Agent

    工作流:
      receive_chunks → extract_per_chunk → deduplicate → resolve_entities → output_triples
    """

    BATCH_SIZE = 5

    def __init__(self) -> None:
        # 惰性构造，避免未配置 Key 时导入/启动即失败
        self.llm = LazyLLM(temperature=0)

    # ── public API ───────────────────────────────────────────

    async def extract(self, chunks: list[DocumentChunk]) -> list[ExtractionResult]:
        """从一组文档块中抽取知识"""
        results: list[ExtractionResult] = []
        for i in range(0, len(chunks), self.BATCH_SIZE):
            batch = chunks[i : i + self.BATCH_SIZE]
            for chunk in batch:
                result = await self._extract_from_chunk(chunk)
                results.append(result)
        merged = self._deduplicate(results)
        return merged

    async def extract_single(self, text: str, chunk_id: str = "") -> ExtractionResult:
        """从单段文本中抽取知识"""
        return await self._extract_from_text(text, chunk_id)

    # ── core extraction ──────────────────────────────────────

    async def _extract_from_chunk(self, chunk: DocumentChunk) -> ExtractionResult:
        return await self._extract_from_text(chunk.content, chunk.chunk_id)

    async def _extract_from_text(self, text: str, source_id: str) -> ExtractionResult:
        """
        单次抽取，带重试。

        为什么必须重试：一次 159 chunk 的入库会发 159 次 LLM 请求，
        实测 DeepSeek 偶发 `APIConnectionError`（对端提前关闭连接）。
        没有重试时，**任意一次网络抖动都会让整轮入库失败**，
        已完成的 158 次调用全部作废。

        重试策略：指数退避（1s, 2s, 4s），最多 settings.max_retries 次。
        全部失败才抛出，让上层（worker）决定是否重试整篇文档。
        """
        messages = [
            SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
            HumanMessage(content=f"请从以下文本中抽取知识：\n\n{text}"),
        ]

        last_err: Exception | None = None
        # 抽取是长任务（159 次调用），需要比常规请求更宽的容错：
        # 取 max_retries 与 5 的较大值，退避 1/2/4/8s。
        attempts = max(5, int(settings.max_retries))
        for attempt in range(attempts):
            try:
                resp = await self.llm.ainvoke(messages)
                return self._parse_response(resp.content, source_id)
            except Exception as e:
                last_err = e
                if attempt < attempts - 1:
                    delay = 2 ** attempt
                    logger.warning(
                        "知识抽取失败，%.0fs 后重试（%d/%d）: %s",
                        delay, attempt + 1, attempts, type(e).__name__,
                    )
                    await asyncio.sleep(delay)

        logger.error("知识抽取重试耗尽，放弃该块: %s", type(last_err).__name__)
        raise last_err  # type: ignore[misc]

    def _parse_response(self, raw: str, source_id: str) -> ExtractionResult:
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
                cleaned = cleaned.rsplit("```", 1)[0]
            data = json.loads(cleaned)
        except (json.JSONDecodeError, IndexError):
            return ExtractionResult(entities=[], relations=[], events=[], source_chunk_id=source_id)

        entities = [
            Entity(
                name=e.get("name", ""),
                type=e.get("type", "Concept"),
                description=e.get("description", ""),
            )
            for e in data.get("entities", [])
            if e.get("name")
        ]
        relations = [
            Relation(
                head=r.get("head", ""),
                relation=r.get("relation", "related_to"),
                tail=r.get("tail", ""),
                confidence=float(r.get("confidence", 0.5)),
            )
            for r in data.get("relations", [])
            if r.get("head") and r.get("tail")
        ]
        events = [
            KnowledgeEvent(
                trigger=ev.get("trigger", ""),
                type=ev.get("type", ""),
                participants=ev.get("participants", []),
            )
            for ev in data.get("events", [])
        ]
        return ExtractionResult(
            entities=entities,
            relations=relations,
            events=events,
            source_chunk_id=source_id,
        )

    # ── deduplication & entity resolution ────────────────────

    @staticmethod
    def _deduplicate(results: list[ExtractionResult]) -> list[ExtractionResult]:
        """
        跨 chunk 去重: 同名同类型实体合并，关系去重
        """
        seen_entities: dict[str, Entity] = {}
        seen_relations: set[tuple[str, str, str]] = set()
        deduped: list[ExtractionResult] = []

        for result in results:
            unique_entities: list[Entity] = []
            for ent in result.entities:
                key = f"{ent.name}::{ent.type}"
                if key not in seen_entities:
                    seen_entities[key] = ent
                    unique_entities.append(ent)

            unique_relations: list[Relation] = []
            for rel in result.relations:
                key = (rel.head, rel.relation, rel.tail)
                if key not in seen_relations:
                    seen_relations.add(key)
                    unique_relations.append(rel)

            deduped.append(ExtractionResult(
                entities=unique_entities,
                relations=unique_relations,
                events=result.events,
                source_chunk_id=result.source_chunk_id,
            ))
        return deduped
