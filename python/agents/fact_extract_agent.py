"""
答案性事实抽取（Fact Extraction）—— 补齐 Entity/Relation 表达不了的结构

为什么要独立于 Entity/Relation
------------------------------
P4-C 审计（data/eval/p4c_audit.json）对 15 条漏抽事实做了根因分类：

    S  schema 问题（数值/约束无处安放）  8 条  53%
    M  模型漏抽（文本明确但没抽）        7 条  47%

S 类的证据非常明确：

    原文: 工作日延长工作时间按本人小时工资的 150% 支付，周末按 200%，
          法定节假日按 300%

    抽取结果:
      工作日延长工作时间（Concept）在工作日延长工作时间的情况
      工作日延长工作时间 -[related_to]-> 小时工资
      周末加班（Event）在周末进行的加班
      周末加班 -[related_to]-> 加班费
      法定节假日加班 -[related_to]-> 加班费

    概念、关系全抽对了，**唯独 150% / 200% / 300% 一个都没留下**。

原因是现有 schema 只有：

    Entity(name, type, description)
    Relation(head, relation, tail)

**数值既不是实体，也不是关系的端点** —— 它在结构上无处安放。

（早期把数值硬塞成实体名的做法，就是
 模型在缺少 schema 时的自发变通：把数值硬塞成实体名。）

这同时解释了 P3.2 为什么 graph 对 QA 无贡献：

    QA 问: 加班费怎么算？   答案: 150% / 200% / 300%
    关系型表示: 工作日加班 -[related_to]-> 加班费   <- 有关系、无取值

因此本模块**独立**抽取事实层，不合并进 entity/relation 流程 ——
便于归因：Fact 效果不好时能确定是 Fact 抽取器的问题，
而不是把 entity/relation 的 prompt 也搞坏了。
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from config import settings
from services.llm_factory import LazyLLM

logger = logging.getLogger(__name__)

FACT_SYSTEM_PROMPT = """\
你是一个**制度事实抽取引擎**。你的任务是抽取文档中的「答案性事实」——
即用户提问时真正需要的那条信息。

特别注意：**数值、条件、时限是重点**。制度文档的核心价值就在这些地方，
但它们常常在普通知识抽取中被漏掉。

需要抽取的事实类型（fact_type）：
1. rate       —— 比率/金额标准（如"工作日加班 150%"、"住宿费 500 元/晚"）
2. deadline   —— 时限（如"出差结束后 10 个工作日内报销"、"提前 3 个工作日申请"）
3. threshold  —— 阈值/分档（如"满 10 年不满 20 年 10 天"、"连续 3 次爽约"）
4. requirement—— 要求/条件（如"须提交营业执照、资质、案例"）
5. prohibition—— 禁止事项（如"禁止直接向 main 提交"、"空白文件严禁盖章"）
6. sequence   —— 有序步骤（如"先止损、后取证、再恢复"）

对每条事实，抽取以下字段：
- fact_type : 上面六类之一
- condition : 适用条件（如"工作日加班"、"一线城市"、"满10年不满20年"）
- value     : 具体取值（如"150"、"10"、"500"、"先止损后取证再恢复"）
- unit      : 单位（如"%"、"个工作日"、"元/晚"、"天"；无单位留空）
- quote     : 原文中的依据句子（**必须逐字摘录，不得改写**）

严格返回 JSON：
{
  "facts": [
    {"fact_type":"rate","condition":"工作日加班","value":"150","unit":"%","quote":"工作日延长工作时间按本人小时工资的 150% 支付"}
  ]
}

要求：
- **不要遗漏任何数值**（百分比、金额、天数、次数、小时）
- **不要遗漏任何时限**（几个工作日、几小时内、提前多久）
- **不要遗漏禁止性规定**
- 一个句子含多个事实时，拆成多条
- 只返回 JSON，不要其他文字
"""


@dataclass
class Fact:
    """答案性事实"""

    fact_type: str          # rate / deadline / threshold / requirement / prohibition / sequence
    condition: str = ""     # 适用条件
    value: str = ""         # 取值
    unit: str = ""          # 单位
    quote: str = ""         # 原文依据
    source_id: str = ""     # 来源 chunk

    @property
    def display(self) -> str:
        """可读展示（供检索/引用用）"""
        parts = []
        if self.condition:
            parts.append(self.condition)
        if self.value:
            v = f"{self.value}{self.unit}" if self.unit else self.value
            parts.append(v)
        return "：".join(parts) if len(parts) == 2 else " ".join(parts) or self.quote[:60]


@dataclass
class FactResult:
    facts: list[Fact] = field(default_factory=list)
    source_chunk_id: str = ""


class FactExtractAgent:
    """
    事实抽取 Agent —— 与 KnowledgeExtractAgent **并行**运行，不合并。

    独立的原因（用户要求）：便于归因。若 Fact 效果不好，
    能确定是 Fact 抽取器的问题，而不是污染了 entity/relation 的 prompt。
    """

    def __init__(self) -> None:
        self.llm = LazyLLM(temperature=0)

    async def extract(self, text: str, source_id: str = "") -> FactResult:
        """从一段文本抽取事实，带重试"""
        messages = [
            SystemMessage(content=FACT_SYSTEM_PROMPT),
            HumanMessage(content=f"请从以下制度文本中抽取答案性事实：\n\n{text}"),
        ]

        last_err: Exception | None = None
        attempts = max(5, int(settings.max_retries))
        for attempt in range(attempts):
            try:
                resp = await self.llm.ainvoke(messages)
                return self._parse(resp.content, source_id)
            except Exception as e:
                last_err = e
                if attempt < attempts - 1:
                    delay = 2 ** attempt
                    logger.warning(
                        "事实抽取失败，%.0fs 后重试（%d/%d）: %s",
                        delay, attempt + 1, attempts, type(e).__name__,
                    )
                    await asyncio.sleep(delay)

        logger.error("事实抽取重试耗尽: %s", type(last_err).__name__)
        raise last_err  # type: ignore[misc]

    def _parse(self, raw: str, source_id: str) -> FactResult:
        try:
            cleaned = (raw or "").strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
                cleaned = cleaned.rsplit("```", 1)[0]
            data = json.loads(cleaned)
        except (json.JSONDecodeError, IndexError):
            logger.warning("事实抽取响应无法解析，返回空")
            return FactResult(facts=[], source_chunk_id=source_id)

        items = data.get("facts")
        if not isinstance(items, list):
            return FactResult(facts=[], source_chunk_id=source_id)

        facts = []
        for it in items:
            if not isinstance(it, dict):
                continue
            facts.append(Fact(
                fact_type=(it.get("fact_type") or "").strip(),
                condition=(it.get("condition") or "").strip(),
                value=str(it.get("value") or "").strip(),
                unit=(it.get("unit") or "").strip(),
                quote=(it.get("quote") or "").strip(),
                source_id=source_id,
            ))
        return FactResult(facts=facts, source_chunk_id=source_id)
