"""
P4-C 原型：结构化事实查询（Fact-based Structured Query）

要证明的问题
------------
chunk 向量检索**结构上做不到**的事，Fact 层能做到：

  问题类型                         chunk 检索        Fact 查询
  ------------------------------  ---------------  -----------
  "加班费怎么算"                    ✅ 能做           ✅ 能做
  "哪些制度规定了时限"                ❌ 做不到（无聚合）  ✅ 能做
  "所有超过 30 天的时限有哪些"         ❌ 做不到（无数值筛选）✅ 能做
  "有哪些禁止性规定"                 ❌ 做不到（无类型筛选）✅ 能做
  "150% 出现在哪条规则里"            ⚠️ 弱（关键词命中）  ✅ 精确（数值索引）

本原型不走检索模型，直接对 Fact 做**结构化过滤**：
  - 按 fact_type 筛（rate / deadline / prohibition / threshold）
  - 按数值范围筛（如 unit=个工作日 且 value > 5）
  - 返回 Fact + quote（可回溯到原文）

数据源：data/eval/p4c_facts.json（2 篇文档，不扩全量）
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
FACTS = json.loads((ROOT / "data" / "eval" / "p4c_facts.json").read_text(encoding="utf-8"))

print(f"Fact 库: {len(FACTS)} 条，来自 {len({f['doc'] for f in FACTS})} 篇文档")
print()

TYPE_NAMES = {
    "rate": "比率/金额标准",
    "deadline": "时限",
    "threshold": "阈值/上限",
    "requirement": "要求/条件",
    "prohibition": "禁止事项",
    "sequence": "有序步骤",
}


def fmt(f: dict) -> str:
    unit = f["unit"]
    val = f"{f['value']}{unit}" if unit else f["value"]
    return f"{f['condition'] or '(无条件)'} → {val}"


# ══════════════════════════════════════════════════════════
# 四类"chunk 检索做不到"的查询
# ══════════════════════════════════════════════════════════
QUERIES = [
    {
        "q": "公司有哪些禁止性规定？",
        "why": "chunk 检索无法按「规则类型」聚合；向量检索没有'禁止'这个维度",
        "fn": lambda: [f for f in FACTS if f["fact_type"] == "prohibition"],
    },
    {
        "q": "有哪些时限要求？",
        "why": "chunk 检索要把所有章节翻一遍才能凑齐；Fact 层一次筛选完成",
        "fn": lambda: [f for f in FACTS if f["fact_type"] == "deadline"],
    },
    {
        "q": "有哪些金额/比率标准？",
        "why": "数值不是实体，向量检索无法'按数值类型'检索",
        "fn": lambda: [f for f in FACTS if f["fact_type"] == "rate"],
    },
    {
        "q": "超过 5 个工作日的时限有哪些？",
        "why": "**需要数值比较** —— 向量检索完全没有这个能力",
        "fn": lambda: [
            f for f in FACTS
            if f["fact_type"] == "deadline"
            and f["unit"] in ("个工作日", "工作日", "天")
            and (m := re.search(r"\d+", f["value"]))
            and int(m.group()) > 5
        ],
    },
    {
        "q": "加班相关的所有规则（跨类型）",
        "why": "按主题聚合跨 fact_type 的规则；chunk 检索会漏掉分散在不同章节的条目",
        "fn": lambda: [f for f in FACTS if "加班" in f["condition"] or "加班" in f["quote"]],
    },
]

for i, item in enumerate(QUERIES, 1):
    res = item["fn"]()
    print("=" * 88)
    print(f"Q{i}: {item['q']}")
    print(f"     为什么 chunk 检索做不到: {item['why']}")
    print("=" * 88)
    print(f"   命中 {len(res)} 条:")
    for f in res[:12]:
        print(f"     [{TYPE_NAMES.get(f['fact_type'], f['fact_type'])}] {fmt(f)}")
        print(f"        来源: {f['doc']} / {f['section']}")
        print(f"        原文: {f['quote'][:74]}")
    if len(res) > 12:
        print(f"     ... 还有 {len(res)-12} 条")
    print()

# ── 引用可追溯性演示 ────────────────────────────────────
print("=" * 88)
print("引用可追溯性：Answer → Fact → quote → Chunk → Document")
print("=" * 88)
sample = [f for f in FACTS if f["fact_type"] == "rate"][:2]
for f in sample:
    print(f"\n  回答需要的信息: {fmt(f)}")
    print(f"    ↓ Fact.quote（原文逐字）")
    print(f"    「{f['quote']}」")
    print(f"    ↓ 定位")
    print(f"    章节: {f['doc']} / {f['section']}")
    print(f"    ↓ 文档")
    print(f"    {f['doc']}.md")

# ── 统计 ────────────────────────────────────────────────
print()
print("=" * 88)
print("Fact 类型分布")
print("=" * 88)
cnt = defaultdict(int)
for f in FACTS:
    cnt[f["fact_type"]] += 1
for t, c in sorted(cnt.items(), key=lambda x: -x[1]):
    print(f"  {t:<14} {TYPE_NAMES.get(t, ''):<16} {c}")
