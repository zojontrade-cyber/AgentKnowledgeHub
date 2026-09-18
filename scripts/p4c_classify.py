"""
P4-C 审计结论：对 15 条漏抽事实做根因分类。

分类依据（基于原文 vs 抽取结果的逐条对照）
------------------------------------------
  P  prompt 问题   —— prompt 未要求抽取该类信息
  S  schema 问题   —— 三元组结构无法表达（如"150%"不是实体）
  M  模型问题     —— 文本明确，模型未抽
  J  judge 误判   —— 实际抽到，判定错

自动判定 + 人工复核标签。
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent

# ── 人工标注（基于上一步逐条审计的原文/抽取结果对照）──────
# 判定标准：
#   若抽取结果里出现了**该事实的核心词但缺数值** -> S（schema：数值无处安放）
#   若抽取结果完全无关该主题 -> M（模型漏抽）
#   若关键词+数值都在，只是表述不同 -> J（judge 误判）
LABELS = {
    ("外勤要提前报备吗？", "外勤前登记地点、事由和预计返回时间"): (
        "M", "抽取结果只到「外勤管理(Concept)员工外出工作的管理」层级，"
             "未抽出「登记地点/事由/预计返回时间」三个动作要素"
    ),
    ("年假要提前多久申请？", "提前 3 个工作日"): (
        "S", "抽到「申请与折算规则」概念，但「提前3个工作日」这一数值约束无处安放"
    ),
    ("加班费怎么算？", "工作日 150%"): (
        "S", "抽到「工作日延长工作时间」与「小时工资」，但 150% 未落成任何节点/属性"
    ),
    ("加班费怎么算？", "周末 200%"): (
        "S", "抽到「周末加班」「加班费」，200% 未落成任何节点/属性"
    ),
    ("加班费怎么算？", "法定节假日 300%"): (
        "S", "抽到「法定节假日加班」「加班费」，300% 未落成任何节点/属性"
    ),
    ("加班要提前申请吗？", "下班前 2 小时提交申请"): (
        "M", "该 chunk 只抽到「申请(Concept)加班流程中的申请环节」，"
             "完全未抽「下班前2小时」；同 chunk 的「提前1个工作日」也漏了"
    ),
    ("用印要走什么流程？", "注明文件名称、份数、用途和审批人"): (
        "S", "抽到「刻制/保管」等环节概念，但「申请单应注明哪些字段」"
             "这类**结构化约束**无处安放"
    ),
    ("劳动合同一般签几年？", "首次合同期限一般为 3 年"): (
        "S", "抽到「劳动合同」实体与关系，但「3 年」这一期限数值无字段可存"
    ),
    ("工资几号发？", "每月 10 日"): (
        "S", "只抽到「员工月薪(Concept)」，连「发放日」概念都没有，"
             "「每月10日」这个数值完全无处安放"
    ),
    ("发生安全事件怎么处置？", "先止损、后取证、再恢复"): (
        "M", "抽到事件与处置相关概念，但**有序步骤序列**未抽出 —— "
             "三元组是二元关系，表达不了 A→B→C 的顺序"
    ),
    ("分支怎么管理？", "main 保持可发布"): (
        "M", "该 chunk 只抽到分支相关上位概念，未抽「main 分支的状态约束」"
    ),
    ("分支怎么管理？", "禁止直接向 main 提交"): (
        "M", "「禁止」类否定规则未被抽出；同 chunk 的其它约束也未抽到"
    ),
    ("会议室预订后能取消吗？", "会议取消后应在系统中释放预订"): (
        "M", "未抽到「取消后需释放预订」这一动作要求"
    ),
    ("会议室预订后能取消吗？", "连续 3 次爽约限制预订 1 周"): (
        "S", "含「3 次」「1 周」两个数值构成的**惩罚规则**，无处安放"
    ),
    ("工作时间能处理私事吗？", "不从事与工作无关且影响效率的活动"): (
        "M", "「禁止从事无关活动」这类**行为禁令**未被抽出"
    ),
}


def classify():
    fr = json.loads((ROOT / "data" / "eval" / "p4b_fact_recall.json").read_text(encoding="utf-8"))
    missing = fr["A_1chunk"]["missing"]
    return missing


missing = classify()
print("=" * 88)
print(f"A 臂（chunk 级基线）漏抽事实根因分类  —— 共 {len(missing)} 条")
print("=" * 88)
print()

counts = Counter()
labeled = []
for q, fact, reason in missing:
    lab, why = LABELS.get((q, fact), ("?", "未标注"))
    counts[lab] += 1
    labeled.append({"q": q, "fact": fact, "root": lab, "why": why})

order = {"S": 0, "M": 1, "P": 2, "J": 3, "?": 4}
for item in sorted(labeled, key=lambda x: order.get(x["root"], 9)):
    print(f"[{item['root']}] {item['q']}")
    print(f"     事实: {item['fact']}")
    print(f"     根因: {item['why']}")
    print()

print("=" * 88)
print("分类汇总")
print("=" * 88)
names = {
    "S": "schema 问题（数值/约束无处安放）",
    "M": "模型漏抽（文本明确但没抽）",
    "P": "prompt 问题（未要求该类信息）",
    "J": "judge 误判（实际抽到）",
}
for k in ("S", "M", "P", "J"):
    if counts.get(k):
        print(f"  {k}  {names[k]:<34} {counts[k]}")

print()
print("=" * 88)
print("结论")
print("=" * 88)
print("""
  主导根因是 **S（schema）**：抽取器确实读到了这些句子，
  也抽出了相关概念（「工作日延长工作时间」「周末加班」「法定节假日加班」
  「小时工资」「加班费」），但 **150% / 200% / 300% 这些数值
  没有任何地方可以存放** —— 因为当前的 schema 只有：
      Entity(name, type, description)
      Relation(head, relation, tail)
  数值既不是实体，也不是关系的端点。

  次因是 **M（模型漏抽）**：整句条件（"下班前2小时"）被跳过，
  只抽出了上位概念「申请」。

  这解释了 P3.2 为什么 graph 对 QA 无贡献：
  QA 问的是「加班费怎么算」，答案是「150%/200%/300%」；
  而关系型表示是「工作日加班 -[related_to]-> 加班费」——
  **有关系、无取值**。检索到这些边也答不出问题。
""")

out = ROOT / "data" / "eval" / "p4c_audit.json"
out.write_text(json.dumps({
    "total_missing": len(missing),
    "counts": dict(counts),
    "items": labeled,
}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"写入 {out}")
