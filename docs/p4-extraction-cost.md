# P4 抽取成本优化 — 阶段报告

**日期**：2026-09
**状态**：进行中（fact recall 实验运行中）

---

## 一、最重要的修正：答案准确率是 100%，不是 66.7%

### 之前的错误

我用**字符串匹配**判定答案对错（"答案是否包含 gold 里的数字/片段"），
得到 exact 组 66.7% 的准确率，并据此认为"系统需要救 RAG"。

### 实际结果（LLM-as-Judge）

| 判定方式 | 正确率 |
|---|---|
| 字符串匹配（错） | 61.1% |
| **LLM judge（对）** | **100.0%（36/36，正确性均分 5.00/5）** |

**误差 39 个百分点。**

### 为什么字符串匹配必然失败

gold 是**整个章节全文**（可能含整张表），而正确答案只需覆盖
**问题所问的那一项**：

```
问题: 出差住宿费能报多少？
gold: 表格（住宿 500 / 交通 100 / 餐补 150 三列）
答案: "一线城市 500 元/晚，省会 400，其他 300"     ← 完全正确
旧判定: 要求答案同时含 100 和 150 -> 判为错
```

同类误判：
- 问"保密级别分几级" → 答"分为 4 级：公开、内部、机密、绝密" → 判错
- 问"招聘有哪些渠道" → 答与 gold 逐项相同 → 判错

### 工程规则（本次建立）

> **字符串匹配只能作为 debug signal，不得作为 QA quality gate。**

后续 benchmark 固定使用：
- Retrieval metrics（Recall@5 / MRR）
- Citation metrics
- Answer correctness（LLM judge）
- Answer completeness（LLM judge）

---

## 二、修正后的真实系统基线

| 模块 | 状态 |
|---|---|
| Retrieval | ✅ 检索失败 0/36 |
| Citation | ✅ 100% |
| Answer correctness | ✅ **100%**（LLM judge，36 题 exact）|
| Answer completeness | ✅ 4.67/5 |
| Graph QA | ❌ 无贡献（已默认关闭）|
| Extraction cost | ⏳ 待优化 |

### 方向修正

```
之前以为:  QA 66.7%  ->  需要救 RAG
实际情况:  QA 100%   ->  不要动 RAG，只优化成本
```

---

## 三、抽取成本结构（硬数据）

### 现状（chunk 级，159 次 LLM 调用）

```
system prompt        675 字符   ← 每次调用都要重发
chunk 正文（中位）    107 字符
prompt / 正文 比      6.2x
```

**86% 的 token 花在重复发送同一段 prompt 上。**

| 策略 | 调用数 | 正文 token | prompt token | 总 token | 节省 |
|---|---|---|---|---|---|
| A 1 chunk/call（现状）| 159 | 11,865 | 73,140 | 85,005 | — |
| B 2 chunks/call | 89 | 11,865 | 40,940 | 52,805 | **38%** |
| C 3 chunks/call | 60 | 11,865 | 27,600 | 39,465 | **54%** |
| （文档级，参考）| 30 | 12,019 | 13,800 | 25,819 | 70% |

耗时实测：A 378.7s / B 269.0s / C 215.6s。

### 文档级已被否决

实测（3 篇文档）：

```
实体: 159 -> 76  (-52%)
关系: 142 -> 64  (-55%)
```

丢失的是**数值型事实**（"10 个工作日"、"9:00-18:00"、"8 小时"），
而这些正是制度问答的答案本体。原因：一次处理 6 个章节时，
LLM 倾向于抽取"主题性"实体而非"细节性"实体。

---

## 四、Entity recall（可信但**不作决策依据**）

集合运算，不涉及语义匹配，因此数据可信：

| 方案 | Jaccard（相对 baseline） |
|---|---|
| A 1 chunk | 100.0% |
| B 2 chunks | 72.9% |
| C 3 chunks | 61.1% |

**但 P3.2 已证明 `Entity ↓ ⇏ QA ↓`**（实体层对 QA 无贡献），
所以实体流失本身不构成否决理由。

**注意区分**（这是本阶段最容易犯的逻辑跳跃）：

```
已证明: Entity 数量 ↓ 不一定导致 QA ↓
未证明: Extraction fact ↓ 不会导致 QA ↓     <- 这才是要测的
```

---

## 五、P3.2 结论回顾：实体层在 QA 上无贡献

| 指标 | A vector-only | B vector+graph |
|---|---|---|
| Answer（LLM judge）| **97.2%** | 94.4% |
| Citation | 100% | 100% |
| Graph 被引用 | — | **仅 7.1%** |

实体层在 52.4% 的题上"参与"了检索，但只在 7.1% 的题上被答案引用。
根因是**用途错配**：返回"相关实体"而非"答案"，且 1 跳邻居
无法区分问题意图（三个不同的年假问题返回完全相同的四条上下文）。

**处置**：`graph_qa_enabled=False`（QA 主路径关闭），
`graph_search_enabled=True`（保留供调试/admin/未来 agent tool）。

---

## 六、canonical 修复的定位

| 项目 | 结果 |
|---|---|
| Entity 节点 | 794 → 759 |
| canonical_id 覆盖 | 0 → **759/759** |
| 关系类型 | 8 种全部保留（PART_OF/USES/DEPENDS_ON 等未丢）|
| 邻居结构变化 | **416/418 不变**（仅"企业 IM"空格归一）|

**结论**：Entity resolution 修复改善的是**实体一致性**，
对当前 QA 检索收益**无显著影响**。这是基础设施修复，不是 recall 提升。

---

## 七、待完成

1. **Answer-bearing Fact Recall 实验**（运行中）
   - 三档 batch 的抽取结果，用 LLM 语义判定是否表达关键事实
   - 0/1/2 三档评分，Fact Recall = sum/2n
   - 选择标准：Fact Recall ≥ baseline − 2pp

2. **P5 Answer completeness**（暂缓）
   - 唯一真实的生成缺陷：远程办公申请题只答了部分条件
   - 仅 1/36，不打断 P4

---

## 八、产物

| 文件 | 说明 |
|---|---|
| `data/eval/p4b_baseline_judge.json` | 固定基线（batch=1，LLM judge，100%）|
| `data/eval/p4b_fact_recall.json` | 语义事实召回（运行中）|
| `scripts/p4b_fact_recall.py` | 本阶段核心实验 |
| `python/services/entity_type_policy.py` | 类型策略（18 单测）|
