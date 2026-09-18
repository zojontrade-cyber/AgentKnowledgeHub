# AgentKnowledgeHub 架构冻结报告

**日期**：2026-09
**版本**：v2（检索重构 + 抽取能力澄清）
**状态**：架构冻结，QA 达 100% 正确率

---

## 一、执行摘要

### 最终系统链路

```
用户问题
   │
   ├─ 意图分类（function calling + enum）
   ├─ 查询改写（1-3 个检索查询）
   │
   ▼
BM25 稀疏检索（主召回，129 可召回章节）
   │
   ├─ 多查询合并（按 chunk 去重，保留 hit_count）
   ├─ small-to-big 展开（chunk 定位于 llm_context）
   │
   ▼
top-20 候选
   │
   ▼
Cross-Encoder 精排（BAAI/bge-reranker-v2-m3）
   │
   ▼
top-8 上下文
   │
   ├──→ llm_context（父文档全文）→ LLM 生成
   └──→ content（命中章节）      → 用户引用

旁路（已关闭）：
```

### 核心指标

| 指标 | 数值 | 说明 |
|---|---|---|
| **Answer correctness** | **100%**（36/36）| LLM-as-Judge，exact 组 |
| Answer completeness | 4.67/5 | |
| Citation accuracy | **100%** | 引用命中 gold 章节 |
| 检索失败 | **0/36** | gold 章节均进入引用 |
| Chunk Recall@5 | 100% | 修正后基准 |
| Chunk Top-1 | 91.7% | 含 reranker |
| p50 延迟 | 3.3s | 主要为 LLM 生成 |

---

## 二、本阶段关键修正

### 2.1 检索：稠密向量 → BM25（决定性）

**问题**：用户报告"入库后找不到信息"。

**实测**（159 章节 / 78 题文档级 / 55 题 chunk 级，无 reranker）：

| 方法 | Doc R@5 | Chunk R@5 | Chunk MRR |
|---|---|---|---|
| 稠密（Qwen3-Embedding-8B）| 41.0% | **23.6%** | 0.131 |
| **BM25** | **93.6%** | **89.1%** | **0.637** |

**根因**：制度语料含大量判别性低频词项（请假/年假/保密/印章/报销）。
BM25 视其为高 IDF 强信号；稠密模型在 4096 维空间将其稀释，
所有章节分数挤压在 0.44-0.60 且无区分度。

**RRF 融合被证伪**：等权融合 50.9%、权重 0.3:0.7 时 70.9%，
**均低于纯 BM25 的 89.1%** —— 稠密一路给正确答案投 0 票却给噪声投票。

### 2.2 引用契约修复（数据契约错误）

**问题**：`_expand_small_to_big` 把 `content` 替换为父文档全文，
导致每个引用的开头恒为"## 1. 文档说明"，用户无法核对答案出处。

**修复**：三层分离

| 字段 | 用途 | 消费者 |
|---|---|---|
| `content` | 命中的**章节** | 排序 + **用户引用** |
| `llm_context` | 展开的**父文档** | 仅 LLM 生成 |

**效果**：引用中 `section=文档说明` 比例 **100% → 0%**，
端到端引用准确率 **44.4% → 100%**。

### 2.3 多查询合并去重键错误

**问题**：`_expand_small_to_big` 按 `parent_id` 去重，而
`parent_id` 形如 `<doc_id>#doc` 是**文档级**的 —— 同一文档的
4-6 个章节共享同一个 id，导致**每个文档只保留一个章节**，
真正的答案章节被静默丢弃。

**证据**：
```
Q: 出差回来多久内要报销？
   vs.search()        → 5 条，gold 在 rank 2      ✅
   _vector_retrieve() → 2 条，gold 消失            ❌
```

**修复**：去重键改为 `doc_id#chunk_index`，并保留 `hit_count`。

### 2.4 文档说明块排除

**问题**：「1. 文档说明」类块信息量低、套话重，
在稠密与 BM25 下都系统性抢占前排。

**修复**：`classify_section()` 标记 `doc_overview`，
`searchable=0` 退出第一阶段召回，**但保留在 `parent_content` 中**
供 small-to-big 展开（回答仍可引用制度名称/适用范围）。

**效果**：排除 30 个块（精确等于文档数），
Chunk Top-1 **47.3% → 61.8%**，MRR **0.637 → 0.739**。

### 2.5 评估口径修正（影响最大）

**问题**：用字符串匹配判定答案对错，得到 66.7% 准确率，
据此误判为"系统需要救 RAG"。

**根因**：gold 是**整个章节全文**（含整张表），而正确答案只需
覆盖问题所问的那一项。字符串匹配要求答案含 gold 里所有数字，
把"保密级别分 4 级：公开/内部/机密/绝密"这类完全正确的答案判错。

**修正**：改用 LLM-as-Judge。**准确率 66.7% → 100%。**

**工程规则（本次建立）**：
> 字符串匹配只能作为 debug signal，**不得作为 QA quality gate**。

### 2.6 实体消歧统一（基础设施修复）

**问题**：30 篇制度文档经 `knowledge_update_agent` 入库，
走的是 `upsert_entity`（无消歧），而 ingest 流水线走
`upsert_entity_canonical`。两条路径并存。

**证据**：794 个 Entity 中只有 10 个带 `canonical_id`，
且全部来自一个测试文件。

**修复**：
- `upsert_entity` 内部转调 `upsert_entity_canonical`（统一入口）
- 新增 `entity_type_policy`（类型归一化 + 兼容性判定，18 单测）
- 关系类型**全部保留**（PART_OF 192 / USES 46 / DEPENDS_ON 38 等 8 种）

**定位**：改善实体一致性，
**对当前 QA 检索收益无显著影响**（邻居结构 416/418 不变）。

问题意图（三个不同的年假问题返回完全相同的四条上下文）。

**处置**：`graph_qa_enabled=False`（QA 主路径关闭），
`graph_search_enabled=True`（保留供调试/admin/未来 agent tool）。

### 2.8 抽取成本实验（P4-B，负结果）

| 方案 | 调用 | token 节省 | Fact Recall |
|---|---|---|---|
| A 1 chunk/call | 159 | — | 76.4% |
| B 2 chunks/call | 89 | 38% | 74.3%（−2.1pp）|
| C 3 chunks/call | 60 | 54% | 68.6%（−7.9pp）|
| （文档级）| 30 | 70% | 实体 −52% |

**结论：不采用批量优化。** 三档均低于门槛（≥ baseline −2pp），
且与文档级实验同向：**批量越大，信息损失越多**。

### 2.9 抽取漏抽根因（P4-C，重要澄清）

对 15 条漏抽事实做根因分类：

| 根因 | 数量 |
|---|---|
| **schema 问题**（数值/约束无处安放）| **8** |
| 模型漏抽 | 7 |
| prompt 问题 | 0 |
| judge 误判 | 0 |

**证据**：
```
原文: 工作日延长工作时间按本人小时工资的 150% 支付，周末按 200%，法定节假日按 300%
抽取: 工作日延长工作时间 -[related_to]-> 小时工资
      （概念/关系全对，150%/200%/300% 全部丢失）
```

现有 schema `Entity(name,type,description)` + `Relation(head,rel,tail)`
中，**数值既不是实体也不是关系端点，结构上无处安放**。

**验证**：新增 Fact 层（6 类事实 + quote），
**8 条 S 类事实恢复 7 条**，达标。

**但**：Fact 对 QA **无提升**（A/B 均 5.00/5）——
因为 **QA 走 chunk 检索，而 chunk 本身含完整原文**。
抽取丢失的数值在 chunk 里完好无损。

---

## 三、架构决策记录

| 决策 | 结论 | 依据 |
|---|---|---|
| 主召回通道 | **BM25** | Chunk R@5 89.1% vs 稠密 23.6% |
| RRF 融合 | **不采用** | 低于纯 BM25（50.9% / 70.9% vs 89.1%）|
| Reranker | **默认开启** | Top-1 +13.9pp，MRR +0.083 |
| rerank_candidates | **20** | 20 与 50 指标相同，但延迟 548ms vs 884ms |
| 两阶段（文档→章节）| **不采用** | Chunk@1 76.3% → 42.1%（−34pp）|
| 文档说明块 | **排除但保留父块** | Top-1 +14.5pp |
| 抽取批量 | **保持 chunk 级** | 批量越大信息损失越多 |
| Fact 层 | **保留但非 QA 路径** | QA 无提升；结构化查询有价值 |
| uvicorn reload | **默认关闭** | reload 导致 vector_store 半初始化 |

---

## 四、已知限制与风险

### 4.1 未解决

| 项目 | 说明 |
|---|---|
| Answer completeness | 4.67/5；1/36 题答案不完整（远程办公申请漏审批层级）|
| `documents` 计数全为 0 | 32/32 文档的 `chunks_count` 等未回填（数据本身正常）|
| `source_node_count` 语义 | 实为"合并前节点数"，非"文档出现次数"（已标注）|
| Outbox 从未使用 | 失败重放路径无实战验证 |
| ACL fail-open | `source_path` 匹配失败时默认放行（记录了 warning）|
| 104 个孤立 Entity | 无任何关系（13.7%）|
| NumPy 1.x/2.x 冲突 | Anaconda 与 user site-packages 各一份，影响 pandas 系依赖 |

### 4.2 环境约束

- 无 Docker；嵌入式 Chroma + SQLite，**不需要任何外部服务**
  - DeepSeek API 偶发 `APIConnectionError`（已加重试 + 断点续传）
- 8080 上的旧进程若为提权启动，无法从普通 shell 终止；用 `stop-api.bat`，
  必要时以管理员身份 `taskkill /PID <pid> /F`（见 [`runbook.md`](./runbook.md)）

---

## 五、产物清单

### 生产代码

| 文件 | 变更 |
|---|---|
| `python/services/bm25.py` | **新增**：BM25 索引（bigram+单字，零依赖）|
| `python/services/vector_store.py` | `search()` 按 `retrieval_mode` 路由；BM25 索引懒建 |
| `python/services/entity_canonicalizer.py` | 新增 `get_canonicalizer()` 单例 |
| `python/services/knowledge_graph.py` | `upsert_entity` 转调 canonical |
| `python/services/entity_type_policy.py` | **新增**：类型归一化 + 兼容性（18 单测）|
| `python/services/reranker.py` | 三重启用条件 + `fail_streak` |
| `python/agents/fact_extract_agent.py` | **新增**：Fact 抽取（6 类 + quote）|
| `python/agents/qa_agent.py` | `RetrievedContext` 三层契约；去重键修复 |
| `python/agents/knowledge_extract_agent.py` | 指数退避重试 |
| `python/agents/doc_parser_agent.py` | `classify_section()`；`searchable`/`section_type` |
| `python/config/settings.py` | `retrieval_mode` / `rerank_enabled` / `graph_qa_enabled` / `api_reload` |
| `python/api/main.py` | 引用契约；health 暴露 reranker 状态 |

### 评测与报告

| 文件 | 说明 |
|---|---|
| `data/eval/bench_v3.jsonl` | 50 题（36 exact / 6 multi / 8 unavailable）|
| `data/eval/p4b_baseline_judge.json` | QA 基线（100%）|
| `data/eval/p4c_audit.json` | 15 条漏抽根因分类 |
| `data/eval/p4c_facts.json` | 52 条结构化事实 |
| `docs/retrieval-bm25-migration.md` | BM25 迁移报告 |
| `docs/p22-merge-fix-report.md` | 多查询合并修复报告 |
| `docs/p4-extraction-cost.md` | 抽取成本报告 |
| `docs/architecture-freeze.md` | **本文件** |

---

## 六、测试状态

```
105 passed
  ├─ 87 原有（Phase 0 加固 / QA 重构 / 持久化 ACL）
  └─ 18 新增（entity_type_policy）
```

---

## 七、后续建议（按价值排序）

| 优先级 | 项目 | 理由 |
|---|---|---|
| 1 | **工程健壮性** | QA 已 100%；真风险在并发/监控/一致性 |
| 2 | 重启 8080 使改动生效 | 当前跑的是旧代码 |
| 3 | 回填 `documents` 计数 | 影响可观测性 |
| 4 | Answer completeness | 仅 1/36，收益小 |
| 5 | Fact 结构化查询能力 | 已验证有价值，但非当前瓶颈 |
