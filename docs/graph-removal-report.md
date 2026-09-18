# 知识图谱移除报告

**日期**：2026-09
**范围**：L1 数据 + L2 代码 + L3 Neo4j 安装，全部移除
**状态**：已完成并通过验证

---

## 一、移除依据（实测数据，非主观判断）

图谱在 QA 链路中的实测贡献（42 题，`data/eval/p32a_final.json`）：

| 指标 | Pipeline（无图谱） | Pipeline + 图谱 |
|---|---|---|
| Answer Correctness | **97.2%** | 94.4% |
| Citation Correctness | 100% | 100% |
| **Graph 被引用率** | — | **仅 7.1%** |

**图谱在 52.4% 的题上"参与"了检索，但只在 7.1% 的题上被答案引用** ——
返回 ≠ 贡献。

根因是**用途错配**：图谱返回「相关实体」而非「答案」。

```
问"年假有多少天"（答案：5/10/15 天）
图谱给：年假 --[USES]--> OA系统
        年假 --[PART_OF]--> 请假流程
        年假 --[RELATED_TO]--> 员工
```

同一实体的 1 跳邻居固定不变，**无法区分不同问题意图** ——
三个不同的年假问题返回完全相同的四条上下文。

ReAct 模式（让 LLM 自主选工具）二次验证：204 次工具调用中图谱工具仅占
10.8%，四项指标无显著改善（`docs/react-ab-report.md`）。

---

## 二、删除内容

### L1 数据

| 项 | 数量 |
|---|---|
| Entity 节点 | 1,615 |
| Entity 关系 | 1,425（14 种类型）|
| ConceptNet Concept 节点 | 75,603 |
| 数据目录体积 | 118.5 MB |

### L2 代码（38 个文件）

| 类别 | 文件 |
|---|---|
| 核心服务 | `knowledge_graph.py`(330行) / `graph_queries.py`(138行) / `graph_rag.py`(188行) |
| 实体消歧 | `entity_canonicalizer.py`(222行) / `entity_type_policy.py`(126行) |
| 测试 | `test_entity_type_policy.py`(18 单测) + 3 个 check 脚本 |
| 分析脚本 | 28 个（合并/迁移/验证/分析）|

全部移至 **`_deprecated/`**（未物理删除，可恢复）。

### 生产代码改动（18 个文件）

| 文件 | 改动 |
|---|---|
| `config/settings.py` | 删除 `neo4j_*` / `graph_qa_enabled` / `graph_search_enabled`；删除 Neo4j 弱口令校验 |
| `orchestrator/graph.py` | 删除 `persist_graph` 节点、`get_canonicalizer` 单例；`replay_document` 简化；新增 `from config import settings` |
| `agents/qa_agent.py` | 删除 `_link_entities` / `_run_template` / `_subgraph_retrieve` / `_path_retrieve` / `_overview_retrieve` / `_format_graph_record`（**121 行**）；`__init__` 去掉 `knowledge_graph` |
| `agents/react_qa_agent.py` | 删除 3 个图工具（`search_graph` / `find_path` / `list_overview`）与 `_allow_graph` |
| `agents/knowledge_update_agent.py` | 删除图谱写入与删除分支；`__init__` 去掉 `knowledge_graph` |
| `services/ingest_worker.py` | `__init__` 去掉 `knowledge_graph`；`replay_document` 调用简化 |
| `api/main.py` | 删除实例 / 初始化 / 关闭 / health 依赖 / stats 字段 / ui_routes.bind 参数 |
| `api/ui_routes.py` | 删除 `/graph` 端点（**134 行**）与 `_kg()`；overview 去掉图谱统计 |
| `api/static/index.html` | 删除图谱导航按钮、view-graph 区块（**55 行**）、台账/概览图谱行、vis-network script |
| `api/static/app.js` | 删除图谱 JS 模块（**280 行**）、VIEW_TITLE/config/ledger 中的图谱项 |
| `services/__init__.py` | 文档字符串更新 |

### L3 安装

```
E:\neo4j  →  已删除（371 文件 / 292.6 MB）
```

同时删除 `api/static/vendor/vis-network.min.js`（**689 KB**，仅图谱可视化使用）。

**合计释放磁盘：约 293 MB（Neo4j）+ 689 KB（前端库）**

---

## 三、保留的可重建资产

| 资产 | 位置 | 说明 |
|---|---|---|
| **Entity 最终快照** | `data/backup/graph-final-*.json`（1001 KB）| 含 canonical 合并结果，1,615 节点 + 1,425 关系，**关系类型完整** |
| Entity 合并前快照 | `data/backup/graph-entity-*.json`（2×421 KB）| 用于对比 |
| **ConceptNet 源数据** | `data/conceptnet_zh/nodes.csv`, `relations.csv` | 可直接重新导入 |
| ConceptNet 导入脚本 | `data/backup/conceptnet-rebuild/`（4 个）| |
| 全部图谱代码 | `_deprecated/` | 含 README 说明重建步骤 |

**快照字段完整性校验**：`canonical_id` 1615/1615、`aliases` 1615/1615、
`merge_reason` 34（等于合并组数）。

---

## 四、验证结果

### 单元测试

```
43 passed
```

（原 105 → 43：移除 18 个 entity_type_policy 单测 + 图谱守卫/模板测试；
`test_phase0_hardening.py` 移除 4 个关系类型安全测试；
`test_qa_refactor.py` 移除 8 个 Cypher 守卫测试，新增 2 个去重/排序测试）

### API 端到端（**未启动 Neo4j**）

| 端点 | 结果 |
|---|---|
| `/api/health` | **HTTP 200** ✅ —— `{"vector_store":"ok","reranker":"ok"}`，**无 knowledge_graph 依赖** |
| `/api/ui/overview` | HTTP 200，无 `knowledge_graph` 字段 ✅ |
| `/api/ui/documents` | HTTP 200，total=57，`display_name` 正常 ✅ |
| `/api/ui/graph` | **HTTP 404** ✅（已移除）|
| `/` 首页 | HTTP 200，无图谱导航/区块/vis-network ✅ |

### 问答功能

```
Q: 标准工作时间是几点到几点？
A: 标准工作时间为周一至周五 9:00 至 18:00…  [来源: YQ-INST-001-员工考勤管理制度.md]

Q: 加班费怎么算？
A: 工作日 150% / 周末 200% / 法定节假日 300%…

Q: 公司有没有规定员工可以无限期远程办公？
A: 没有。根据《远程办公管理办法》…
```

**三项均正确，且系统已完全不依赖 Neo4j。**

### ReAct 工具集

```
['search_docs']   ← 从 4 个减为 1 个
```

---

## 五、移除的代价（如实记录）

| 项目 | 影响 |
|---|---|
| **实体消歧能力** | `esntity_canonicalizer` + `entity_type_policy`（348 行 + 18 单测）成为死代码。那次修复（759 节点归一、类型噪声治理）的价值一并归档 |
| **多跳关系查询** | 系统不再具备"A 与 B 有何关联"的显式推理能力 |
| **图谱可视化** | UI 图谱页移除 |
| **ReAct 实验结论** | 原"通道选择"实验的基础消失；ReAct 现在只有 1 个工具，退化为简单循环 |

**但换个角度**：这些能力实测对 QA 零贡献，维护成本却是真实的
（Neo4j 需常驻进程、抽取需 159 次 LLM 调用、每次入库多一个失败点）。

---

## 六、已知遗留

| 项目 | 说明 |
|---|---|
| `knowledge_extract_agent.py` | 实体/关系抽取代码仍在，但**已无消费者**（图谱是它唯一用途）。保留未删 —— 若将来做 Fact 层可复用 |
| `fact_extract_agent.py` | 事实抽取（6 类 + quote），当前未接入检索 |
| `data/conceptnet_zh/` | 780 MB 源数据仍在磁盘；若确定不再需要可删 |
| `_deprecated/` | 38 个文件保留在项目内；若确定不再恢复可删 |
| `GRAPH_DONE` 状态 | `DocStatus` 中保留该枚举值（历史数据可能仍带此状态）|

---

## 七、复现验证

```bash
# 单元测试
cd python && python -m pytest tests/ -q

# 端到端冒烟（无需 Neo4j）
python scripts/smoke_after_graph_removal.py

# 启动服务（无需先启 Neo4j）
python -m api.main          # 或双击 start-api.bat
```

`start-api.bat` 中「Neo4j must be running first」的提示已过时，应更新。
