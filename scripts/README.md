# scripts/ 说明

> 这里**不全是测试**。正式测试在 `python/tests/`（`python -m pytest tests/ -q`，121 项）。
> 本目录是**评测复现 / 端到端验证 / 运维工具**。

所有脚本都从自身位置推导项目根目录（`Path(__file__).resolve().parent.parent`），
克隆到任意路径都能跑。

---

## 1. 端到端验证（改代码后跑这些）

需要先启动服务（`start-api.bat`）。

| 脚本 | 验证内容 |
|---|---|
| `smoke_e2e.py` | 冒烟：应用能启动、编排图能构建、问答能跑通 |
| `verify_chat_metrics.py` | 真实 HTTP：计量字段、轮次递增、会话回填、actor 隔离（21 项） |
| `verify_chat_history.py` | 真实 HTTP：历史恢复、匿名单次不写、历史未进 prompt（16 项） |
| `verify_ui_metrics.py` | 无头 Chrome 走真实浏览器：readout 8 格、侧栏、刷新恢复（43 项） |
| `verify_ui.py` | 静态校验 HTML 结构与前端依赖的接口契约 |
| `acceptance_async.py` | 异步入库全链路：上传 → 轮询 → 一致性 |
| `acceptance_test.py` | 模拟用户真实操作流程 |
| `accept_retrieval.py` | 通过**生产代码路径**重跑检索，确认 BM25 生效 |
| `verify_citation_fix.py` | 引用契约：`sources[].content` 必须是命中章节 |
| `verify_three_layer.py` | 三层分离：embedding 输入 / 展示文本 / 父块 |
| `verify_worker_fix.py` | 入库 worker 队列不再死锁 |
| `verify_ingest.py` / `verify_new_docs_retrieval.py` / `verify_display_name.py` | 入库结果与检索可达性 |

> **本机注意**：HTTP 验证脚本用 `httpx.Client(trust_env=False)` —— 本机配了系统代理时，
> httpx 默认会把 `127.0.0.1` 也走代理而挂死。

## 2. 评测复现（产生 README 里的指标）

| 脚本 | 产出 |
|---|---|
| `p5_prompt_ab.py` | 生成 Prompt 结构 A/B（**含 LLM-as-Judge 四项指标**），预热后取最后一遍 |
| `p5_cache_hit.py` | token 与缓存命中实测（逐 stage / 逐题型），落盘 `data/eval/` |
| `compare_ab.py` | Pipeline vs ReAct 四项指标对比 |
| `p1_dense_vs_bm25.py` | 稠密 vs BM25（**这就是"23.6% → 89.1%"的来源**） |
| `p1_two_stage.py` / `p2_rerank.py` / `p2_rerank_p50.py` | 两阶段检索与精排实验 |
| `exp1/2/3_*.py` | embedding 输入形态 / 两阶段 / BM25 vs 稠密 vs RRF |
| `p4_extraction_cost.py` / `p4b_*` / `p4c_*` / `p4d_*` | 抽取成本与批处理实验（**负结果**） |
| `audit_bench_gold.py` / `audit_bench_multi.py` | 评测集 gold 标签审计 |

## 3. 造数据与工具

| 脚本 | 用途 |
|---|---|
| `build_bench_v4.py` | 生成 54 题六类型评测集（当前主评测集） |
| `build_bench_v3.py` / `build_bench_v2.py` / `build_eval_set.py` | 早期评测集 |
| `dump_corpus.py` | 导出语料全文（出题时核对真实内容，避免编造答案） |
| `migrate_add_searchable.py` | 为既有 chunk 补 `searchable` / `section_type` 元数据 |
| `purge_chroma.py` | 彻底重建向量库（换 embedding 模型时用） |
| `show_db_state.py` | 查看持久化层状态 |
| `screenshot_ui.py` / `shot_chat_metrics.py` / `shot_conversation_history.py` | 用 CDP 截图，产出 `docs/screenshots/` |
| `probe_reranker.py` / `probe_usage_fields.py` | 探测外部接口可用性与字段（只读） |
| `test_llm_config.py` / `test_qwen_embedding.py` / `test_reranker.py` | 配置连通性自检 |

---

## 已删除的脚本（及原因）

为避免以后重复踩坑，记录已清理的内容：

| 类别 | 数量 | 原因 |
|---|---|---|
| 依赖已删除模块的脚本（`p32a_*`、`p22_merge` 等） | 38 | import 已不存在的模块，必然 `ImportError` |
| 针对已移除的**外部数据集**（huatuo / ConceptNet） | 5 | 数据集本身已从仓库移除 |
| **一次性诊断**（`diag_*` / `trace_*` / `probe_live_api` / `check_llm_usage`） | 33 | 排查某个历史 bug 的过程脚本，结论已写入 `docs/` |
| **历史一次性操作**（改名 / 迁移 / fix / audit） | 11 | 操作已完成 |

对应的结论都留在文档里，不需要靠脚本复现排查过程：
`docs/retrieval-bm25-migration.md`、`docs/graph-removal-report.md`、
`docs/chat-metrics.md`、`docs/p4-extraction-cost.md`、`docs/retrieval-bm25-migration.md`。
