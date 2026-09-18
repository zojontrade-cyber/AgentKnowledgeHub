# AgentKnowledgeHub

> 基于 LangGraph 的企业知识库智能问答系统，支持文档解析、知识抽取、RAG 检索、引用溯源与智能问答。

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-FF6B6B)
![SQLite](https://img.shields.io/badge/SQLite-Embedded-003B57?logo=sqlite)

## ✨ Features

* **pipeline Agent**：基于 LangGraph 编排文档解析、知识抽取、问答、知识更新等 Agent。
* **RAG 检索链路**：Query Rewrite → BM25 多查询召回 → Cross-Encoder Rerank → Answer Generation。
* **可核验引用**：答案关联具体文档与章节，支持引用溯源与前端核验。
* **异步知识入库**：支持 PDF、图片、Excel、CSV、TXT、Markdown 等文档解析；按内容哈希幂等去重，内容变化时整篇重建。
* **全链路可观测**：记录模型调用、Token、缓存命中及各阶段耗时。
* **OpenAI Compatible**：支持 DeepSeek、智谱、通义、硅基流动及本地 Ollama 等兼容接口。

## 🏗️ Architecture

```text
                ┌──────────────┐
                │    文档输入    │
                └──────┬───────┘
                       ↓
              ┌─────────────────┐
              │   文档解析 Agent  │
              └────────┬────────┘
                       ↓
          ┌─────────────────────────┐
          │     知识抽取 Agent        │
          └────────────┬────────────┘
                       ↓
              ┌─────────────────┐
              │  向量库 / BM25   │
              └────────┬────────┘
                       ↓
用户提问 → 意图识别 → 查询改写 → 检索召回 → 重排序 → 答案生成
                                                     ↓
                                              答案 + 引用溯源
```

### Agents

| Agent                   | Responsibility   |
| ----------------------- | ---------------- |
| `DocParserAgent`        | 文档解析与章节切分        |
| `KnowledgeExtractAgent` | 实体与关系抽取          |
| `QAAgent`               | 意图识别、改写、检索、重排、生成 |


## 🛠️ Tech Stack

* **后端**: FastAPI + Uvicorn
* **Agent 编排**: LangGraph
* **检索**: BM25
* **重排序**: BAAI/bge-reranker-v2-m3
* **向量嵌入**: Qwen3-Embedding-8B
* **大语言模型**: OpenAI-Compatible API
* **存储**: SQLite + Chroma
* **前端**: HTML / CSS / JavaScript
* **测试**: Pytest

## 📊 Evaluation

基于 54 个测试问题进行评测：

| Metric               |    Result |
| -------------------- | --------: |
| Recall@5             |  **100%** |
| Faithfulness         | **97.8%** |
| Answer Correctness   |  **100%** |
| Citation Correctness | **95.6%** |

## 🚀 Quick Start

### 1. Install

```bash
cd python
pip install -r requirements.txt
```

### 2. Configure

```bash
copy .env.example .env
```

配置 LLM：

```env
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat
```

### 3. Run

```bash
python -m api.main
```

打开：

```text
http://127.0.0.1:8080
```

## 📁 Project Structure

```text
python/                       # 后端核心代码
├── agents/                   # Agent 模块（文档解析 / 知识抽取 / QA / ReAct QA / 知识更新）
├── api/                      # FastAPI 接口（main.py、安全、上传处理、UI 路由）
├── orchestrator/             # LangGraph 工作流编排（graph.py）
├── services/                 # BM25 / 向量存储 / 重排序 / 数据库 / 入库 Worker / LLM 工厂
├── config/                   # 配置管理（settings.py）
├── utils/                    # 工具模块（日志配置）
├── tests/                    # 测试用例
├── chroma_data*/             # Chroma 向量库数据（含备份与 v2）
├── uploads/                  # 用户上传文件
└── requirements.lock         # 依赖锁定

clean/                        # 企业文档示例语料
├── institutional/            # 公司制度文档（30 份：考勤、休假、报销、保密、数据安全等）
└── products/                 # 产品文档（25 份：低代码平台、数据分析平台的介绍/手册/FAQ/发布记录等）

scripts/                      # 评测与诊断脚本
├── accept_*.py               # 验收测试（e2e QA、检索、异步）
├── build_bench_*.py          # 评测集构建（v2/v3/v4）
├── exp*_*.py / p*_*.py       # 实验：嵌入形式、两阶段检索、BM25 vs Dense、重排序、抽取成本
└── compare_ab.py / audit_*.py  # A/B 对比与审计

docs/                         # 架构与技术文档
├── architecture.md / agent-flow.md   # 架构与 Agent 流程
├── retrieval-bm25-migration.md       # 检索迁移说明
├── runbook.md / usage-guide.md       # 运维与使用指南
└── PROJECT-SUMMARY.md                # 项目总结

data/                         # 数据目录
├── eval/                     # 评测数据集（eval_set.jsonl、bench_v2~v4、检索验收数据）
└── agenthub.db               # SQLite 业务数据库

start-api.bat / stop-api.bat  # Windows 启停脚本
README.md                     # 项目说明（架构图、Agent 职责表、技术栈）
```

## 🔍 Highlights

实际测试中，BM25 Recall@5 从原稠密检索的 **23.6% 提升至 89.1%**，最终评测 Recall@5 达到 **100%**；引用准确率达到 **95.6%**。
