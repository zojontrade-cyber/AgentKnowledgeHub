# 使用手册

> 项目：AgentKnowledgeHub —— 基于 LangGraph 编排的企业知识库问答系统
>
> 面向「日常使用」：如何启动、上传、提问、核对出处、排障。

---

## 零、Web 界面（推荐入口）

服务启动后，浏览器打开：

## **http://127.0.0.1:8080**

内置界面（「索引台」）包含三个模块：

| 模块 | 功能 |
|---|---|
| **智能问答** | 答案下方给出本轮 **意图 / 置信度 / 出处数 / 检索步骤 / 轮次 / Tokens / 缓存命中 / 耗时**；右侧「本次检索」逐条列出出处（文件名、章节位置、块号、重排得分），可点编号逐条核验 |
| **知识入库** | 拖拽上传；上传接口 202 立即返回，页面轮询 `GET /api/jobs/{doc_id}` 显示入队 → 解析 → 向量 → 入库的真实进度 |
| **数据概览** | 向量块、已上传文件、当前模型配置与各依赖服务状态 |

**首次使用**：页面自动带上默认 Key（`dev-key-1`）。改过 `.env` 里的 Key 后，
在浏览器控制台执行 `localStorage.setItem('akh_key', '你的Key')` 再刷新。

> 视觉规范见 [`ui-design-system.md`](./ui-design-system.md)。
> 开发者入口：**http://127.0.0.1:8080/docs**（Swagger API 文档）。

---

## 一、启动与停止

**只需要一个终端窗口，不需要任何外部服务**
（向量库是嵌入式 Chroma + SQLite，无需 Docker 或任何外部服务）。

```powershell
start-api.bat        # 启动
stop-api.bat         # 停止
```

也就等价于：

```powershell
cd python
python -m api.main
```

就绪标志：

```
[INFO] api.main: 向量库初始化成功
[INFO] api.main: 编排流水线就绪
[INFO] api.main: 对话历史清理任务已启动
INFO:     Uvicorn running on http://127.0.0.1:8080
```

> **端口被占用时不会"初始化到一半才报错"**：启动前会检查端口，
> 若已有本服务在跑会直接提示"无需重复启动"（退出码 0）；
> 若是别的程序占用，会打印它的 PID 与三种处理方式。详见 [`runbook.md`](./runbook.md)。

### 验证服务

```powershell
curl.exe -s http://127.0.0.1:8080/api/health
```

```json
{"status":"ok","service":"AgentKnowledgeHub",
 "dependencies":{"vector_store":"ok","reranker":"ok"}}
```

`dependencies` 是**真实探测**的结果，不是硬编码：向量库不可用会返回 503。
reranker 不可用只降级排序质量（自动回退 BM25 原序），不算故障。

---

## 二、接口鉴权

除 `/api/health` 外，所有接口都需要 API Key，二选一：

```powershell
# 方式 1：X-API-Key 头（推荐）
-H "X-API-Key: dev-key-1"

# 方式 2：Authorization 头
-H "Authorization: Bearer dev-key-1"
```

| Key | 角色 | 数据范围 |
|---|---|---|
| `dev-key-1` | 普通 | 仅 `public` 范围文档 |
| `dev-admin-key-1` | 管理员 | `*`（不限制），可访问 `/api/admin/*` |

Key 在 `python/.env` 的 `API_KEYS` / `ADMIN_API_KEYS` 中配置。
**数据级 ACL**：检索时会按 Key 的 scope 过滤文档，不同角色看到的知识范围不同。

---

## 三、日常使用（三种方式）

### 方式 1：Web 界面（推荐）

见上文第零节。

### 方式 2：命令行（curl）

```powershell
# 提问
curl.exe -s -X POST "http://127.0.0.1:8080/api/qa/ask" `
  -H "X-API-Key: dev-key-1" -H "Content-Type: application/json" `
  -d '{"question":"年假最多有多少天？"}'
```

返回字段：

| 字段 | 说明 |
|---|---|
| `answer` | 答案正文（其中的 `[来源: xxx]` 是模型标注的引用） |
| `sources[]` | 命中章节（`title` / `section` / `content` / `score` / `parent_available`） |
| `intent` / `confidence` | 意图分类结果 |
| `reasoning_steps[]` | 检索与生成过程 |
| `turn` | 服务端计号的会话轮次 |
| `usage` | token、缓存命中、每次调用的阶段明细 |
| `session` | 会话累计（轮数 / tokens / 缓存命中率） |
| `metrics_degraded` | 统计库写入失败时为 `true`（此时数字仅本请求有效） |

> **会话**：不传 `session_id` 就是**匿名单次问答**（`turn=1`、不累计、不落历史，
> 且服务端生成的匿名 id 不会交回给你）。想在界面上累计与恢复历史，
> 用同一个 `session_id` 连续提问即可（Web 界面自动管理）。

### 方式 3：Python 脚本

```python
import httpx

with httpx.Client(base_url="http://127.0.0.1:8080",
                  headers={"X-API-Key": "dev-key-1"},
                  trust_env=False) as c:          # trust_env=False 见下方"已知坑"
    r = c.post("/api/qa/ask", json={"question": "密码长度要求多少位？"})
    data = r.json()
    print(data["answer"])
    for s in data["sources"]:
        print("-", s["title"], ">", s["section"])
```

---

## 四、上传文档

```powershell
curl.exe -s -X POST "http://127.0.0.1:8080/api/ingest/upload" `
  -H "X-API-Key: dev-key-1" `
  -F "file=@D:\docs\员工手册.pdf"
```

上传是**异步**的：立即返回 `202` 与 `doc_id`，后台按状态机处理。

```json
{"doc_id":"156fc719...","status":"PENDING","duplicate":false}
```

### 支持的文件类型

`.pdf` `.png` `.jpg` `.jpeg` `.xlsx` `.xls` `.csv` `.txt` `.md`

### 查询进度

```powershell
curl.exe -s -H "X-API-Key: dev-key-1" `
  http://127.0.0.1:8080/api/jobs/156fc719...
```

状态机（可重放，失败可定位到阶段）：

```
PENDING → PROCESSING → VECTOR_DONE → COMMITTED
                └──────────→ FAILED（超过重试上限）
```

重放失败任务：

```powershell
curl.exe -s -X POST -H "X-API-Key: dev-admin-key-1" `
  http://127.0.0.1:8080/api/jobs/156fc719.../retry
```

> 落盘文件名是**服务端生成的 UUID**（上传接口出于防路径穿越刻意不采用客户端文件名），
> 界面上展示的是 `documents.original_name`（你上传时的中文名）。

---

## 五、入库之后怎么确认检索能命中

```powershell
cd python
python ..\scripts\diag_doc_retrieval.py      # 按文档名查它是否进了索引
python ..\scripts\show_db_state.py           # 看文档状态与向量数
```

若某份文档检索不到，按顺序检查：

1. `GET /api/jobs/{doc_id}` 的状态是不是 `COMMITTED`
2. `documents.searchable` 元数据是否为真（**「文档说明」类章节会被刻意排除**，
   避免它们抢占召回位）
3. 用文档里的原文关键词而不是你自己的概括去检索

---

## 六、故障排查

| 现象 | 原因与处理 |
|---|---|
| 启动报端口被占用 | 跑 `stop-api.bat`；占用者是别的程序时启动日志会打印 PID 与处理方式 |
| `/api/health` 返回 503 | `dependencies` 会指出哪个依赖不可用（通常是向量库） |
| 401 | Key 不对；检查 `.env` 的 `API_KEYS` 与请求头 |
| 提问返回"未检索到相关上下文" | 文档还没入库完（看 `/api/jobs/{id}`），或该问题在语料里确实没有答案 |
| 答案里没有引用 | 模型未按格式标注；可用 `dev-key-1` 跑 `scripts/verify_chat_metrics.py` 复核链路 |
| 换 embedding 模型后检索变差 | **必须重建向量库**（维度不同）：删掉 `python/chroma_data` 重新入库 |
| 脚本请求本机接口莫名超时 | 本机若配了系统代理，httpx 默认会走代理 → 用 `trust_env=False`（见 `runbook.md`） |

---

## 七、配置参考（`python/.env`）

| 变量 | 默认 | 说明 |
|---|---|---|
| `OPENAI_API_KEY` | — | **必填**，LLM 的 Key |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | 任意 OpenAI 兼容接口 |
| `OPENAI_MODEL` | `gpt-4o` | 对话模型（建议用支持 function calling 的） |
| `EMBEDDING_MODEL` | `Qwen/Qwen3-Embedding-8B` | 向量模型（换它必须重建向量库） |
| `RETRIEVAL_MODE` | `bm25` | `bm25` / `dense` / `hybrid`（实测 bm25 最好） |
| `RERANK_ENABLED` | `true` | cross-encoder 精排开关 |
| `API_KEYS` / `ADMIN_API_KEYS` | `dev-key-1` / `dev-admin-key-1` | 鉴权 Key |
| `AUTH_ENABLED` | `true` | 生产环境不要关 |
| `QA_MODE` | `pipeline` | `pipeline` / `react`（ReAct 实测更慢且无质量收益） |
| `ANSWER_PROMPT_MODE` | `stable` | 生成 Prompt 结构；`legacy` 仅供 A/B 对照 |
| `QA_HISTORY_RETENTION_DAYS` | `30` | 对话历史保留天数，`0` = 永久 |
| `MAX_UPLOAD_BYTES` | 见 `.env.example` | 单文件大小上限 |

各 LLM 服务商的具体填法见 [`llm-config-guide.md`](./llm-config-guide.md)。

---

## 八、这份文档之外

- 想了解**为什么这么设计**：`architecture.md`、`retrieval-bm25-migration.md`
- 想了解**指标怎么算**：`chat-metrics.md`、`conversation-history.md`
- 想了解**哪些方向试过但没做**：`graph-removal-report.md`、`p4-extraction-cost.md`、`PROJECT-SUMMARY.md`
- 启动/停止/端口问题的完整排障：`runbook.md`
