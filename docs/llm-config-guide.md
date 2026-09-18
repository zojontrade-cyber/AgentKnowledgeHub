# LLM 配置指南

> 你只需要填 `python/.env` 里的 **3 行**，然后重启服务。

---

## 快速开始

打开 `python/.env`，找到这一段：

```env
OPENAI_API_KEY=sk-你的Key
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
```

按你用的服务商改成下表对应的值即可。

---

## 各服务商配置对照

### 1. DeepSeek（推荐，国内便宜）

```env
OPENAI_API_KEY=sk-你的Key
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-chat
EMBEDDING_MODEL=text-embedding-3-small
```

⚠️ **注意**：DeepSeek **不提供 embedding 接口**。检索需要 embedding，所以要另配：
- 用硅基流动的免费 embedding：`OPENAI_BASE_URL` 冲突，需要单独配置
- 或改用下面的**智谱**（模型和 embedding 都有）

### 2. 智谱 AI（GLM，模型+embedding 齐全）

```env
OPENAI_API_KEY=你的智谱Key
OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4
OPENAI_MODEL=glm-4-plus
EMBEDDING_MODEL=embedding-3
```

✅ 推荐：一个 Key 同时解决对话和向量

### 3. 阿里通义千问

```env
OPENAI_API_KEY=sk-你的DashScopeKey
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_MODEL=qwen-plus
EMBEDDING_MODEL=text-embedding-v3
```

### 4. 硅基流动（有免费额度）

```env
OPENAI_API_KEY=sk-你的Key
OPENAI_BASE_URL=https://api.siliconflow.cn/v1
OPENAI_MODEL=Qwen/Qwen2.5-7B-Instruct
EMBEDDING_MODEL=BAAI/bge-m3
```

### 5. OpenAI 官方

```env
OPENAI_API_KEY=sk-你的Key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-small
```

### 6. 本地 Ollama（完全免费）

先安装 [Ollama](https://ollama.ai/)，然后：

```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

```env
OPENAI_API_KEY=ollama
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=qwen2.5:7b
EMBEDDING_MODEL=nomic-embed-text
```

---

## ⚠️ 三个必须知道的坑

### 坑 1：模型必须支持 function calling

本项目重构后，意图识别用了 `with_structured_output`（结构化输出）。**不支持的模型会降级为关键词规则匹配**——功能不中断，但意图识别精度下降。

**日志里会出现**：`意图分类结构化输出失败，降级为规则匹配`

支持的模型：`gpt-4o`、`glm-4-plus`、`qwen-plus`、`deepseek-chat` 等主流模型

### 坑 2：embedding 模型必须与语料一致

如果你换 embedding 模型，**必须重建向量库**（因为维度不同）。切换方式：

```powershell
# 删掉旧的向量数据（在 python/ 目录下执行）
Remove-Item chroma_data -Recurse -Force
```

### 坑 3：中文场景建议用中文友好的 embedding

| embedding 模型 | 中文效果 |
|---|---|
| `text-embedding-3-small` | 一般 |
| `BAAI/bge-m3` | ✅ 好（推荐中文） |
| `embedding-3`（智谱） | ✅ 好 |
| `text-embedding-v3`（通义） | ✅ 好 |
| `nomic-embed-text` | 一般（偏英文） |

---

## 配置完怎么验证

### 第 1 步：启动服务

```powershell
cd python
python -m api.main
```

**看到这行就说明配置成功**：
```
[INFO] api.main: 编排流水线就绪
```
**不应出现**：
```
[WARNING] api.main: 未配置 OPENAI_API_KEY
```

### 第 2 步：验证 LLM 能调用

```powershell
$key = "dev-key-1"
$body = '{"question":"测试"}'
curl.exe -s -X POST "http://127.0.0.1:8080/api/qa/ask" `
  -H "X-API-Key: $key" -H "Content-Type: application/json" -d $body
```

- 返回 `answer` 字段 → ✅ 成功
- 返回 401 / 500 且报 `Missing credentials` → Key 没填对
- 返回 `Connection error` → `OPENAI_BASE_URL` 错了

### 第 3 步：跑通完整链路

打开 Web 界面 <http://127.0.0.1:8080>，上传一份文档（入库是异步的，
界面会显示进度），然后提问。

**不需要任何外部服务** —— 向量库是嵌入式 Chroma，无需 Docker；
无需任何外部服务。

---

## 当前环境状态

| 组件 | 状态 |
|---|---|
| 嵌入式 Chroma 向量库 | ✅ 无需 Docker，启动即用 |
| 重排（cross-encoder） | ✅ `BAAI/bge-reranker-v2-m3`，不可用时自动回退 BM25 原序 |
| 外置数据库 / 图数据库 | — **不需要**（SQLite + 嵌入式 Chroma） |
| 鉴权 | ✅ 已启用（`X-API-Key: dev-key-1`） |
| **LLM** | ⏳ **等你配置** |

配置好之后，你就能上传自己的企业文档并提问了。

> 仓库里 `data/eval/` 下带着评测集（`bench_v4.jsonl`，54 题 / 六种题型）
> 与历史评测结果，可用 `scripts/` 里的脚本复现检索与答案质量指标。
