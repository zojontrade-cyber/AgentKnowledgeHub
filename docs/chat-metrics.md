# 对话指标：轮次 / Token 消耗 / 缓存命中

面向维护者。回答三件事：这些数字**从哪来**、**怎么读**、**什么时候不可信**。

## 1. 术语（先钉死，否则一定会串）

| 概念 | 归属 | 含义 |
|---|---|---|
| `session_id` | **客户端自有** | Web UI 存在 `sessionStorage`；服务端不自动复用匿名 id |
| `turn` | **服务端维护** | 由 DB 记录在 session 内递增取号，客户端**不上报**轮次 |
| 两者 | —— | **都不代表多轮对话记忆** —— 问答链路是无状态的，不向 prompt 注入历史 |

> 补充（对话历史功能上线后）：每轮问答的**内容快照**会落到 `qa_messages` 表，
> 用于刷新 / 重开后恢复界面。但那**仍然只是展示** ——
> **历史不进 prompt，单轮语义没有变化**（实测：同一问题在有 2 轮历史的会话与
> 全新会话里，prompt token 数差为 0）。详见 `docs/conversation-history.md`。

`turn` 的取值规则：`SELECT COUNT(*) WHERE session_id = ? AND actor = ?` + 1。

**并发前提（重要）**：取号与插入在同一个 SQLite 事务内完成，因此**只在单进程 API 下**保证不重号。
`uvicorn --workers 2` / gunicorn 多进程 / 多实例部署时，SQLite 的行锁无法跨进程串行化，
两个进程可能取到同一个 `turn`。届时应改为**数据库原子序列或唯一约束**，而不是继续依赖 `COUNT`。
当前开发环境是单进程，未受影响。

## 2. 分层契约

```
services/usage_meter.py   归一字段 + stage 归因 + 聚合      （不落库、不返回 HTTP）
agents/*                  只贴 @stage(...) 标签             （不聚合、不算率、不碰 DB）
services/database.py      落库 / 取号 / 汇总查询            （不理解 SDK 字段）
api/main.py               传输
static/app.js             展示                              （不认识 SDK 字段）
```

Pipeline / ReAct / 未来任何 Agent 复用同一套设施 —— 新增可重复的 stage 时，
在 `usage_meter._REPEATABLE_STAGES` 登记，即可自动获得 `xxx_step1/2/...` 命名。

## 3. 字段从哪来

调用 LLM 有两条互不相通的路径，返回形状完全不同，**全部在 `usage_meter` 收口**
（实测见 `scripts/probe_usage_fields.py`）：

| 路径 | 富字段位置 | 回退 |
|---|---|---|
| LangChain `AIMessage`（`LazyLLM.ainvoke`） | `response_metadata["token_usage"]` —— 含 `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens` | `usage_metadata`：`input_tokens`/`output_tokens`，`input_token_details.cache_read` 存在时视作可测 |
| 原生 `Completion`（`_classify_intent`、ReAct 循环） | `resp.usage.prompt_cache_hit_tokens` / `prompt_cache_miss_tokens` | `prompt_tokens_details.cached_tokens` |

归一化后对外**只有一种形状**：

```json
{"call_index": 3, "stage": "react_step2", "model": "...",
 "prompt_tokens": 1200, "completion_tokens": 72, "total_tokens": 1272,
 "cache_hit_tokens": 800, "cache_miss_tokens": 400, "cache_supported": true}
```

注意：`usage_metadata` 回退路径里的 `cache_read` 是否真的对应 DeepSeek 的命中 token
**尚未验证**（探针里两者同为 0，无法区分）。DeepSeek 请求走的是 `token_usage` 主路径，
该回退只在 `token_usage` 缺失时生效。

### 每问几次调用

pipeline 模式 **3 次**：`intent` → `rewrite` → `generate`（reranker 是非 LLM HTTP 调用，不计入）。
react 模式另有 ≤`qa_max_steps` 次 `react_step{N}`。

若 `intent` 调用异常被吞（降级为规则匹配），`llm_calls` 会是 2 而非 3 —— 这是真实信息，不做补齐。

## 4. 缓存的两个口径（最容易误读的地方）

```
cache_hit_rate  = hit_tokens / (hit_tokens + miss_tokens)   ← Token 比例
cache_hit_calls = 命中(hit>0)的调用次数                      ← 调用次数比例
```

两者**不是一回事**：

> 调用 A：0 hit / 1000 miss　调用 B：900 hit / 1000 miss
> → `cache_hit_rate = 45%`，但 `cache_hit_calls = 1/2`

UI 把**前者**称为「缓存命中」，tooltip 明确写出 `Hit / (Hit + Miss)`；
后者的信息只在出处抽屉的逐调用明细里以 `●` 标注。

### `cache_supported` / `cache_measured_calls`

- `cache_measured_calls`：本轮有多少次调用**给出了可识别的 cache 字段**
- `cache_supported`：**全部**调用都可测才为 `true`
- 命中率的**分母只包含可测调用** —— 否则某个阶段缺字段会把有效命中率一起抹掉
- 部分可测时 UI 显示 `45.0%（3/4 次可测）`

「不可测」与「可测但命中 0」必须区分：DeepSeek 明确返回 `hit=0` 属于**可测且未命中**，
provider 根本不报字段才是**不可测**。前者显示 `0.0%`，后者显示 `—`。

## 5. 实测结果与适用范围

`scripts/p5_cache_hit.py` 跑真实 QA 链路，按 stage 展开每次调用：

```bash
python scripts/p5_cache_hit.py                 # 全量 54 题
python scripts/p5_cache_hit.py --limit 10      # 子集
python scripts/p5_cache_hit.py --mode react
```

明细落盘 `data/eval/p5_cache_hit.json`（pipeline）/ `data/eval/p5_cache_hit_react.json`（react），
含 `by_stage` / `by_qtype` / `per_question`。**输出按模式分文件** —— 早期两者共用一个文件名，
跑一次 react 就会把 pipeline 的实测结果覆盖掉。

### 实测结果（bench_v4 全量 54 题，pipeline 模式）

同一配置跑了两次，**两次结果不同** —— 这本身就是结论的一部分：服务端缓存在数小时内长驻，
受前序请求影响，无法真正"冷启动"。

| 口径 | 第一次 | 第二次 |
|---|---|---|
| LLM 调用 | 162（3.00 次/题） | 162（3.00 次/题） |
| Prompt / Completion tokens | 158,399 / 16,132 | 158,128 / 15,884 |
| **Token 命中率** `hit/(hit+miss)` | **74.7%** | **76.7%** |
| **调用命中比例** | 65.4%（106/162） | 66.0%（107/162） |

按 stage（第二次的数据）：

| stage | 调用 | Prompt | 命中 | 未命中 | 命中率 |
|---|---|---|---|---|---|
| `generate` | 54 | 128,870 | 107,388 | 21,482 | **83.3%**（第一次 81.0%） |
| `intent` | 54 | 24,133 | 13,824 | 10,309 | 57.3%（**两次完全一致**） |
| `rewrite` | 54 | 5,125 | 0 | 5,125 | **0.0%**（**两次完全一致**） |

**波动最大的是题型维度，不要在单次结果上做结论**：第一次 compare 仅 60.3%、第二次 82.4%；
single 第一次 73.8%、第二次 66.0%。`intent` / `rewrite` 两次完全一致，
差异集中在 `generate` —— 它的前缀来自检索到的父文档，受服务端缓存里"上一位用户留下了什么"影响。

### ReAct 模式的计量

`react` 模式每问 **5 次**调用，stage 命名由 meter 自动合成：

```
intent:444  rewrite:92  react_step1:791  react_step2:2032  generate:2944
```

顺带修掉了一个**既有缺陷**：`react_qa_agent.py` 的 `_react_loop` 一直调用两个
从未定义的模块级 helper（`_fmt_args` / `_ctx_key`），因此 `qa_mode=react`
在第一次工具调用时必抛 `NameError` —— 整条 ReAct 链路此前**完全不可用**。
因为默认是 `qa_mode=pipeline`，这个缺陷一直没暴露。现已补齐，并加了静态守卫测试
（`tests/test_react_agent_helpers.py` 会扫描"被调用但未定义"的名字）。

`_ctx_key` 的去重粒度**必须是块级 `doc_id#chunk_index`**：用文档级 `parent_id` 去重
会把同一文档的所有章节折叠成一条，静默丢掉命中章节（这是早前 gold 章节"检索不到"的根因之一）。
测试里对这条语义有显式反例断言。

### ⚠️ 一次自我更正：此前"命中率是个位数"的判断是错的

在本功能开工前，基于 `p4d_cache_measure.py`（**抽取链路**，`EXTRACTION_SYSTEM_PROMPT` × 8 次）
的观测，我曾预期问答链路的命中率也是低个位数百分比。**基准实测否定了这个预期**：两次为 74.7% / 76.7%。

但线上真实使用又回到了 8.7% —— 见 5.1 节：两者都对，
**差别在于基准跑的时候缓存是热的**。这恰好说明我最初的"外推"错在
把一个**对缓存温度极度敏感**的数字当成了链路的固有属性。

原因（可从数据读出，不是推测）：

- `generate` 阶段的 prompt 里，上下文来自 **small-to-big 展开后的父文档**（`llm_context`）。
  同一份文档的多个问题会共享一段很长、从头开始就完全相同的文本，正好满足
  "从第 0 个 token 起完全相同的前缀"这个命中条件 → 该阶段 81% 命中。
- `intent` 阶段命中的是固定的 `INTENT_PROMPT` 前缀（每次约 256 token）。
- `rewrite` 阶段**54 次全部 0 命中**，其 prompt 仅约 95 token。阈值行为的具体机制**未验证**，
  只作为观测记录。

教训：**抽取链路的缓存结论不能外推到问答链路** —— 两者的 prompt 结构完全不同。

另一个必须说明的偏差：基准里相邻题目常来自同一份文档，
因此上面 74.7% 相对**生产流量构成是偏乐观的**（生产里问题会在文档间跳转，共享前缀更少）。
`data/eval/p5_cache_hit.json` 的 `per_question` 可直接看出这点 ——
例如相邻的 `消息队列 v4.2 与 v4.1` / `容器平台 v3.4 与 v3.3`（各自来自不同产品文档）
只有 9.1% / 10.3%，而同文档的题目普遍在 80% 以上。

不变量校验：每次可测调用都满足 `hit + miss == prompt_tokens`（162/162），
说明计量没有重复计数。

## 5.1 这个数字为什么不可比：一次线上/基准对照

线上实际问了三个问题，命中率是 **25.5% / 8.7% / 8.7%**，远低于基准的 74.7%–76.7%。
用 `scripts/diag_cache_hit_live.py` 重放同一批问题后，结论如下（全部是算出来的，不是推测）。

### ① 8.7% 是「地板值」，不是 bug

按每次调用的实测阶段长度反推（`intent ≈ 440–446`，其中固定命中 256；`rewrite ≈ 88–94`，命中恒 0；
其余全归 `generate`）：

| 问题 | 截图 total | 截图命中率 | 反推 hit | 其中 intent | 其中 generate |
|---|---|---|---|---|---|
| 产品上线流程中有哪些关键审批节点？ | 3,385 | 25.5% | 848 | 256 | **592** |
| 公司名字叫什么 | 3,071 | 8.7% | 262 | 256 | **≈6** |
| 上线部署要做什么 | 3,409 | 8.7% | 291 | 256 | **≈35** |

即 **8.7% ≈ 256 / 3,010** —— 只有固定的 `INTENT_PROMPT` 前缀命中，
而占全部 token 约 80% 的 `generate` 阶段**整段未命中**。

### ② 同样的代码、同样的问题，我这边重放是 27.5%–51.5%

| 问题 | 线上 generate hit | 我重放 generate hit | 重放命中率 |
|---|---|---|---|
| 产品上线流程中有哪些关键审批节点？ | 592 | **1,152** | 51.5% |
| 公司名字叫什么 | ≈6 | **1,280** | 48.2% |
| 上线部署要做什么 | ≈35 | **512** | 27.5% |

同一个问题 `公司名字叫什么`，`generate` 的 hit 从 **6 变成 1,280（约 200 倍）**，
代码与检索结果都没有变。差别只有一个：**我刚刚把 54 题的 benchmark 跑了两遍**，
同一批父文档文本已经被发过很多次，服务端缓存是热的。

> 结论：这个指标衡量的是「**这批 token 最近有没有被（任何人）发过**」，
> 不是「我们的检索/生成质量」，也**不能跨会话、跨时间比较**。
> 基准的 74.7% 是被基准自身的结构抬高的 —— 54 题连跑会把整个语料自己预热一遍。
> 第 5 节里"偏乐观"的说明方向是对的，但**低估了程度**：真正的驱动因素是"自己给自己预热"。

### ③ 我们自己的 prompt 布局在压制前缀复用（**可改进**）

三个结构性事实：

1. `rewrite` 阶段的 prompt 只有约 90 token，**在全部 162 次基准调用 + 所有线上调用里命中率恒为 0**
   （只作观测，不宣称这是 Provider 的阈值机制）。
2. **意图相关的风格句放在 system prompt 里**：
   `ANSWER_PROMPT + "\n\n本题意图为「factoid」，直接给出事实要点…"`。
   于是**不同意图的两条问题，前 149 字符之后就分叉**（实测分叉点就在「`factoid」`/`procedural」`）。
   只有同意图的连续追问才可能共享前缀。
3. 每条上下文标签里嵌了**分数**：`[来源 1: xxx | 类型: vector | 分数: 0.77]`。
   分数每问都不同，因此在第一个来源处就把前缀打断了。

实测对照：`generate` 的**最长公共前缀只有 0 / 149 / 234 字符**（约 0 / 100 / 160 token），
但实测 hit 是 **1152 / 1280 / 512 token**。

> **更正（见 5.3）**：当时据此推断"Provider 会在 prompt 任意位置复用缓存块"。
> 该推断已被否证 —— stable 结构冷缓存时，尽管父文档正文完全相同，
> `generate` 的 hit 仍为 0。命中确实依赖**从头开始的前缀匹配**；
> 这里解释不了的差值，实际来自"同一问题被重复发送过"（整条 prompt 近似逐字节相同）。

### ④ 想提高命中率，可试的改动

| 改法 | 状态 | 实测结果 |
|---|---|---|
| 意图风格句移出 system + 上下文标签去分数/类型 + 固定动态段顺序 | ✅ **已实施**（`services/prompt_builder.py`） | 稳定前缀 490→1,360 token，但稳态命中率 **−1.2pp（噪声内）**，见 5.3 |
| 让 `_rewrite_query` 可复现（缓存或确定性化） | ❌ 未实施（需改检索行为） | 这是检索结果与 prompt 波动的**上游根因**，见 5.3 |
| 把固定语料前言放在最前 | ❌ 未实施 | — |

**不要**把这个指标当回归门槛：它对缓存温度极度敏感，且受 `_rewrite_query`
非确定性影响。`data/eval/p5_cache_hit.json` 的 `per_question` 里能看到
同一题型在不同轮次间从 60% 跳到 82%。

## 5.3 生成 Prompt 重构：稳定前缀（P0/P1）

### 改了什么

Prompt 构造已从 `QAAgent._generate_answer` 内联字符串抽到
`services/prompt_builder.py`，结构改为：

```
system（常量，所有问题逐字节相同 —— 不要在这里插值）
   角色 / 回答规则 / 引用规则 / 按问题类型的组织规则 / 输出要求

user（动态，顺序固定 metadata → context → query）
   <request_metadata> intent=procedural </request_metadata>
   <retrieved_context> <source id="1" document="…"> … </source> … </retrieved_context>
   <user_query> … </user_query>
```

改动点：

| 项 | 改前 | 改后 |
|---|---|---|
| 意图 | 插值进 **system**：`本题意图为「factoid」，…` | 放进动态段 `<request_metadata>` |
| 上下文标签 | `[来源 1: x \| 类型: vector \| 分数: 0.77]` | `<source id="1" document="x">` |
| 分数 / 类型 | 发给 LLM | **不再发送**（保留在 `RetrievedContext.score`） |
| 稳定前缀长度 | ~150 字符 | **411 字符** |
| 动态段顺序 | 未定义 | 固定 metadata → context → query（问题放最后） |

**一处有意偏离**：文档名仍随来源发送（作为 `document` 属性）。
不能删 —— system prompt 要求用 `[来源: 文档名]` 标注引用，删掉会让引用退化成
无法核对的下标，直接损害引用准确率。它是稳定的（同文档 → 同字符串），不破坏前缀复用。
`tests/test_prompt_builder.py` 对以上每一条都有断言，包括"换意图/换问题 system 段必须逐字节相同"。

### ⚠️ 两次被否证的结论（都记在这里，避免以后再踩）

**否证一：「改后缓存更差」是冷缓存假象。**
第一版 A/B 每种结构只跑一遍，得到 legacy 85.4% / stable **7.6%**。
那是错的：服务端缓存长驻，legacy 结构当天已被发过几百次，stable 一次都没发过。
加上预热后（`--passes`，只取最后一遍）：legacy 84.9% / stable 83.7%，
**差距 1.2pp，落在噪声内**。

**否证二：「命中来自任意位置的缓存块」不成立。**
此前根据"实测 hit（1152）远大于最长公共前缀（0 字符）"推断 Provider 会复用
prompt 中任意位置的内容块。**这个推断是错的**：stable 结构冷缓存时，
尽管父文档正文与 legacy 完全相同，generate 的 hit 仍是 0。
说明命中确实依赖**从头开始的前缀匹配**；之前解释不了的差值，
实际来自"同一问题被重复发过"（整条 prompt 几乎逐字节相同）。
第 5.1 节的推断段落已按此更正。

### 稳态实测（6 题 × 4 遍，取最后一遍）

| 指标 | legacy | stable | 差异 |
|---|---|---|---|
| 缓存① 总命中率 | 84.9% | 83.7% | −1.2pp |
| `generate` 命中率 | 93.1% | 91.3% | −1.8pp |
| `intent` / `rewrite` | 57.7% / 0% | 57.7% / 0% | 0 |
| 稳定前缀 token 总量 | 490 | **1,360** | **+870（约 2.8×）** |

**缓存结论**：稳定前缀确实变长了约 2.8 倍，但在本语料 + 本缓存条件下
**没有测出命中率收益**（−1.2pp 落在噪声内）。原因是复用的大头来自**检索上下文**
（同一问题重复问时整条 prompt 近似逐字节相同），而不是 system 段。
P0 的价值在于**稳健性**（不再有分数这种每问必变的早期分叉点），不在短期命中率数字。

### 质量实测（bench_v4 全 54 题，各跑 1 遍 + LLM-as-Judge）

| 指标 | legacy（改前） | stable（改后） | 差异 |
|---|---|---|---|
| Recall@5（检索，未改动） | 100.0% | 100.0% | 0 |
| **Faithfulness** | 88.9% | **97.8%** | **+8.9pp** |
| 平均分 Faithfulness | 4.58 | 4.93 | +0.36 |
| Answer Correctness | 100.0% | 100.0% | 0 |
| **Citation Correctness** | 80.0% | **95.6%** | **+15.6pp** |
| 拒答率 / 幻觉率 | 100% / 0% | 100% / 0% | 0 |

判分输入对两种结构**完全一致**（judge 看的是正则抽出的 `[来源: xxx]` 与
按 metadata 重建的上下文，不是 prompt 本身），因此可比。

引用准确率 +15.6pp 的合理解释：新 system prompt 明确要求"用 `<source>` 的
`document` 属性作为来源名、不得伪造来源"，并且 `<source id="N">` 的结构化块
比 `[来源 N: x | 类型 | 分数]` 更容易让模型正确归属。

**两点必须说明的保留意见**：

1. 历史归档的 legacy 基线是 Faithfulness 91.1% / Citation 84.4%，
   今天重跑 legacy 得到 88.9% / 80.0% —— 说明 **legacy 自身就有 ±3–4pp 的波动**
   （`_rewrite_query` 非确定性 + judge 波动）。stable 的 97.8% / 95.6%
   同时高于两个 legacy 测量值，因此提升**方向可信**，但具体幅度不要当成精确值。
2. 缓存列的 54 题数字（legacy 78.0% / stable 22.5%）**不可比**：
   stable 结构在那一遍是冷缓存（见上）。稳态比较只认 6 题 × 4 遍那一组。

### 顺手查明的真正瓶颈：检索结果本身不确定

同一问题连问三遍，`generate` 的 prompt **变了**（source #3 从"远程办公管理办法"
变成"加班管理制度"，长度 4856 → 5881）。逐层排查（`scripts/diag_*`）：

| 环节 | 是否可复现 | 证据 |
|---|---|---|
| `_classify_intent` | ✅ 可复现 | 3/3 都是 `factoid`，conf 0.95 |
| **`_rewrite_query`** | ❌ **不可复现** | 同一问题三次给出不同 query 变体与 entities |
| Reranker（SiliconFlow） | ✅ 顺序完全一致 | 4 次调用顺序相同，分数仅 4e-4 浮点噪声 |
| → 最终 top-8 上下文 | ❌ 随 rewrite 变化 | 训练题出现 2 种不同组合 |

所以：**`_rewrite_query` 的 LLM 非确定性会顺着查询变体 → BM25 候选集 →
上下文顺序一路传到 prompt**，这才是命中率与检索结果波动的上游根因。
它的温度已是 0，但 Provider 不保证贪心解码完全可复现。

这解释了基准两次 74.7% / 76.7% 的差异，以及题型维度 60%↔82% 的跳动。
**修复它需要改检索行为（例如对 rewrite 结果做缓存、或让它确定性化），
属于独立决策，本次未实施。**

## 5.4 关于缓存结论的适用范围（**不要外推**）

**在当前使用的模型 / Provider / 请求配置下**实测观察到：重复出现的上下文文本与固定模板前缀发生缓存命中。
该结论属于**当前配置下的实测结果，不视为所有 DeepSeek 模型或接口的通用行为**。
`scripts/probe_usage_fields.py`（132-token prompt，`hit=0 / miss=132`）证明的同样只是
当前请求条件下的行为。

**一句话总结这个指标**：它主要衡量"这批 token 最近被发过没有"，
**不是**链路或回答质量的度量，也**不适合跨会话、跨时间比较**。
需要它稳定可比的场景，应改成"同一批问题、同一缓存状态下连续两次运行取第二次"这类受控测量。

### 一个观察（非本期问题）

响应回报的 `model` 是 `deepseek-flash`，而 `.env` 里请求的是 `OPENAI_MODEL=deepseek-chat`。
计量不受影响，但值得留意。

## 6. 持久化与下钻

两张表（`CREATE TABLE IF NOT EXISTS`，旧库重启自动补）：

- `qa_metrics` —— 每轮一行（轮级汇总，含 `trace_id` / `actor` / `session_id` / `turn`）
- `qa_metrics_calls` —— 每次 LLM 调用一行（按 `stage` 展开）

拆表的理由：轮表只能回答"花了多少"，回答不了"**哪一步**花的"。

`trace_id` 复用日志中间件的 `request_id`（`main.py` 每次请求 `set_request_id`，
并回写 `X-Request-ID` 响应头），因此**用量与日志同源可对**。脱离 HTTP 调用（脚本 / 单测）
时 `request_id` 为 `"-"`，此时改用 `uuid4().hex[:12]`。

定位"某一题 token 突然变大"：

```sql
-- ① 找到那一轮
SELECT turn, ts, total_tokens, trace_id, question
  FROM qa_metrics
 WHERE session_id = 'xxx' AND actor = 'key_...'
 ORDER BY turn;

-- ② 用 trace_id 展开到每一步
SELECT call_index, stage, prompt_tokens, completion_tokens,
       cache_hit_tokens, cache_miss_tokens, cache_supported
  FROM qa_metrics_calls
 WHERE trace_id = '...' AND actor = 'key_...'
 ORDER BY call_index;
```

### 数据隔离

`session_id` 是客户端自有的随机串。**只按它查询**等于"猜到 id 就能读到别人的用量"，
因此：

- `qa_session_summary(session_id, actor)` 与 `qa_session_turns(...)` 都按 **`session_id + actor`** 双条件
- `qa_metrics_calls` **自带 `actor` 列**，`qa_trace_calls(trace_id, actor)` 自身即可隔离，
  不依赖调用方先查轮表验证归属
- `/api/ui/qa-session` 的 `actor` 取自当前 API Key 指纹

## 7. session_id 语义

| 调用方 | 行为 |
|---|---|
| Web UI | 自生成 id（`sessionStorage`），可持续累计 |
| curl / 外部 | 不传 → 为本次请求生成 `anon-{uuid4[:12]}`，`turn = 1`，不形成累计 |
| 客户端显式回传某 id | 由客户端定义为会话（id 随机且按 actor 隔离） |

**关键约束**：未传 `session_id` 时，**响应里的 `session_id` 是 `null`** ——
服务端不把 `anon-*` 交出去，杜绝"服务端生成的 id 被后续请求复用而变成真 session"。
`anon-*` 仍落库（成本核算需要）。

`session_id` 形状白名单：`^[A-Za-z0-9_-]{1,64}$`。不合法视为未提供（参数化查询已防注入，
这里是防脏数据污染索引与聚合）。

## 8. 失败模式

| 场景 | 行为 |
|---|---|
| provider 不返回 usage | 全 0、`cache_supported=false`、UI 显示 `—`（不是 `0%`） |
| 部分阶段缺 cache 字段 | 命中率只在可测调用内计算，UI 标注 `(3/4 次可测)` |
| 调用发生但用量解析失败 | 仍记一行零值（保住 `llm_calls` 的真实性），标记不可测 |
| **统计库写入失败** | `turn=1` + 本轮 usage 合成 `session_stats` + `metrics_degraded=true`；响应始终合法 |
| `metrics_degraded=true` 时 UI | 侧栏显示「统计不可用」，**不把 `turn=1` 当真值展示** |
| 无 active recorder（入库链路） | 采集全部 no-op —— 入库的 LLM 调用不会污染问答统计 |
| 嵌套 `usage_meter()` | 退出时**恢复 previous 而非置 None**，内层不破坏外层 |
| 旧库无表 | 启动 `init_db` 自动补 |

## 9. 前端

- 每条答案的 readout 行：`轮次 / Tokens / 缓存命中 / 耗时`（原 4 格 → 8 格）
- 侧栏「本次会话」：`对话轮数 / 累计 Tokens / 缓存命中`
- 出处抽屉「模型调用（N 次）」：按 `call_index` / `stage` 展开，`●` 标记确实命中的调用
- 刷新页面后调 `/api/ui/qa-session` 回填侧栏，否则"持久化"只是名义上的
- 所有累计数字**取自服务端返回值**，前端不自算权威值（避免与服务端计号漂移）

### 顺带修复的前端缺陷

`app.js` 此前**没有任何引导代码**：`checkHealth()` / `paintLedger()` 定义了却从未被调用，
且没有任何视图被标为 `is-active` —— 而 CSS 是 `.view{display:none}`，
于是首屏是空白的（需点一次导航才出现）。会话回填必须挂在启动流程里，故在文件末尾补了
`boot()`（`setView('chat')` + `checkHealth()` + `loadOverview()` + `restoreSession()`）。

## 10. 验证脚本

| 脚本 | 验什么 |
|---|---|
| `python -m pytest tests/test_usage_meter.py tests/test_qa_metrics.py -q` | 归一化/口径/嵌套/取号/隔离（27 项） |
| `python scripts/p5_cache_hit.py [--limit N] [--mode react]` | 真实链路 token 与缓存命中，落盘 `data/eval/p5_cache_hit.json` |
| `python scripts/verify_chat_metrics.py [--url ...]` | 真实 HTTP 路径：计量字段、轮次递增、会话回填、actor 隔离、匿名语义（21 项） |
| `python scripts/verify_ui_metrics.py [--port N]` | 无头 Chrome 走真实浏览器：首屏可见、readout 8 格、侧栏递增、刷新回填、调用明细（27 项） |
| `python scripts/shot_chat_metrics.py` | 截图 `docs/screenshots/07-chat-metrics.png` |

**本机注意事项**：`verify_chat_metrics.py` 必须用 `httpx.Client(trust_env=False)`。
本机注册表配置了系统代理，httpx 默认读取后会把 `127.0.0.1` 的请求也走代理，
实测**全部 ReadTimeout（20s 挂死）**，而服务端其实在 6.8s 内就正常返回了 200 ——
排查时不要误判成服务端卡死。

## 11. 运维前提

`8080` 端口上运行的可能仍是**旧代码**（提权进程，普通 shell 杀不掉）：

- `.html/.js/.css` 由 `StaticFiles` 每次请求读盘 → **刷新浏览器即可见**
- 新增的 Python 逻辑与 `/api/ui/qa-session` 路由 → **必须手动重启 API 进程**
- 未重启期间：readout 无新格、侧栏显示 `—`，属预期而非 bug
