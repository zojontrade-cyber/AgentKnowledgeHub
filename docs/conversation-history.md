# 对话历史（展示保留）

面向维护者。这个机制**只做一件事：让人在刷新/重开后还能看到之前的问答与出处**。

## 0. 最重要的一句话

> **历史不注入 prompt。** 问答链路仍是**单轮无记忆**，
> `answer(question, acl_scopes)` 的签名与语义都没变。

如果你想做的是"让模型记得上文"（支持「它 / 还有呢 / 那这个」这类指代追问），
那是**另一件事**，本机制不提供，也不应该顺手加上去 —— 它要改 prompt 结构，
而实测证明前缀复用对 prompt 结构极其敏感（见 `chat-metrics.md` 5.3）。

判据：`scripts/verify_chat_history.py` 第 5 节会断言
**"同一问题在有 2 轮历史的会话里 vs 全新会话里，prompt token 数不因历史而变大"**。
实测 diff = 0。这条断言就是"历史没进 prompt"的可执行证明。

## 1. 存什么

新建表 `qa_messages`（与 `qa_metrics` **分开**）：

```sql
CREATE TABLE IF NOT EXISTS qa_messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              REAL NOT NULL,
    trace_id        TEXT,          -- 与 qa_metrics 同源，可关联
    actor           TEXT,          -- API Key 指纹
    session_id      TEXT NOT NULL,
    turn            INTEGER NOT NULL,   -- 与 qa_metrics.turn 同值
    payload_version INTEGER NOT NULL DEFAULT 1,
    payload         TEXT NOT NULL       -- 当轮响应快照 JSON
);
CREATE INDEX IF NOT EXISTS idx_qam_session ON qa_messages(actor, session_id, turn);
CREATE INDEX IF NOT EXISTS idx_qam_ts      ON qa_messages(ts);
```

### 为什么与 `qa_metrics` 分表

| | `qa_metrics` | `qa_messages` |
|---|---|---|
| 回答的问题 | "花了多少" | "当时答了什么" |
| 行大小 | 小（纯数字） | 约 5KB/轮 |
| 保留 | 长期（成本趋势） | 30 天 TTL |
| 访问模式 | 聚合 | 按会话顺序读 |

混在一张表里，以后想"只清内容、保留成本数据"就清不动了。

### 为什么存快照 JSON 而不是拆列

前端 `renderQA` / `renderProv` 需要的正是这个形状（`question / answer / sources /
reasoning_steps / intent / confidence / turn / usage.per_call / metrics_degraded`）。
存快照后，恢复可以直接复用**与实时回答完全相同**的渲染函数 ——
**不新增渲染代码，也不可能出现"恢复出来的卡片和当时看到的不一样"**。

代价：答案文本不能 SQL 查询。展示保留不需要检索历史，所以接受。

### 为什么出处正文要一起存，读取时不重取

文档会被重新入库/更新。若读取时按 `source` 重新取正文，
恢复出来的引用会与当时答案依据的文本不一致 —— 用户核对时就会得出"引用是错的"。

**存下来的才是"当时引用的证据"。**

### 引用契约（不要改）

`sources[].content` 必须是**命中的章节**（`RetrievedContext.display_content`），
不是 small-to-big 展开后的父文档（`llm_context`）。存错会让每条引用的开头
永远是"## 1. 文档说明"，用户无法核对出处 —— 这正是早期引用准确率只有 44.4% 的原因。
`tests/test_qa_history.py::test_sources_use_hit_section_not_parent_document` 钉住了这条。

### 唯一一份序列化

`services/qa_payload.py` 的 `build_turn_payload()` 同时被**编排节点（落库）**
与 **API（响应）** 调用。两边各写一份的话，一旦漂移就会出现
"实时答得对、刷新后恢复出来的卡片不一样"这种极难排查的问题。

## 2. 写路径

`orchestrator/graph.py` 的 `process_question`，在现有用量写入的**同一个 `try/except`** 内：

```python
if result and session_id and not session_id.startswith("anon-"):
    db.record_qa_message(actor=..., session_id=..., turn=turn,
                         trace_id=..., payload=build_turn_payload(...))
```

- `turn` 直接取 `record_qa_metrics()` 的返回值 —— **轮次的权威在 `qa_metrics`**，
  这里只沿用，不重新取号（否则两张表可能算出不同的轮次）
- **匿名请求不写**：服务端生成的 `anon-*` 永不交给客户端（见 `api/main.py`
  的匿名语义），写进去没有任何人能读回来，纯占空间
- **非原子**：metrics 成功而这里失败时，历史缺一轮，但 `turn` 不错位。
  历史是展示数据，**有意接受**这种非原子性，以免改动已经验证过的 `record_qa_metrics`
- 写库失败不影响问答（`try/except` + 告警）

## 3. 读路径

```
GET /api/ui/qa-history?session_id=xxx&limit=50
```

返回：

```json
{"session_id": "...", "total": 3, "returned": 1, "truncated": true,
 "turns": [ { "payload_version": 1, "turn": 3, "question": "...", "answer": "...",
              "sources": [...], "reasoning_steps": [...], "usage": {...} } ]}
```

- `limit` 默认 50、上限 200；返回**最近 N 轮但按 `turn` 升序**（便于直接顺序渲染）
- `total` 是会话内历史总轮数，`truncated = total > returned`，前端据此提示
  「仅显示最近 N 轮（共 M 轮）」
- **必须 `session_id + actor` 双条件过滤** —— `session_id` 是客户端自有的随机串，
  只按它查询等于"猜到 id 就能读到别人的问答内容与出处"
- 坏 `payload` 行**跳过并告警**，不让一行坏数据打挂整段历史

## 4. 保留期（TTL）

- `settings.qa_history_retention_days`，默认 **30**；**<= 0 表示永久保留**
- `settings.qa_history_purge_interval_hours`，默认 **6**
- `api/main.py` 的 `lifespan`：启动时清理一次，然后起一个周期 asyncio 任务，
  关闭时 `cancel`；`retention_days <= 0` 时**不启动**任务
- `db.purge_qa_messages(days)` 在 `days <= 0` 时**直接返回 0** ——
  绝不能把 0 解释成"删光"（测试 `test_purge_zero_days_keeps_everything` 钉住）
- **清理只作用于 `qa_messages`**，不动 `qa_metrics`

## 5. 前端

- `boot()` 顺序：`setView('chat')` → `checkHealth()` → `loadOverview()` →
  `restoreSession()`（侧栏数字）→ `restoreHistory()`（历史卡片）
- `restoreHistory()` 逐轮调 `renderQA(turn.question, turn)` —— 与实时同一条渲染路径
- 抽屉（"本次检索"）恢复为**最后一轮**的出处；`truncated` 时在顶部插一条 `.qa-note`
- 旧进程没有该路由（404）或网络失败 → **静默降级**，实时问答不受影响

### 两个顺带修掉的问题

1. **`renderQA` 原本读模块级状态**（`qa.degraded` / `qa.turn`）而不是 `reply`。
   逐条恢复多条历史时，所有卡片会被套上"当前这一轮"的轮次与降级标记。
   现已改为一律从 `reply` 读。
2. **点历史卡片的引用会高亮错卡片**。`selectCite()` 原先固定操作 `qa.qaEl`
   （最后一张卡）。现改为按被点击的卡片作用域，并把抽屉切到**那一轮**的出处
   （否则会出现"卡片是第 1 轮、抽屉列的是第 3 轮证据"的错配）。
   实现方式：`renderQA` 在卡片上挂 `wrap._reply`。

## 6. 安全取舍（**有意为之**）

历史里存的是**当时的答案与出处**。若某个 Key 之后被收窄了 ACL，
它仍能从历史读到当初基于更宽权限生成的答案。

**默认不做读取时的 ACL 重校验**，理由：该内容**本来就已经发给过同一个主体**。

若要加固，需要为每轮存下命中文档的 `acl_scope`，读取时逐条重校验，
命中越权的轮次只回问题不回答案 —— 成本明显更高，**本次未实施**。

其他隔离措施：

- 读写都按 `session_id + actor` 双条件
- 匿名单次问答不入库（`anon-*` 从不外泄，因此其命名空间里不该有可查询数据）
- `tests/test_qa_history.py` 覆盖 actor / session 双向隔离

## 7. 验证

| 脚本 | 覆盖 |
|---|---|
| `pytest tests/test_qa_history.py -q` | 快照契约 / 读写 / 隔离 / limit / TTL 边界（20 项） |
| `python scripts/verify_chat_history.py` | 真实 HTTP：2 轮恢复、匿名不写、actor 隔离、引用契约、**历史未进 prompt**、truncated（16 项） |
| `python scripts/verify_ui_metrics.py` | 真实浏览器：刷新后卡片数 / readout 格数 / 轮次 / 出处按钮 / 抽屉恢复（43 项） |

**本机注意**：验证脚本用 `httpx.Client(trust_env=False)` ——
本机注册表配了系统代理，httpx 默认会把 `127.0.0.1` 的请求也走代理而挂死。
