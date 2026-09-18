# Agent 结构与一次问答的完整链路

**日期**：2026-09
**当前模式**：`qa_mode = pipeline`（ReAct 已实现但非默认，见 `docs/react-ab-report.md`）

---

## 一、整体结构

### 三张独立编排图（LangGraph StateGraph）

```
orchestrator/graph.py
  build_knowledge_graph_workflow()
      ├── ingest  文档入库：parse → extract → vectors → graph → commit   （串行）
      ├── qa      智能问答：answer（单节点，逻辑在节点内）                （线性）
      └── update  文档更新：process ⇄ retry → END                        （有界循环）
```

> **注**：三张图都是 `graph.compile()` **无参**编译 —— 未使用 LangGraph 的
> checkpointer / interrupt / 时间旅行等能力。所谓"可恢复"是本项目用 SQLite
> 状态机自行实现的（`replay_document`）。

### 四个 Agent 的实际角色

| Agent | 是什么 | 参与问答？ |
|---|---|---|
| `DocParserAgent` | 解析切块 | ❌ 仅入库 |
| `KnowledgeExtractAgent` | LLM 抽取三元组 | ❌ 仅入库 |
| `QAAgent` | 意图 → 检索 → 重排 → 生成 | ✅ **主链路** |
| `KnowledgeUpdateAgent` | 文档更新处理器（整篇替换）| ❌ 未被触发 |

**四个 Agent 之间不互相调用**，由编排图用固定边串联 —— 是流水线阶段，不是协商型多智能体。

### 存储

| 存储 | 内容 | 问答时是否使用 |
|---|---|---|
| **Chroma** | 向量索引（BM25 也读它）| ✅ 主检索 |
| SQLite | 文档状态机 / Outbox / 对话历史 / 用量指标 / 审计 | ⚠️ 仅 ACL、用量与显示名 |

---

## 二、一次真实问答的完整链路

**输入**：`请总结员工考勤管理制度的主要内容。`

### ① 意图分类  `[1.55s]`

```
方式：LLM function calling + enum 强制取值（temperature=0）
结果：intent=exploratory   confidence=0.9
静态计划：{'vector': True}
回答风格：先给整体概览，再列出关键要点。
⚠️ 检索计划只读 `vector`，五种意图产出完全相同的检索动作
```

**这是唯一一次"路由决策"。** 分类用 tool calling 的 `enum` 由服务端约束取值，
不可能输出枚举外的值（DeepSeek 不支持 `response_format` json_schema，实测 400）。

### ② 查询改写  `[0.89s]`

```
方式：LLM 生成 1-3 个检索查询 + 实体抽取
queries : ['员工考勤管理制度 主要内容', '考勤管理制度 规定 总结', '公司考勤制度 要点']
entities: ['员工考勤管理制度']
```

### ③ 检索（多查询并发 BM25）  `[0.25s]`

```
命中并去重后：6 条

1. [ 48.34] 员工考勤管理制度 / 6. 异常处理
2. [ 43.85] 员工考勤管理制度 / 5. 外勤与出差
3. [ 40.00] 员工考勤管理制度 / 3. 打卡与迟到早退
4. [ 37.75] 员工考勤管理制度 / 4. 请假流程
5. [ 33.99] 员工考勤管理制度 / 2. 工作时间
6. [ 20.89] 远程办公管理办法 / 4. 工作与沟通     ← 混入的无关文档
```

关键：每条 chunk 同时携带 `content`（命中章节）与 `llm_context`（父文档全文，880 字符）。

### ④ Cross-Encoder 精排  `[0.48s]`

```
模型：BAAI/bge-reranker-v2-m3
取前 20 条送排，输出 top-8

1. [0.8363] 员工考勤管理制度 / 6. 异常处理     (重排前 48.34)
2. [0.8007] 员工考勤管理制度 / 5. 外勤与出差   (重排前 43.85)
3. [0.7500] 员工考勤管理制度 / 3. 打卡与迟到早退 (重排前 40.00)
4. [0.7426] 员工考勤管理制度 / 2. 工作时间     (重排前 33.99)
5. [0.5336] 员工考勤管理制度 / 4. 请假流程     (重排前 37.75)
```

**注意第 4、5 名发生了互换** —— rerank 把"工作时间"提到"请假流程"之前。
这正是 reranker 的价值区间（召回已解决，排序未解决）。

### ⑤ 生成  `[2.59s]`

```
system prompt = ANSWER_PROMPT + 风格「先给整体概览，再列出关键要点。」
喂给 LLM 的上下文 = 每条取 llm_context（父文档全文），非 content
上下文总字符数 ≈ 4800
```

**回答**（节选）：

```markdown
# 员工考勤管理制度主要内容总结
...（工作时间 / 打卡 / 请假 / 外勤 / 异常处理，五节完整）...

**补充说明**：远程办公另有专项规定（如提前 3 个工作日申请…），
可参考《远程办公管理办法》。[来源: YQ-INST-018-远程办公管理办法.md]
```

**总耗时 5.76s**（4 次 LLM 调用 + 1 次 rerank）

---

## 三、这个例子暴露的真实问题

回答**正确且完整**，但末尾那段"补充说明"就是评测中
**summary 类 faithfulness 仅 77.8%** 的原因：

```
检索到的 6 条里只有 1 条来自《远程办公管理办法》（第 6 名，score 20.89）
rerank 后它已被挤出前 5
但 LLM 在生成时仍然"想起"了它并主动补充 + 标注引用
```

**问题出在生成阶段，不在检索阶段。** 检索没问题（`Recall@5` 100%），
rerank 也没问题（把它排到最后），是 LLM 倾向于"多给一点"。

这是 54 题评测里 **citation_correctness 唯一系统性失分点**（86.7% → 若修掉可达 95%+）。

---

## 四、关键参数一览

| 参数 | 值 | 含义 |
|---|---|---|
| `qa_mode` | `pipeline` | 入口意图路由 + 静态计划 |
| `retrieval_mode` | `bm25` | 主召回通道 |
| `rerank_enabled` | `True` | 启用 cross-encoder 精排 |
| `rerank_candidates` | `20` | 送排条数（实测 20 与 50 同效果，延迟减半）|
| `retrieval_mode` | `bm25` | 检索通道（实测优于稠密向量）|
| `qa_max_steps` | `4` | ReAct 循环上限（仅 react 模式生效）|
| 上下文窗口 | `top-8` | 交给 LLM 的条数 |

---

## 五、延迟构成（本次实测）

```
① 意图分类   1.55s   ← LLM
② 查询改写   0.89s   ← LLM
③ BM25 检索  0.25s   ← 本地，近零成本
④ rerank     0.48s   ← 外部 API
⑤ 生成       2.59s   ← LLM
────────────────────
合计         5.76s
```

**LLM 调用占 87% 的延迟**（1.55+0.89+2.59 = 5.03s）。
检索本身（BM25）仅 0.25s —— 所以优化检索对延迟无意义。

---

## 六、ReAct 模式的差异（同一问题）

若设 `qa_mode = react`：

```
意图分类 → 只作 hint（不路由）
   ↓
循环：
  LLM 决定调 search_docs("员工考勤管理制度 主要内容")
    → 看结果
  LLM 判断是否需要补充
    → 可能再调 search_docs / search_graph / list_overview
   ↓
停止（LLM 不再发起工具调用）
   ↓
汇总 → 重排 → 同一个生成链（引用契约不变）
```

实测差异：**工具调用 3.8 次/题，延迟 10.4s（2.05x），四项指标无显著改善。**

---

## 七、复现方式

```bash
# 查看单题完整链路（本文件的数据来源）
python scripts/trace_one_question.py

# Pipeline vs ReAct A/B
python scripts/eval_rag_v4.py --mode pipeline
python scripts/eval_rag_v4.py --mode react
python scripts/compare_ab.py

# 延迟对比
python scripts/compare_latency.py
```
