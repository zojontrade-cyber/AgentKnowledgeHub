# CDC / 增量更新 移除报告

**日期**：2026-09
**范围**：Kafka CDC 链路、watchdog 文件监听、相关配置与依赖
**结论**：全部删除。**当前系统没有增量更新** —— 真实生效的是「幂等整篇替换」。

---

## 一、为什么删

### 1. 它从来没有被接线过

`CDCProcessor`（`services/cdc_processor.py`）**全项目只有定义处一处**：
没有任何模块 import 它，`api/main.py` 的 lifespan 里没有启动逻辑，
编排图里也没有节点调用它。

同样地，`KnowledgeUpdateAgent` 的这三个方法**零调用方**：

| 方法 | 调用方 |
|---|---|
| `start_kafka_consumer()` | **无** |
| `start_watching()` / `stop_watching()` | **无** |
| `detect_changes()` / `commit_hash()` / `forget_hash()` | **无** |

`start-api.bat` 的启动日志里从来没有出现过"文件监听已启动"或"CDC 消费者已启动"——
因为它从来不会被调用。

### 2. 它连依赖都没装正确的路径

`confluent-kafka` 虽然在 `requirements.in` 里，但 `confluent_kafka` 只在
`start_kafka_consumer()` 内部**函数级 import**。既然函数从不被调用，
这条链路从未真正加载过 —— 属于"看起来有、实际为零"的典型。

### 3. 它宣称的是"增量"，实际做不到

`KnowledgeUpdateAgent._handle_modify()` 的全部逻辑是：

```python
doc_id = sha256(file_path)[:16]
await vector_store.delete_by_doc_id(doc_id)   # 全删
await self._handle_create(change, result)     # 全量重建
```

没有 diff、没有只处理变更块。数据结构里的 `DocumentChange.diff_chunks` 字段
**全项目没有任何地方赋值**。

### 4. 而且这条路径还有 doc_id 口径不一致的 bug

- `_handle_modify` / `_handle_delete` 用 `sha256(file_path)[:16]`
- 上传时用的是**随机 uuid4**

两者**永不相等**，因此 `delete_by_doc_id` 删掉 **0 条**，随后**再追加一份新的向量** ——
结果是**重复写，不是替换**。

即便修好这个，该路径仍难使用：传入的 `file_path` 必须落在受管 uploads 目录内，
而那里存的是 uuid 文件名，调用方无从得知。

---

## 二、删了什么

| 文件 / 位置 | 处理 |
|---|---|
| `python/services/cdc_processor.py` | **整文件删除**（224 行） |
| `KnowledgeUpdateAgent.start_kafka_consumer()` | **方法删除** |
| `KnowledgeUpdateAgent.start_watching()` / `stop_watching()` | **方法删除** |
| `__init__` 里的 `_observer` / `_stop_watching` / `_watch_queue` | **字段删除** |
| `__init__` 里的 `import threading` | 删除（已无用途） |
| `settings.kafka_bootstrap_servers` / `kafka_topic_doc_changes` / `kafka_topic_kg_updates` | **配置删除** |
| `requirements.in`：`confluent-kafka`、`watchdog` | **依赖删除** |
| `cdc_processor.py` 里的 `CDCEvent` / `CDCProcessResult` | 随文件删除 |

`knowledge_update_agent.py`：**343 → 241 行**。

---

## 三、留下了什么（以及为什么）

**保留** `KnowledgeUpdateAgent` 的 `process_change` / `detect_changes` / `commit_hash`
与 `POST /api/admin/update` 手动更新接口。

理由：删掉它们会连带删掉 `/api/admin/update` 这个 API 与 `update` 编排流水线，
属于更大的接口变更。当前它们**没有任何调用方**（自动路径已删），
但手动接口仍在路由表里。

**代价（已知且记录在案）**：`/api/admin/update` 仍会走
`_handle_modify` → `sha256(file_path)[:16]` 这条 doc_id 口径不一致的路径。
修它需要改成「按 doc_id 从 documents 表反查」——
本次未做，因为它**不是 CDC 的一部分**，属于另一个独立问题。

---

## 四、当前真实的"更新"语义

对外描述请用这个说法：

> **入库是幂等的整篇替换**（内容哈希去重 + SQLite 状态机可重放），
> **不是增量更新**。

`POST /api/ingest/upload` 的实际行为：

| 情况 | 行为 |
|---|---|
| 内容哈希已存在且 `COMMITTED` | 幂等跳过，返回 `duplicate: true` + 原 `doc_id` |
| 内容变了 | **新建 doc_id**，整篇重新解析 → 抽取 → 建索引 |

**注意**：内容变了会新建 doc_id，旧文档的向量**仍留在库里**（不自动清理）。
如果需要"替换"语义，目前要手动删旧文档。

「状态机 + Outbox 可重放」是**可靠性**机制（失败能从阶段断点重跑），
不是**增量性**机制 —— 重放仍然是整篇重跑。

---

## 五、将来若要做真正的增量

需要改动的部分（供后续参考，本次未实施）：

1. **按 doc_id 而非 file_path 定位**：从 `documents` 表反查，而不是重算 `sha256(path)`
2. **块级 diff**：以 `doc_id#chunk_index` 为键比对新旧块，只重建变化的块
3. **旧块清理**：新 doc_id 建立后删除旧 doc_id 的向量（或改为原地更新）
4. **触发机制**：重新引入文件监听或显式变更事件（本次删掉的部分）

**暂不建议做**：当前语料规模约 258 个块，整篇替换的绝对耗时很低；
增量的收益还撑不起它引入的复杂度（块级 diff + 向量键变更 + 一致性校验）。

---

## 六、验证

```powershell
cd python
python -m pytest tests/ -q          # 121 passed

# CDC 相关符号应全部消失（除注释）
python -c "import pathlib; t=pathlib.Path('agents/knowledge_update_agent.py').read_text(encoding='utf-8'); print([k for k in ('start_kafka_consumer','start_watching','_watch_queue','confluent') if k in t.replace('#','')])"
```

同时确认删掉的依赖没有被其它地方使用：

```powershell
# 应无输出
Select-String -Path python\**\*.py -Pattern 'confluent_kafka|watchdog\.'
```
