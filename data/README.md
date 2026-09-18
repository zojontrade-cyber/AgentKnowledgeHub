# data/ 目录说明

本目录**只保留评测集与历史评测结果**。

```
data/
└── eval/
    ├── bench_v4.jsonl        ← 主评测集：54 题 / 六种题型（每类 9 题）
    ├── bench_v3.jsonl        ← 早期评测集（36 题）
    ├── bench_v2.jsonl        ← 更早版本
    ├── filename_map.json     ← 上传文件名(UUID) → 中文文档名的映射
    └── *.json                ← 历次实验的评测结果（见下）
```

## 评测集结构（`bench_v4.jsonl`）

每行一题：

| 字段 | 说明 |
|---|---|
| `question` | 问题 |
| `qtype` | `single` / `multi` / `multihop` / `summary` / `compare` / `noanswer` |
| `answerable` | 知识库中是否有答案（`noanswer` 类为 `false`） |
| `doc_title` / `doc_titles` | gold 文档 |
| `expected_answer` | 标准答案（来自知识库原文，是 LLM Judge 的判定依据） |
| `why_not_in_corpus` | `noanswer` 题用来向 Judge 解释"为什么语料里没有" |

## 结果文件怎么读

| 文件 | 内容 |
|---|---|
| `rag_four_metrics_v4_pipeline.json` | 四项指标（pipeline 模式）：Recall@5 / Faithfulness / Correctness / Citation |
| `rag_four_metrics_v4_react.json` | 同上，ReAct 模式（用于 A/B，结论是保留 pipeline） |
| `p5_prompt_ab_stable.json` / `_legacy.json` | 生成 Prompt 稳定前缀改造的 A/B（含每题的逐次调用明细） |
| `p5_cache_hit.json` | token 与缓存命中实测（逐 stage / 逐题型 / 逐题） |
| `p32a_*.json` | 知识图谱 vs 向量的 A/B —— **图谱因此被移除**（被引用率 7.1%，准确率 −2.8pp） |
| `p4b_*` / `p4c_*` | 抽取批处理与事实层实验（负结果） |
| `p1_*` / `p2_*` | 检索与精排的早期实验 |

复跑评测用 `scripts/p5_cache_hit.py`、`scripts/p5_prompt_ab.py`
（四项指标的 LLM Judge 版本见脚本内的 judge prompt 定义）。

## 为什么大文件不在仓库里

以下数据曾是外部测试集，**体积大且可重新下载**，已从仓库移除：

- `huatuo_qa/`（约 295 MB）—— 中文医学问答数据集
- `conceptnet_zh/`（约 487 MB）—— 中文常识图谱（用于已移除的图谱功能）
- `backup/`（约 11 MB）—— 图谱数据备份

它们的用途是验证"多跳推理"和"信息不足时是否编造"，但**不是项目源码的一部分**。
企业知识问答的实际语料在 `python/uploads/`（本地数据，同样不入库）。
