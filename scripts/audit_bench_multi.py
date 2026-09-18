"""
审计 bench_v3 的 exact 标签：是否存在**多个章节都能回答问题**的情况。

方法：对每题，取其 top5 BM25 候选，若前 5 名里出现了**与 gold 不同文档
且标题含问题关键词**的章节，则该题可能是 multi 而非 exact。

只读审计，不修改文件。
"""

import json
import os
import sys
from pathlib import Path

_PY = Path(Path(__file__).resolve().parent.parent / "python")
os.chdir(_PY)
sys.path.insert(0, str(_PY))
sys.stdout.reconfigure(encoding="utf-8")

from agents.doc_parser_agent import DocParserAgent
from services.bm25 import BM25Index

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(
    r"C:\Users\滨滨\Desktop\Knowledge-Base-Intelligent-Q-A-main\data\clean\institutional"
)
BENCH = ROOT / "data" / "eval" / "bench_v3.jsonl"

parser = DocParserAgent()
chapters = []
for f in sorted(SRC.glob("*.md")):
    dt, secs = parser.parse_tree(parser.strip_metadata(f.read_text(encoding="utf-8", errors="ignore")))
    if not dt:
        continue
    for i, s in enumerate(secs):
        if not s.body:
            continue
        if parser.classify_section(s.title, s.body) == "doc_overview":
            continue
        chapters.append({"doc": dt, "idx": i, "sec": s.title, "body": s.body,
                         "text": f"{dt}\n章节：{s.title}\n{s.body}"})

idx = BM25Index()
idx.build([c["text"] for c in chapters], [str(i) for i in range(len(chapters))])

rows = [json.loads(l) for l in BENCH.read_text(encoding="utf-8").splitlines() if l.strip()]
exact = [r for r in rows if r["gold_type"] == "exact"]

print(f"审计 exact 类 {len(exact)} 题\n")
suspects = []
for r in exact:
    q = r["question"]
    gdoc, gidx = r["doc_title"], r["gold_chunk_index"]
    hits = idx.rank_ids(q, top_k=5)
    others = []
    for sid, sc in hits:
        c = chapters[int(sid)]
        if c["doc"] == gdoc and c["idx"] == gidx:
            continue
        # 该竞争章节的标题是否包含问题中的关键词
        kws = [w for w in ["申诉", "绩效", "加班", "年假", "报销", "保密", "导出",
                           "归档", "盘点", "报废", "评审", "分支", "提交", "审批",
                           "申请", "流程", "交接", "离职", "培训", "渠道"]
               if w in q]
        if any(k in c["sec"] for k in kws):
            others.append((c, sc))
    if others:
        suspects.append((r, others))

print("=" * 78)
print(f"疑似 multi（存在标题命中问题词的竞争章节）: {len(suspects)}")
print("=" * 78)
for r, others in suspects:
    print(f"\nQ: {r['question']}")
    print(f"   gold: {r['doc_title']} / {r['section_title']}")
    for c, sc in others:
        print(f"   竞争者 [{sc:6.2f}] {c['doc']} / {c['sec']}")

if not suspects:
    print("\n未发现明显的多章节竞争。")
