"""
校验 bench_v3 的 exact 类标签：逐条打印 gold 章节的**真实正文**，
供人工确认"该章节确实回答了该问题"。

不做自动判定 —— 之前的教训是自动定位会产生"相关但没回答"的假 gold。
"""

import json
import os
import sys
from pathlib import Path

_PY = Path(Path(__file__).resolve().parent.parent / "python")
os.chdir(_PY)
sys.path.insert(0, str(_PY))
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(
    r"C:\Users\滨滨\Desktop\Knowledge-Base-Intelligent-Q-A-main\data\clean\institutional"
)

from agents.doc_parser_agent import DocParserAgent

parser = DocParserAgent()
docs = {}
for f in sorted(SRC.glob("*.md")):
    dt, secs = parser.parse_tree(parser.strip_metadata(f.read_text(encoding="utf-8", errors="ignore")))
    if dt:
        docs[dt] = [(s.title, s.body) for s in secs]

rows = [
    json.loads(l)
    for l in (ROOT / "data" / "eval" / "bench_v3.jsonl").read_text(encoding="utf-8").splitlines()
    if l.strip()
]
exact = [r for r in rows if r["gold_type"] == "exact"]

for n, r in enumerate(exact, 1):
    secs = docs.get(r["doc_title"])
    body = ""
    if secs and r["gold_chunk_index"] is not None and r["gold_chunk_index"] < len(secs):
        body = secs[r["gold_chunk_index"]][1]
    print(f"[{n:>2}] Q: {r['question']}")
    print(f"     gold: {r['doc_title']} / {r['section_title']}")
    print(f"     为什么: {r['why_answerable']}")
    print(f"     正文: {body.strip()[:220].replace(chr(10), ' / ')}")
    print()
