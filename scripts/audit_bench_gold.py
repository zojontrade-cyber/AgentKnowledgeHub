"""审计 bench_v2 的 chunk 级 ground truth 是否可信。"""
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

b = [
    json.loads(l)
    for l in (ROOT / "data" / "eval" / "bench_v2.jsonl").read_text(encoding="utf-8").splitlines()
    if l.strip()
]
items = [r for r in b if r["in_corpus"] and r["gold_chunk_index"] is not None
         and not r["ambiguous"]]

BOILER = ("本办法", "本规定", "本制度", "本规范", "本管理办法")

print(f"chunk 级可评测: {len(items)}")
print()

suspect = [r for r in items if str(r["expected_answer"]).startswith(BOILER)]
print(f"expected_answer 以套话开头（疑似错标）: {len(suspect)} / {len(items)}")
print()

# 对每题检查：答案文本是否真的出现在其标注的章节里
from agents.doc_parser_agent import DocParserAgent

parser = DocParserAgent()
docs = {}
for f in sorted(SRC.glob("*.md")):
    dt, secs = parser.parse_tree(parser.strip_metadata(f.read_text(encoding="utf-8", errors="ignore")))
    if dt:
        docs[dt] = [(s.title, s.body) for s in secs]

import unicodedata
import re


def norm(s):
    return re.sub(r"[\s\u3000]+", "", unicodedata.normalize("NFKC", s or ""))


bad = 0
for r in items:
    title = r["doc_title"]
    gi = r["gold_chunk_index"]
    secs = docs.get(title)
    if not secs or gi >= len(secs):
        print(f"  [越界] {r['question']}")
        bad += 1
        continue
    ans = norm(str(r["expected_answer"]))
    body = norm(secs[gi][1])
    if ans[:30] and ans[:30] not in body:
        print(f"  答案不在标注章节内: {r['question']}")
        print(f"    标注章节: {secs[gi][0]}")
        print(f"    答案: {str(r['expected_answer'])[:60]}")
        bad += 1

print()
print(f"答案与标注章节不匹配: {bad} / {len(items)}")
