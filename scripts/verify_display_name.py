"""
display_name 改造的最小回归验证（用户指定的 6 项）

  1. 文档列表 -> 中文名称
  2. QA 回答 -> 中文引用
  3. 检索     -> source_path 不变
  4. Chroma   -> 不变
    6. 57 篇文档 -> 状态仍全部正常
"""

import json
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

BASE = "http://127.0.0.1:8093"
KEY = "dev-key-1"


def api(path):
    req = urllib.request.Request(BASE + path, headers={"X-API-Key": KEY})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def ask(q):
    body = json.dumps({"question": q}).encode("utf-8")
    req = urllib.request.Request(
        BASE + "/api/qa/ask", data=body, method="POST",
        headers={"X-API-Key": KEY, "Content-Type": "application/json; charset=utf-8"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode("utf-8"))


print("=" * 84)
print("1. 文档列表 -> 中文名称")
print("=" * 84)
d = api("/api/ui/documents")
print(f"  总数: {d['total']}")
for x in d["documents"][:10]:
    print(f"    {x['display_name'][:40]:<42} [{x['status']}] {x['size_human']}")
uuids = [x for x in d["documents"] if x["display_name"].endswith(".md")
         and len(x["display_name"].split("-")[0]) == 32]
print(f"\n  仍是 UUID 名的: {len(uuids)}")
missing = [x for x in d["documents"] if not x["display_name"].strip()]
print(f"  空名称的: {len(missing)}")

print()
print("=" * 84)
print("2. QA 回答 -> 中文引用")
print("=" * 84)
for q in ("请假要走什么流程？", "网关被限流后返回什么？"):
    r = ask(q)
    ans = (r.get("answer") or "")
    print(f"\n  Q: {q}")
    print(f"  A: {ans[:160]}")
    import re
    cites = re.findall(r"\[来源[:：]\s*([^\]]+)\]", ans)
    print(f"  引用标签: {cites}")
    for c in cites:
        if len(c.split("-")[0]) == 32:
            print(f"     ❌ 仍含 UUID: {c}")
    srcs = r.get("sources") or []
    print(f"  sources[0].title = {srcs[0].get('title') if srcs else '-'}")

print()
print("=" * 84)
print("3-4. 底层存储未变（source_path / Chroma）")
print("=" * 84)
import os
import sqlite3
from pathlib import Path

os.chdir(str(Path(__file__).resolve().parent.parent) / "python")
sys.path.insert(0, ".")
for _m in ("pandas", "pyarrow"):
    sys.modules.setdefault(_m, None)
from config import settings

conn = sqlite3.connect(str(Path(settings.sqlite_path)))
conn.row_factory = sqlite3.Row
rows = list(conn.execute("SELECT source_path, original_name FROM documents LIMIT 3"))
print("  SQLite:")
for r in rows:
    print(f"    source_path={str(r['source_path'])[:46]}")
    print(f"    original_nm={str(r['original_name'])[:46]}")
n_uuid_path = conn.execute(
    "SELECT COUNT(*) FROM documents WHERE source_path LIKE '%\\%' "
    "AND length(source_path) - length(replace(source_path,'\\','')) = 1"
).fetchone()[0]
print(f"    文档总数: {conn.execute('SELECT COUNT(*) FROM documents').fetchone()[0]}")
conn.close()

import chromadb

c = chromadb.PersistentClient(path=settings.chroma_path)
col = c.get_or_create_collection("knowledge_chunks")
print(f"  Chroma: {col.count()} 条向量（应为入库后数量，未因展示改造变化）")
got = col.get(limit=2, include=["metadatas"])
for m in got["metadatas"]:
    print(f"    metadata.source = {str((m or {}).get('source'))[:60]}")


print()
print("=" * 84)
print("6. 文档状态")
print("=" * 84)
from collections import Counter
st = Counter(x["status"] for x in d["documents"])
for k, v in st.most_common():
    print(f"    {k:<14} {v}")
print(f"    合计 {sum(st.values())}")
