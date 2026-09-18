"""验证 Qwen3-Embedding-8B 是否可用，并测试其区分度"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
sys.stdout.reconfigure(encoding="utf-8")

from openai import OpenAI

from config import settings

client = OpenAI(
    api_key=settings.effective_embedding_api_key,
    base_url=settings.effective_embedding_base_url,
    timeout=60,
)

MODEL = "Qwen/Qwen3-Embedding-8B"

print(f"=== 测试模型: {MODEL} ===")
try:
    r = client.embeddings.create(model=MODEL, input="测试文本")
    dim = len(r.data[0].embedding)
    print(f"  [OK] 可用，维度 = {dim}")
except Exception as e:
    print(f"  [FAIL] {type(e).__name__}: {str(e)[:200]}")
    sys.exit(1)

# ── 区分度测试：用真实文档内容 ──────────────────────────
print()
print("=== 区分度测试（这是关键）===")

import math

SRC = Path(r"C:\Users\滨滨\Desktop\Knowledge-Base-Intelligent-Q-A-main\data\clean\institutional")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from agents.doc_parser_agent import DocParserAgent

parser = DocParserAgent()

# 取 6 篇不同文档的正文片段
samples = []
for f in sorted(SRC.glob("*.md"))[:6]:
    raw = f.read_text(encoding="utf-8", errors="ignore")
    clean = parser.strip_metadata(raw)
    title = parser.extract_title(clean)
    body = clean[:300]
    samples.append((title, body))

# 查询：属于第 1 篇文档的内容
queries = [
    ("上班时间是几点到几点", "员工考勤"),
    ("出差住宿费标准", "差旅报销"),
    ("保密级别分几级", "保密管理"),
]

# 一次性编码所有文本
all_texts = [s[1] for s in samples] + [q[0] for q in queries]
resp = client.embeddings.create(model=MODEL, input=all_texts)
vecs = [d.embedding for d in resp.data]
doc_vecs = vecs[:len(samples)]
q_vecs = vecs[len(samples):]


def cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-9)


for qi, (q, expect) in enumerate(queries):
    print(f"\n  Q: {q}   (期望: {expect})")
    scores = []
    for i, (title, _) in enumerate(samples):
        s = cos(q_vecs[qi], doc_vecs[i])
        scores.append((s, title))
    scores.sort(reverse=True)
    for s, title in scores:
        mark = "★" if expect in title else " "
        print(f"    {mark} {s:.4f}  {title}")
    # 区分度：最高分与次高分之差
    gap = scores[0][0] - scores[1][0]
    print(f"    区分度（1位-2位）: {gap:.4f}")
