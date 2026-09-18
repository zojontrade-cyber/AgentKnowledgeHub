"""
P1：Dense vs BM25 —— retrieval-only，无 reranker，文档级 + chunk 级双指标

冻结变量：
  - 语料：institutional 30 篇 / 159 章节（与生产 choma_data_v2 一致）
  - 不用 reranker
  - 不调生成 LLM
  - 两套问题：original（含泄漏）/ leakfree（剥掉标题核心词）
"""

import asyncio
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

_PY = Path(__file__).resolve().parent.parent / "python"
os.chdir(_PY)
sys.path.insert(0, str(_PY))
sys.stdout.reconfigure(encoding="utf-8")

from langchain_openai import OpenAIEmbeddings

from agents.doc_parser_agent import DocParserAgent
from config import settings

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(
    r"C:\Users\滨滨\Desktop\Knowledge-Base-Intelligent-Q-A-main\data\clean\institutional"
)
BENCH = ROOT / "data" / "eval" / "bench_v2.jsonl"

# ══════════════════════════════════════════════════════════
# BM25（生产可用：纯 Python，无依赖）
# ══════════════════════════════════════════════════════════
def tokenize(text: str) -> list[str]:
    """中文分词：字符 bigram + 单字（无需 jieba，召回优先）"""
    t = unicodedata.normalize("NFKC", text or "")
    t = re.sub(r"[^\w\u4e00-\u9fff]+", "", t)
    toks = [t[i : i + 2] for i in range(len(t) - 1)]
    toks.extend(t)
    return toks


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(d) for d in docs]
        self.N = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / max(self.N, 1)
        self.tf = [Counter(d) for d in self.docs]
        df = Counter()
        for d in self.docs:
            for w in set(d):
                df[w] += 1
        self.idf = {w: math.log(1 + (self.N - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def rank(self, query: str) -> list[int]:
        q = tokenize(query)
        scored = []
        for i in range(self.N):
            tf, dl = self.tf[i], len(self.docs[i])
            s = 0.0
            for w in q:
                f = tf.get(w)
                if not f:
                    continue
                s += self.idf.get(w, 0.0) * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                )
            scored.append((s, i))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [i for _, i in scored]


def rrf(rank_lists: list[list[int]], weights: list[float] | None = None, k: int = 60) -> dict[int, float]:
    w = weights or [1.0] * len(rank_lists)
    out: dict[int, float] = {}
    for wt, rl in zip(w, rank_lists):
        for rank, idx in enumerate(rl, 1):
            out[idx] = out.get(idx, 0.0) + wt / (k + rank)
    return out


# ══════════════════════════════════════════════════════════
def cos(a, b):
    d = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return d / (na * nb + 1e-9)


async def main():
    parser = DocParserAgent()
    docs: dict[str, dict] = {}
    for f in sorted(SRC.glob("*.md")):
        raw = f.read_text(encoding="utf-8", errors="ignore")
        dt, sections = parser.parse_tree(parser.strip_metadata(raw))
        if not dt:
            continue
        secs = [(s.title, s.body) for s in sections if s.body and len(s.body) >= 30]
        if secs:
            docs[dt] = {"file": f.name, "sections": secs}

    titles = list(docs.keys())
    # 章节级候选
    chunks = []
    for t in titles:
        for i, (st, bd) in enumerate(docs[t]["sections"]):
            chunks.append({
                "doc": t, "idx": i, "sec": st, "body": bd,
                "embed": f"{t}\n章节：{st}\n{bd}",
            })
    print(f"语料: {len(titles)} 文档 / {len(chunks)} 章节")

    emb = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.effective_embedding_api_key,
        base_url=settings.effective_embedding_base_url,
    )
    cvecs = await emb.aembed_documents([c["embed"] for c in chunks])
    bm25_chunk = BM25([c["embed"] for c in chunks])

    # 文档级向量 = 各章节向量的均值（简单且可复现）
    doc_vecs = {}
    for t in titles:
        idxs = [j for j, c in enumerate(chunks) if c["doc"] == t]
        n = len(idxs)
        dim = len(cvecs[0])
        doc_vecs[t] = [sum(cvecs[j][d] for j in idxs) / n for d in range(dim)]
    bm25_doc = BM25([f"{t}\n" + "\n".join(s for s, _ in docs[t]["sections"]) for t in titles])
    bm25_doc_index = {t: i for i, t in enumerate(titles)}
    doc_emb_index = {t: i for i, t in enumerate(titles)}

    # ── 读 benchmark ────────────────────────────────────
    bench = [json.loads(l) for l in BENCH.read_text(encoding="utf-8").splitlines() if l.strip()]
    doc_items = [r for r in bench if r["in_corpus"] and r["doc_title"] in docs]
    chunk_items = [r for r in doc_items if r["gold_chunk_index"] is not None and not r["ambiguous"]]
    print(f"评测集: 文档级 {len(doc_items)} 题 / chunk 级 {len(chunk_items)} 题")
    print()

    def gold_chunk_j(r):
        for j, c in enumerate(chunks):
            if c["doc"] == r["doc_title"] and c["idx"] == r["gold_chunk_index"]:
                return j
        return -1

    results = {}
    for qfield, label in (("question", "original"), ("question_leakfree", "leakfree")):
        acc = {m: {"doc": [], "chunk": []} for m in ("dense", "bm25", "hybrid", "hybrid_w")}
        for r in doc_items:
            q = r[qfield]
            qv = await emb.aembed_query(q)

            # 文档级：dense
            dsc = sorted(
                ((cos(qv, doc_vecs[t]), t) for t in titles), key=lambda x: -x[0]
            )
            d_dense = [t for _, t in dsc]
            d_bm = [titles[i] for i in bm25_doc.rank(q)]
            d_hyb = rrf(
                [[doc_emb_index[t] for t in d_dense], [bm25_doc_index[t] for t in d_bm]]
            )
            d_hyb_w = rrf(
                [[doc_emb_index[t] for t in d_dense], [bm25_doc_index[t] for t in d_bm]],
                weights=[0.3, 0.7],
            )
            for m, order in (
                ("dense", d_dense),
                ("bm25", d_bm),
                ("hybrid", [titles[i] for i, _ in sorted(d_hyb.items(), key=lambda x: -x[1])]),
                ("hybrid_w", [titles[i] for i, _ in sorted(d_hyb_w.items(), key=lambda x: -x[1])]),
            ):
                try:
                    acc[m]["doc"].append(order.index(r["doc_title"]) + 1)
                except ValueError:
                    acc[m]["doc"].append(999)

            # chunk 级
            gi = gold_chunk_j(r)
            csc = sorted(((cos(qv, v), j) for j, v in enumerate(cvecs)), key=lambda x: -x[0])
            c_dense = [j for _, j in csc]
            c_bm = bm25_chunk.rank(q)
            c_hyb = rrf([c_dense, c_bm])
            c_hyb_w = rrf([c_dense, c_bm], weights=[0.3, 0.7])
            if gi >= 0 and r in chunk_items:
                for m, order in (
                    ("dense", c_dense),
                    ("bm25", c_bm),
                    ("hybrid", [j for j, _ in sorted(c_hyb.items(), key=lambda x: -x[1])]),
                    ("hybrid_w", [j for j, _ in sorted(c_hyb_w.items(), key=lambda x: -x[1])]),
                ):
                    try:
                        acc[m]["chunk"].append(order.index(gi) + 1)
                    except ValueError:
                        acc[m]["chunk"].append(999)

        results[label] = acc

    # ── 输出 ────────────────────────────────────────────
    def metrics(ranks, k=5):
        n = len(ranks)
        if not n:
            return (0.0, 0.0, 0.0, 0.0)
        t1 = sum(1 for r in ranks if r == 1) / n
        r5 = sum(1 for r in ranks if r <= k) / n
        r10 = sum(1 for r in ranks if r <= 10) / n
        mrr = sum(1 / r for r in ranks if r < 999) / n
        return (t1, r5, r10, mrr)

    for label in ("original", "leakfree"):
        print("=" * 88)
        print(f"  问题集: {label}")
        print("=" * 88)
        print(f"{'方法':<14}{'Doc T1':>9}{'Doc R@5':>10}{'Doc MRR':>10}"
              f"{'Chunk T1':>10}{'Chunk R@5':>11}{'Chunk MRR':>11}")
        print("-" * 88)
        for m in ("dense", "bm25", "hybrid", "hybrid_w"):
            a = results[label][m]
            dt1, dr5, _, dmrr = metrics(a["doc"])
            ct1, cr5, _, cmrr = metrics(a["chunk"])
            print(f"{m:<14}{dt1:>9.1%}{dr5:>10.1%}{dmrr:>10.3f}"
                  f"{ct1:>10.1%}{cr5:>11.1%}{cmrr:>11.3f}")
        print()

    # 保存
    outp = ROOT / "data" / "eval" / "p1_results.json"
    ser = {
        lab: {m: {"doc": v["doc"], "chunk": v["chunk"]} for m, v in accs.items()}
        for lab, accs in results.items()
    }
    outp.write_text(json.dumps(ser, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"明细写入 {outp}")


asyncio.run(main())
