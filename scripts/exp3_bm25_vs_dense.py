"""
实验 3：BM25 稀疏检索 vs 稠密检索 vs RRF 融合

假设：判别性词项（"请假"、"年假"、"保密"）在稠密空间被稀释，
      BM25 能直接命中，融合后应显著改善。

只读实验，不改生产代码。
"""
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

import asyncio

from langchain_openai import OpenAIEmbeddings

from config import settings

SRC = Path(r"C:\Users\滨滨\Desktop\Knowledge-Base-Intelligent-Q-A-main\data\clean\institutional")

QUERIES = [
    ("公司的标准工作时间是什么时候到什么时候", "员工考勤管理制度", "工作时间"),
    ("请假要走什么流程", "员工考勤管理制度", "请假"),
    ("保密级别分几级", "保密管理制度", "密级"),
    ("出差住宿费能报多少", "差旅报销管理规定", "住宿"),
    ("加班费怎么算", "加班管理制度", "加班费"),
    ("年假有多少天", "年休假管理办法", "年假"),
    ("合同谁来审批", "合同管理办法", "审批"),
    ("客户资料能导出吗", "客户资料管理规定", "导出"),
]


# ── 极简中文 BM25（字符 bigram + 词项）─────────────────────
def tokenize(text: str) -> list[str]:
    """中文无空格：用字符 bigram + 单字，兼顾词项召回"""
    t = unicodedata.normalize("NFKC", text)
    t = re.sub(r"[^\w\u4e00-\u9fff]+", "", t)
    toks = []
    for i in range(len(t) - 1):
        toks.append(t[i : i + 2])
    # 加关键单字，提升"假"、"级"这类判别字权重
    toks.extend(list(t))
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
        self.idf = {
            w: math.log(1 + (self.N - c + 0.5) / (c + 0.5))
            for w, c in df.items()
        }

    def score(self, query: str, idx: int) -> float:
        q = tokenize(query)
        tf, dl = self.tf[idx], len(self.docs[idx])
        s = 0.0
        for w in q:
            if w not in tf:
                continue
            f = tf[w]
            s += self.idf.get(w, 0.0) * f * (self.k1 + 1) / (
                f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            )
        return s

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        scored = [(i, self.score(query, i)) for i in range(self.N)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


def rrf(rank_lists: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion"""
    out: dict[int, float] = {}
    for rl in rank_lists:
        for rank, idx in enumerate(rl, 1):
            out[idx] = out.get(idx, 0.0) + 1.0 / (k + rank)
    return out


async def main():
    # 构造章节级语料
    sys.path.insert(0, str(_PY))
    from agents.doc_parser_agent import DocParserAgent

    parser = DocParserAgent()
    items = []   # (doc_title, sec_title, body, embed_text)
    for f in sorted(SRC.glob("*.md")):
        raw = f.read_text(encoding="utf-8", errors="ignore")
        dt, sections = parser.parse_tree(parser.strip_metadata(raw))
        if not dt:
            continue
        for s in sections:
            if s.body and len(s.body) >= 30:
                items.append((
                    dt, s.title, s.body,
                    f"{dt}\n章节：{s.title}\n{s.body}",
                ))

    print(f"候选章节数: {len(items)}")
    print()

    # 稠密
    emb = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.effective_embedding_api_key,
        base_url=settings.effective_embedding_base_url,
    )
    vecs = await emb.aembed_documents([it[3] for it in items])

    # 稀疏（用 display 文本建 BM25）
    bm25 = BM25([it[3] for it in items])

    def gold_idx(doc, sec):
        for i, (t, st, bd, _) in enumerate(items):
            if t == doc and (sec in st or sec in bd[:80]):
                return i
        return -1

    rows = []
    for q, gdoc, gsec in QUERIES:
        gi = gold_idx(gdoc, gsec)
        qv = await emb.aembed_query(q)

        # 稠密排名
        dsc = sorted(
            ((sum(a * b for a, b in zip(qv, v)) /
              (math.sqrt(sum(a * a for a in qv)) * math.sqrt(sum(b * b for b in v)) + 1e-9), i)
             for i, v in enumerate(vecs)),
            key=lambda x: x[0], reverse=True,
        )
        drank = {i: r for r, (_, i) in enumerate(dsc, 1)}

        # 稀疏排名
        ssc = bm25.search(q, top_k=len(items))
        srank = {i: r for r, (i, _) in enumerate(ssc, 1)}

        # RRF 融合（关键：两路都必须传 int 索引，不能传 (idx, score) 元组）
        fused = rrf([[i for _, i in dsc[:50]], [i for i, _ in ssc[:50]]])
        frank = {i: r for r, (i, _) in enumerate(
            sorted(fused.items(), key=lambda x: x[1], reverse=True), 1)}

        # 诊断：gold 在两路的原始名次与 RRF 得分
        gold_rrf = fused.get(gi, 0.0)
        dense_rrf = 1.0 / (60 + drank.get(gi, 999)) if drank.get(gi, 999) <= 50 else 0.0
        sparse_rrf = 1.0 / (60 + srank.get(gi, 999)) if srank.get(gi, 999) <= 50 else 0.0
        if q == QUERIES[0][0] or q == QUERIES[1][0]:
            print(f"[诊断] {q}")
            print(f"   gold dense_rank={drank.get(gi)}  bm25_rank={srank.get(gi)}")
            print(f"   dense贡献={dense_rrf:.6f}  sparse贡献={sparse_rrf:.6f}  合计={gold_rrf:.6f}")
            top3 = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:3]
            for ii, ss in top3:
                print(f"   竞争者 idx={ii} rrf={ss:.6f} dense={drank.get(ii)} bm25={srank.get(ii)} "
                      f"{items[ii][0]}/{items[ii][1]}")
            print()

        rows.append({
            "q": q, "gold": gi,
            "dense": drank.get(gi, 999), "sparse": srank.get(gi, 999),
            "fused": frank.get(gi, 999),
        })

    print("=" * 84)
    print(f"{'query':<24}{'dense':>8}{'bm25':>8}{'RRF':>8}")
    print("=" * 84)
    for r in rows:
        def fmt(x):
            return f"{x}" if x <= 20 else ">20"
        print(f"{r['q'][:22]:<24}{fmt(r['dense']):>8}{fmt(r['sparse']):>8}{fmt(r['fused']):>8}")

    print("=" * 84)
    n = len(rows)
    for name, key in (("稠密", "dense"), ("BM25", "sparse"), ("RRF融合", "fused")):
        t1 = sum(1 for r in rows if r[key] == 1)
        r5 = sum(1 for r in rows if r[key] <= 5)
        mrr = sum(1 / r[key] for r in rows if r[key] < 999) / n
        print(f"  {name:<8} Top-1 {t1}/{n}   Recall@5 {r5}/{n}   MRR {mrr:.3f}")


asyncio.run(main())
