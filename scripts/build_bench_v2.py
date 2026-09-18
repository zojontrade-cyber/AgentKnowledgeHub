"""
构建 retrieval-only benchmark（冻结变量、可复现）

用户要求：
  - retrieval-only（不调 LLM 生成，不受 reranker 干扰）
  - 文档级 + chunk 级双指标
  - 默认无 reranker

关键修正（原 eval_set.jsonl 的三个缺陷）：
  1. 无 chunk 级 ground truth  -> 本脚本用 expected_answer 反查源章节生成
  2. 15/78 问题含文档标题核心词 -> 生成 leak-free 问题变体，两套都测
  3. 10 组答案重复/错配        -> 标记 ambiguous，单列指标

输出：
  data/eval/bench_v2.jsonl  每行:
    question, question_leakfree, doc_title, file, section_title,
    gold_doc_id, gold_chunk_index, expected_answer, in_corpus, ambiguous
"""

import json
import os
import re
import sys
import unicodedata
from pathlib import Path

_PY = Path(__file__).resolve().parent.parent / "python"
os.chdir(_PY)
sys.path.insert(0, str(_PY))
sys.stdout.reconfigure(encoding="utf-8")

from agents.doc_parser_agent import DocParserAgent

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(
    r"C:\Users\滨滨\Desktop\Knowledge-Base-Intelligent-Q-A-main\data\clean\institutional"
)
OUT = ROOT / "data" / "eval" / "bench_v2.jsonl"

# 文档标题的通用后缀 —— 用于剥离"话题词泄漏"
_SUFFIXES = [
    "管理规定", "管理办法", "管理制度", "管理规范", "管理细则", "管理程序",
    "制度", "办法", "规定", "规范", "细则", "程序", "方案", "预案",
]


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    return re.sub(r"[\s\u3000]+", "", s)


def core_title(t: str) -> str:
    """剥掉通用后缀，得到话题核心词"""
    t = norm(t)
    for suf in _SUFFIXES:
        if t.endswith(suf) and len(t) > len(suf):
            t = t[: -len(suf)]
            break
    return t


def main() -> None:
    parser = DocParserAgent()

    # ── 1. 解析全部源文档 ──────────────────────────────
    docs: dict[str, dict] = {}
    for f in sorted(SRC.glob("*.md")):
        raw = f.read_text(encoding="utf-8", errors="ignore")
        dt, sections = parser.parse_tree(parser.strip_metadata(raw))
        if not dt:
            continue
        secs = [(s.title, s.body) for s in sections if s.body and len(s.body) >= 30]
        if secs:
            docs[dt] = {"file": f.name, "sections": secs}

    print(f"解析文档 {len(docs)} 篇，章节 {sum(len(d['sections']) for d in docs.values())} 个")

    # ── 2. 读原 eval set ───────────────────────────────
    src_eval = ROOT / "data" / "eval" / "eval_set.jsonl"
    rows = [
        json.loads(l)
        for l in src_eval.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

    # 重复答案标记（同一 expected_answer 出现多次 → chunk 级 gold 不可靠）
    ans_count: dict[str, int] = {}
    for r in rows:
        a = norm(str(r.get("expected_answer") or ""))
        if a:
            ans_count[a] = ans_count.get(a, 0) + 1

    out = []
    stats = {"leak": 0, "chunk_gold_ok": 0, "chunk_gold_fail": 0, "ambiguous": 0}

    for r in rows:
        q = r["question"]
        ans = str(r.get("expected_answer") or "")
        title = r.get("expected_title") or ""
        in_corpus = bool(r.get("in_corpus"))

        rec = {
            "question": q,
            "expected_answer": ans,
            "doc_title": title,
            "expected_source": r.get("expected_source") or "",
            "in_corpus": in_corpus,
            "section_title": None,
            "gold_chunk_index": None,
            "ambiguous": False,
            "question_leakfree": q,
        }

        if in_corpus and ans and title in docs:
            # ── 生成 leak-free 问题 ─────────────────────
            core = core_title(title)
            if core and core in q:
                stats["leak"] += 1
                # 用中性替换把话题词抹掉，保留问句意图
                rec["question_leakfree"] = q.replace(core, "公司").replace("公司公司", "公司")
                rec["had_leak"] = True

            # ── 反查 chunk 级 gold ─────────────────────
            na = norm(ans)
            key = na[:40]
            best_i, best_len = None, 0
            for i, (st, bd) in enumerate(docs[title]["sections"]):
                nb = norm(bd)
                # 用答案前 40 字做子串匹配（答案通常直接摘自章节）
                if key and key in nb:
                    if len(nb) > best_len:
                        best_i, best_len = i, len(nb)
            if best_i is not None:
                rec["section_title"] = docs[title]["sections"][best_i][0]
                rec["gold_chunk_index"] = best_i
                stats["chunk_gold_ok"] += 1
            else:
                # 退化：按答案字符重叠度取最高章节
                best_ov, best_i = 0.0, None
                aset = set(na)
                for i, (st, bd) in enumerate(docs[title]["sections"]):
                    bset = set(norm(bd))
                    if not bset:
                        continue
                    ov = len(aset & bset) / max(len(aset), 1)
                    if ov > best_ov:
                        best_ov, best_i = ov, i
                if best_i is not None and best_ov >= 0.6:
                    rec["section_title"] = docs[title]["sections"][best_i][0]
                    rec["gold_chunk_index"] = best_i
                    stats["chunk_gold_ok"] += 1
                else:
                    stats["chunk_gold_fail"] += 1

            if ans_count.get(na, 0) > 1:
                rec["ambiguous"] = True
                stats["ambiguous"] += 1

        out.append(rec)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for rec in out:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print()
    print("=" * 70)
    print(f"写入 {OUT}")
    print(f"  总题数              {len(out)}")
    print(f"  有 chunk 级 gold    {stats['chunk_gold_ok']}")
    print(f"  无 chunk 级 gold    {stats['chunk_gold_fail']}  (仅参与文档级指标)")
    print(f"  答案歧义(重复)      {stats['ambiguous']}  (单列，不污染主指标)")
    print(f"  话题词泄漏          {stats['leak']}  (已生成 leakfree 变体)")
    print(f"  库外问题(应拒答)    {sum(1 for r in out if not r['in_corpus'])}")
    print("=" * 70)

    # 可评测子集
    evaluable = [
        r for r in out
        if r["in_corpus"] and r["doc_title"] in docs
    ]
    chunk_eval = [r for r in evaluable if r["gold_chunk_index"] is not None
                  and not r["ambiguous"]]
    print(f"  文档级可评测        {len(evaluable)}")
    print(f"  chunk 级可评测      {len(chunk_eval)}  (严格子集，无歧义)")


if __name__ == "__main__":
    main()
