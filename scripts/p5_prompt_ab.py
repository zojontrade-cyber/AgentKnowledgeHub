"""
P5 生成 Prompt 结构 A/B：缓存命中 × 四项质量指标

背景
----
把生成 Prompt 从"意图插值进 system + 标签带分数/类型"改成
"稳定 system + 固定顺序动态段"（见 services/prompt_builder.py），
会同时影响缓存前缀复用与答案质量。只报缓存提升就是在拿质量冒险，
所以这里两者一起量。

⚠️ 为什么必须跑多遍（踩过的坑，务必保留）
----------------------------------------
第一版脚本对每种结构只跑一遍，结果 "改后" 的命中率从 85% 掉到 7.6%。
**那个结论是错的**：服务端缓存是长驻的，而 legacy 结构今天已经被发过几百次
（54 题基准跑了两遍 + 多个验证脚本），stable 结构一次都没发过。
冷缓存 vs 热缓存被误读成了"结构优劣"。

正确做法：每种结构先跑一遍**预热**，只把**最后一遍**计入对比。
`--passes` 默认 2，并把每一遍的命中率都打印出来，让冷/热差异可见。

这个坑同时**否证了此前"命中来自任意位置的缓存块"的推断**：
若是块级复用，stable 里同样的父文档文本应当照样命中；实测为 0，
说明命中确实依赖**从头开始的前缀匹配**。

用法
----
    python scripts/p5_prompt_ab.py --limit 18 --passes 2
    python scripts/p5_prompt_ab.py --mode stable --limit 18 --passes 3
"""

import argparse
import asyncio
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

_PY = Path(Path(__file__).resolve().parent.parent / "python")
os.chdir(_PY)
sys.path.insert(0, str(_PY))
sys.stdout.reconfigure(encoding="utf-8")

from openai import AsyncOpenAI  # noqa: E402

from agents.qa_agent import QAAgent  # noqa: E402
from config import settings  # noqa: E402
from services.database import init_db  # noqa: E402
from services.usage_meter import usage_meter  # noqa: E402
from services.vector_store import VectorStoreService  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "data" / "eval" / "bench_v4.jsonl"
OUT_DIR = ROOT / "data" / "eval"

JUDGE = """你是 RAG 系统的严格评测员。请对一次问答做三项判定。

【用户问题】
{question}

【标准答案】（来自知识库原文，是判定依据）
{gold}

【系统检索到的上下文】
{context}

【系统回答】
{answer}

【系统标注的引用】
{cites}

判定三项（每项 1-5 分 + 布尔结论）：

1. correctness：回答是否正确回应了问题？数字/时间/级别必须准确。
   只要覆盖问题所问的要点即算正确，表述不同不影响。

2. faithfulness：回答内容是否**完全来自**上述检索上下文？
   有没有编造上下文里没有的信息？

3. citation_correctness：标注的引用是否真正支撑回答？
   即该来源确实包含回答所依据的内容，且是应依据的章节。

只返回 JSON：
{{"correctness":分,"correct":true/false,
  "faithfulness":分,"faithful":true/false,
  "citation_correctness":分,"citation_ok":true/false,
  "reason":"一句话（无问题写 OK）"}}"""

JUDGE_NOANSWER = """你是一个严格的"拒答能力"评测员。

【用户问题】
{question}

【知识库中是否有答案】
**没有。知识库中不存在该问题的答案。**（原因：{why}）

【系统检索到的上下文】
{context}

【系统回答】
{answer}

请判断系统是否正确处理了这个"知识库无答案"的问题。

**正确行为**：明确表示无法回答 / 制度中未规定 / 未找到相关信息，
且**没有编造**任何具体的数字、天数、金额或条款。

**错误行为（幻觉）**：
  - 编造了具体规定（如"可以请 X 天病假"）
  - 声称制度中有该规定，并给出细节
  - 用常识推断但未声明"制度中未规定"

只返回 JSON：
{{"refused":true/false,
  "hallucinated":true/false,
  "score":1-5,
  "reason":"一句话说明"}}"""


def empty_stage() -> dict:
    return {"calls": 0, "prompt": 0, "hit": 0, "miss": 0}


async def run_pass(agent: QAAgent, bench: list[dict]) -> tuple[list[dict], dict, dict]:
    """跑一遍完整问答，返回 (rows, 轮级 totals, 按 stage 的缓存统计)"""
    by_stage: dict[str, dict] = defaultdict(empty_stage)
    totals = {"prompt": 0, "completion": 0, "hit": 0, "miss": 0,
              "stable": 0, "context": 0, "query": 0, "calls": 0}
    rows: list[dict] = []

    for r in bench:
        q = r["question"]
        try:
            with usage_meter() as rec:
                res = await agent.answer(q, acl_scopes=["public"])
            usage = rec.summary()
        except Exception as e:
            rows.append({"question": q, "qtype": r["qtype"],
                         "answerable": r["answerable"], "error": str(e)[:200],
                         "usage": {}, "answer": "", "context": "", "cites": "",
                         "gold": r.get("expected_answer") or "",
                         "why": r.get("why_not_in_corpus") or ""})
            continue

        ctx_txt = "\n\n".join(
            f"[{(c.metadata or {}).get('title','')} > "
            f"{(c.metadata or {}).get('section_title','')}]"
            f"\n{(c.llm_context or c.content)[:800]}"
            for c in res.contexts[:5]
        )
        answer = res.answer or ""
        cites = sorted(set(re.findall(r"\[来源[:：]\s*([^\]]+)\]", answer)))

        rows.append({
            "question": q, "qtype": r["qtype"], "answerable": r["answerable"],
            "gold": r.get("expected_answer") or "",
            "why": r.get("why_not_in_corpus") or "",
            "answer": answer, "context": ctx_txt, "cites": "; ".join(cites),
            "usage": usage,
        })

        for c in usage.get("per_call") or []:
            st = by_stage[c["stage"]]
            st["calls"] += 1
            st["prompt"] += c["prompt_tokens"]
            st["hit"] += c["cache_hit_tokens"]
            st["miss"] += c["cache_miss_tokens"]
        totals["prompt"] += usage.get("prompt_tokens", 0)
        totals["completion"] += usage.get("completion_tokens", 0)
        totals["hit"] += usage.get("cache_hit_tokens", 0)
        totals["miss"] += usage.get("cache_miss_tokens", 0)
        totals["stable"] += usage.get("stable_prefix_tokens", 0) or 0
        totals["context"] += usage.get("context_tokens", 0) or 0
        totals["query"] += usage.get("query_tokens", 0) or 0
        totals["calls"] += usage.get("llm_calls", 0)

    return rows, totals, by_stage


async def judge_rows(rows: list[dict]) -> None:
    client = AsyncOpenAI(
        api_key=settings.openai_api_key, base_url=settings.openai_base_url,
        timeout=settings.llm_timeout_seconds,
    )
    sem = asyncio.Semaphore(4)

    async def judge(row: dict) -> dict:
        if row.get("error"):
            return {"reason": "ERR generation"}
        async with sem:
            try:
                if row["answerable"]:
                    prompt = JUDGE.format(
                        question=row["question"], gold=row["gold"][:1000],
                        context=row["context"][:2000], answer=row["answer"][:1200],
                        cites=row["cites"][:300])
                else:
                    prompt = JUDGE_NOANSWER.format(
                        question=row["question"], why=row["why"],
                        context=row["context"][:1500], answer=row["answer"][:1200])
                rr = await client.chat.completions.create(
                    model=settings.openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0, response_format={"type": "json_object"},
                )
                return json.loads(rr.choices[0].message.content or "{}")
            except Exception as e:
                return {"reason": f"ERR {type(e).__name__}"}

    verdicts = await asyncio.gather(*(judge(r) for r in rows))
    for r, v in zip(rows, verdicts):
        r["verdict"] = v


def summarize(rows: list[dict], totals: dict, by_stage: dict,
              recall_by_type: dict) -> dict:
    ans_rows = [r for r in rows if r["answerable"] and not r.get("error")]
    na_rows = [r for r in rows if not r["answerable"] and not r.get("error")]

    def rate(subset, key):
        vals = [r for r in subset if r["verdict"].get(key) is not None]
        ok = sum(1 for r in vals if r["verdict"].get(key))
        return (ok / len(vals) if vals else 0.0)

    def avg(subset, key):
        v = [r["verdict"].get(key) for r in subset
             if isinstance(r["verdict"].get(key), int)]
        return sum(v) / len(v) if v else 0.0

    denom = totals["hit"] + totals["miss"]
    stable_reuse = (min(totals["hit"], totals["stable"]) / totals["stable"]
                    if totals["stable"] else None)
    ctx_reuse = (min(max(0, totals["hit"] - totals["stable"]), totals["context"])
                 / totals["context"] if totals["context"] else None)
    rh = sum(v[0] for v in recall_by_type.values())
    rt = sum(v[1] for v in recall_by_type.values())

    return {
        "n": len(rows),
        "recall@5": rh / rt if rt else 0.0,
        "faithfulness": rate(ans_rows, "faithful"),
        "answer_correctness": rate(ans_rows, "correct"),
        "citation_correctness": rate(ans_rows, "citation_ok"),
        "refusal_rate": rate(na_rows, "refused"),
        "hallucination_rate": rate(na_rows, "hallucinated"),
        "avg_scores": {
            "faithfulness": avg(ans_rows, "faithfulness"),
            "correctness": avg(ans_rows, "correctness"),
            "citation_correctness": avg(ans_rows, "citation_correctness"),
        },
        "cache": {
            "prompt_tokens": totals["prompt"],
            "completion_tokens": totals["completion"],
            "cache_hit_tokens": totals["hit"],
            "cache_miss_tokens": totals["miss"],
            "cache_hit_rate": totals["hit"] / denom if denom else 0.0,
            "stable_prefix_tokens": totals["stable"],
            "context_tokens": totals["context"],
            "query_tokens": totals["query"],
            "stable_prefix_reuse_rate": stable_reuse,
            "rag_context_reuse_rate": ctx_reuse,
            "llm_calls": totals["calls"],
        },
        "by_stage": dict(by_stage),
        "recall_by_type": dict(recall_by_type),
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="both", choices=["both", "stable", "legacy"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--passes", type=int, default=2,
                    help="每结构跑几遍；只有最后一遍计入对比（前几遍是预热）")
    args = ap.parse_args()

    init_db(settings.sqlite_path)
    vs = VectorStoreService()
    await vs.init()

    bench = [
        json.loads(line)
        for line in BENCH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        bench = bench[: args.limit]

    modes = ["legacy", "stable"] if args.mode == "both" else [args.mode]
    results: dict[str, dict] = {}

    for mode in modes:
        print()
        print("=" * 96)
        print(f"Prompt 结构 = {mode}   题目 = {len(bench)}   passes = {args.passes}   "
              f"模型 = {settings.openai_model}")
        print("=" * 96)

        settings.answer_prompt_mode = mode
        agent = QAAgent(vector_store=vs)

        # Recall@5 是纯检索指标，与生成 Prompt 无关 —— 只算一次
        recall_by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for r in bench:
            if not r.get("answerable"):
                continue
            gold = set(r.get("doc_titles") or
                       ([r["doc_title"]] if r.get("doc_title") else []))
            res0 = await vs.search(r["question"], top_k=5)
            found = any((d.get("metadata") or {}).get("title") in gold for d, _ in res0)
            recall_by_type[r["qtype"]][1] += 1
            if found:
                recall_by_type[r["qtype"]][0] += 1

        pass_stats: list[dict] = []
        last_rows: list[dict] = []
        last_totals: dict = {}
        last_stage: dict = {}

        for p in range(1, args.passes + 1):
            tag = "冷缓存（预热）" if p == 1 else f"第 {p} 遍"
            rows, totals, by_stage = await run_pass(agent, bench)
            denom = totals["hit"] + totals["miss"]
            rate = totals["hit"] / denom if denom else 0.0
            print(f"  pass {p} [{tag}]  calls={totals['calls']} "
                  f"prompt={totals['prompt']:,} hit={totals['hit']:,} "
                  f"miss={totals['miss']:,} 命中率={rate:.1%}")
            pass_stats.append({"pass": p, "cache_hit_rate": rate,
                               "hit": totals["hit"], "prompt": totals["prompt"],
                               "stable": totals["stable"], "context": totals["context"]})
            if p == args.passes:
                last_rows, last_totals, last_stage = rows, totals, by_stage

        print(f"  LLM-as-Judge（计入对比的最后一遍，{len(last_rows)} 题）…")
        await judge_rows(last_rows)

        summary = summarize(last_rows, last_totals, last_stage, recall_by_type)
        summary["mode"] = mode
        summary["passes"] = pass_stats
        summary["rows"] = last_rows
        results[mode] = summary

        out = OUT_DIR / f"p5_prompt_ab_{mode}.json"
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"  写入 {out}")

    if len(results) == 2:
        print()
        print("=" * 96)
        print("对比：legacy（改前） vs stable（改后）—— 均取预热后的最后一遍")
        print("=" * 96)
        L, S = results["legacy"], results["stable"]
        print(f"  {'指标':<32}{'legacy':>14}{'stable':>14}{'差异':>14}")
        print("-" * 96)

        def line(label, a, b, pct=True, lower_better=False):
            a = a or 0.0
            b = b or 0.0
            if pct:
                print(f"  {label:<32}{a:>13.1%}{b:>14.1%}"
                      f"{(b - a) * 100:>+13.1f}pp")
            else:
                print(f"  {label:<32}{a:>14.2f}{b:>14.2f}{b - a:>+14.2f}")

        line("Recall@5", L["recall@5"], S["recall@5"])
        line("Faithfulness", L["faithfulness"], S["faithfulness"])
        line("Answer Correctness", L["answer_correctness"], S["answer_correctness"])
        line("Citation Correctness", L["citation_correctness"], S["citation_correctness"])
        line("平均分 Faithfulness", L["avg_scores"]["faithfulness"],
             S["avg_scores"]["faithfulness"], pct=False)
        line("平均分 Correctness", L["avg_scores"]["correctness"],
             S["avg_scores"]["correctness"], pct=False)
        line("平均分 Citation", L["avg_scores"]["citation_correctness"],
             S["avg_scores"]["citation_correctness"], pct=False)
        line("拒答率（越高越好）", L["refusal_rate"], S["refusal_rate"])
        line("幻觉率（越低越好）", L["hallucination_rate"], S["hallucination_rate"])
        print("-" * 96)
        line("缓存① 总命中率", L["cache"]["cache_hit_rate"], S["cache"]["cache_hit_rate"])
        line("缓存② 稳定前缀复用", L["cache"]["stable_prefix_reuse_rate"],
             S["cache"]["stable_prefix_reuse_rate"])
        line("缓存③ 检索上下文复用", L["cache"]["rag_context_reuse_rate"],
             S["cache"]["rag_context_reuse_rate"])
        print(f"  {'稳定前缀 token 总量':<32}"
              f"{L['cache']['stable_prefix_tokens']:>14,}"
              f"{S['cache']['stable_prefix_tokens']:>14,}"
              f"{S['cache']['stable_prefix_tokens'] - L['cache']['stable_prefix_tokens']:>+14,}")

        print()
        print("  按 stage（最后一遍）")
        for m, res in (("legacy", L), ("stable", S)):
            print(f"    [{m}]")
            for st, d in sorted(res["by_stage"].items(),
                                key=lambda kv: -kv[1]["prompt"]):
                dd = d["hit"] + d["miss"]
                print(f"      {st:<12} calls={d['calls']:<4} prompt={d['prompt']:>7,} "
                      f"hit={d['hit']:>7,} miss={d['miss']:>7,} "
                      f"命中率={d['hit'] / dd if dd else 0:>6.1%}")

        (OUT_DIR / "p5_prompt_ab_compare.json").write_text(
            json.dumps({m: {k: v for k, v in r.items() if k != "rows"}
                        for m, r in results.items()},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n  对比写入 {OUT_DIR / 'p5_prompt_ab_compare.json'}")


asyncio.run(main())
