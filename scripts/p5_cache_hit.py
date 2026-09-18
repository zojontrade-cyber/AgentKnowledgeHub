"""
P5 实测：QA 链路的 token 消耗与缓存命中

为什么需要单独实测
------------------
`services/usage_meter.py` 只是**把数字采出来**，它不回答"命中率是多少"。
而"缓存命中情况"这个问题只能靠真实调用回答 —— 所以本脚本跑真实 QA 链路，
把每次 LLM 调用的 prompt/completion/hit/miss 按 stage 展开后落盘。

两个必须分开的概念
------------------
    cache_hit_rate   = hit / (hit + miss)      ← 可测 Prompt Token 的命中占比
    cache_hit_calls  = 命中(hit>0)的调用次数     ← 调用次数比例

例：0/1000 与 900/1000 两次调用 -> Token 命中率 45%，但只有 1/2 次调用命中。
本脚本两个都打印，不混为一谈。

关于结论的适用范围
------------------
本脚本的观测结论**只在当前模型 / Provider / 请求配置下成立**，
不构成"DeepSeek 缓存机制如何如何"的通用规律。

用法
----
    python scripts/p5_cache_hit.py                  # 全量 54 题
    python scripts/p5_cache_hit.py --limit 10       # 子集（省钱）
    python scripts/p5_cache_hit.py --mode react
"""

import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

_PY = Path(Path(__file__).resolve().parent.parent / "python")
os.chdir(_PY)
sys.path.insert(0, str(_PY))
sys.stdout.reconfigure(encoding="utf-8")

from config import settings  # noqa: E402
from services.database import init_db  # noqa: E402
from services.usage_meter import usage_meter  # noqa: E402
from services.vector_store import VectorStoreService  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "data" / "eval" / "bench_v4.jsonl"
# 输出按模式分文件：否则跑一次 react 会把 pipeline 的实测结果覆盖掉
OUT_DIR = ROOT / "data" / "eval"


def base_stage(s: str) -> str:
    """react_step2 -> react（聚合时按基名分组，明细里保留原名）"""
    if "_step" in s:
        head, _, tail = s.rpartition("_step")
        if tail.isdigit():
            return head
    return s


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="pipeline", choices=["pipeline", "react"])
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（0=全量）")
    args = ap.parse_args()

    OUT = OUT_DIR / (
        "p5_cache_hit.json" if args.mode == "pipeline" else f"p5_cache_hit_{args.mode}.json"
    )

    init_db(settings.sqlite_path)
    vs = VectorStoreService()
    await vs.init()

    if args.mode == "react":
        from agents.react_qa_agent import ReactQAAgent as Agent

        agent = Agent(vector_store=vs, max_steps=settings.qa_max_steps)
    else:
        from agents.qa_agent import QAAgent as Agent

        agent = Agent(vector_store=vs)

    bench = [
        json.loads(line)
        for line in BENCH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        bench = bench[: args.limit]

    print("=" * 88)
    print("P5 实测：QA 链路的 token 消耗与缓存命中")
    print("=" * 88)
    print(f"  模式        {args.mode}")
    print(f"  模型        {settings.openai_model}")
    print(f"  base_url    {settings.openai_base_url}")
    print(f"  题目        {len(bench)} 题")
    print()
    print("  注：以下结论只代表**当前模型 / Provider / 请求配置**下的实测结果。")
    print()

    rows: list[dict] = []
    for i, r in enumerate(bench, 1):
        q = r["question"]
        try:
            with usage_meter() as rec:
                await agent.answer(q, acl_scopes=["public"])
            u = rec.summary()
            u["qtype"] = r.get("qtype", "")
            rows.append(u)
            n_meas = u["cache_measured_calls"]
            rate = pct(u["cache_hit_rate"]) if (u["cache_hit_tokens"] + u["cache_miss_tokens"]) else "—"
            print(
                f"  {i:>2}. [{u['qtype']:<8}] calls={u['llm_calls']} "
                f"tokens={u['total_tokens']:>6} "
                f"hit={u['cache_hit_tokens']:>5} miss={u['cache_miss_tokens']:>6} "
                f"rate={rate:>6} 可测={n_meas}/{u['llm_calls']}  {q[:26]}"
            )
        except Exception as e:
            rows.append({"qtype": r.get("qtype", ""), "error": f"{type(e).__name__}: {e}",
                         "llm_calls": 0, "total_tokens": 0,
                         "cache_hit_tokens": 0, "cache_miss_tokens": 0,
                         "cache_measured_calls": 0, "cache_hit_calls": 0,
                         "per_call": []})
            print(f"  {i:>2}. 失败 {type(e).__name__}: {str(e)[:70]}")

    # ── 轮级汇总 ─────────────────────────────────────────
    ok = [r for r in rows if not r.get("error")]
    tot_calls = sum(r["llm_calls"] for r in ok)
    tot_prompt = sum(r["prompt_tokens"] for r in ok)
    tot_completion = sum(r["completion_tokens"] for r in ok)
    tot_hit = sum(r["cache_hit_tokens"] for r in ok)
    tot_miss = sum(r["cache_miss_tokens"] for r in ok)
    meas_calls = sum(r["cache_measured_calls"] for r in ok)
    hit_calls = sum(r["cache_hit_calls"] for r in ok)
    denom = tot_hit + tot_miss

    print()
    print("=" * 88)
    print("轮级汇总")
    print("=" * 88)
    print(f"  成功题目          {len(ok)}/{len(rows)}")
    print(f"  LLM 调用总数      {tot_calls}  （{tot_calls / max(len(ok), 1):.2f} 次/题）")
    print(f"  Prompt tokens     {tot_prompt:,}")
    print(f"  Completion tokens {tot_completion:,}")
    print(f"  可测缓存的调用    {meas_calls}/{tot_calls}")
    print()
    print("  缓存命中（Token 口径，只在可测调用内计算）")
    print(f"    命中 token      {tot_hit:,}")
    print(f"    未命中 token    {tot_miss:,}")
    print(f"    命中率          {pct(tot_hit / denom) if denom else '—'}   = Hit / (Hit + Miss)")
    print("  缓存命中（调用次数口径，另一个概念）")
    print(f"    命中调用数      {hit_calls}/{meas_calls}"
          f"{'  = ' + pct(hit_calls / meas_calls) if meas_calls else ''}")

    # ── 按 stage 展开 ────────────────────────────────────
    by_stage: dict[str, dict[str, int]] = defaultdict(
        lambda: {"calls": 0, "prompt": 0, "completion": 0, "hit": 0, "miss": 0,
                 "meas": 0, "hit_calls": 0}
    )
    for r in ok:
        for c in r.get("per_call") or []:
            s = by_stage[base_stage(c.get("stage") or "unknown")]
            s["calls"] += 1
            s["prompt"] += c["prompt_tokens"]
            s["completion"] += c["completion_tokens"]
            s["meas"] += 1 if c["cache_supported"] else 0
            s["hit"] += c["cache_hit_tokens"]
            s["miss"] += c["cache_miss_tokens"]
            s["hit_calls"] += 1 if c["cache_hit_tokens"] > 0 else 0

    print()
    print("=" * 88)
    print("按 stage 展开（token 异常可定位到具体步骤）")
    print("=" * 88)
    print(f"  {'stage':<12}{'调用':>6}{'Prompt':>10}{'Completion':>12}"
          f"{'命中':>8}{'未命中':>9}{'命中率':>9}")
    for name in sorted(by_stage, key=lambda k: -by_stage[k]["prompt"]):
        s = by_stage[name]
        d = s["hit"] + s["miss"]
        print(f"  {name:<12}{s['calls']:>6}{s['prompt']:>10,}{s['completion']:>12,}"
              f"{s['hit']:>8,}{s['miss']:>9,}{(pct(s['hit'] / d) if d else '—'):>9}")

    # ── 按题型 ───────────────────────────────────────────
    by_type: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "tokens": 0, "calls": 0, "hit": 0, "miss": 0}
    )
    for r in ok:
        t = by_type[r.get("qtype") or "?"]
        t["n"] += 1
        t["tokens"] += r["total_tokens"]
        t["calls"] += r["llm_calls"]
        t["hit"] += r["cache_hit_tokens"]
        t["miss"] += r["cache_miss_tokens"]
    print()
    print("=" * 88)
    print("按题型")
    print("=" * 88)
    for name in sorted(by_type):
        t = by_type[name]
        d = t["hit"] + t["miss"]
        print(f"  {name:<10} n={t['n']:<3} tokens/题={t['tokens'] / max(t['n'], 1):>7.0f}"
              f"  calls/题={t['calls'] / max(t['n'], 1):>4.1f}"
              f"  命中率={pct(t['hit'] / d) if d else '—':>7}")

    # ── 最贵的题（定位异常）─────────────────────────────
    print()
    print("=" * 88)
    print("Token 消耗最高的 5 题")
    print("=" * 88)
    for r in sorted(ok, key=lambda x: -x["total_tokens"])[:5]:
        print(f"  {r['total_tokens']:>7,} tokens  calls={r['llm_calls']}  "
              f"{[c['stage'] + ':' + str(c['prompt_tokens']) for c in (r.get('per_call') or [])]}")

    OUT.write_text(
        json.dumps(
            {
                "config": {
                    "mode": args.mode,
                    "model": settings.openai_model,
                    "base_url": settings.openai_base_url,
                    "questions": len(bench),
                    "retrieval_mode": settings.retrieval_mode,
                    "rerank_enabled": settings.rerank_enabled,
                },
                "scope_note": (
                    "以下数字只在本次运行的模型 / Provider / 请求配置下成立，"
                    "不代表该 Provider 缓存机制的通用行为。"
                ),
                "totals": {
                    "ok": len(ok), "llm_calls": tot_calls,
                    "prompt_tokens": tot_prompt, "completion_tokens": tot_completion,
                    "cache_hit_tokens": tot_hit, "cache_miss_tokens": tot_miss,
                    "cache_measured_calls": meas_calls, "cache_hit_calls": hit_calls,
                    "cache_hit_rate": round(tot_hit / denom, 6) if denom else 0.0,
                },
                "by_stage": by_stage,
                "by_qtype": by_type,
                "per_question": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n  明细写入 {OUT}")


asyncio.run(main())
