"""Pipeline vs ReAct 四项指标 A/B 对比。"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
E = ROOT / "data" / "eval"

A = json.loads((E / "rag_four_metrics_v4_pipeline.json").read_text(encoding="utf-8"))
B = json.loads((E / "rag_four_metrics_v4_react.json").read_text(encoding="utf-8"))


def rate(rows, key, qtype=None):
    sub = [r for r in rows if r["answerable"] and (qtype is None or r["qtype"] == qtype)]
    vals = [r for r in sub if r["verdict"].get(key) is not None]
    if not vals:
        return None
    return sum(1 for r in vals if r["verdict"].get(key)) / len(vals)


print("=" * 88)
print("Pipeline  vs  ReAct   （54 题六类型，同一会话背靠背运行）")
print("=" * 88)
print()
print(f"  {'指标':<26}{'Pipeline':>12}{'ReAct':>12}{'差异':>12}")
print("-" * 88)


def line(label, av, bv, pct=True):
    if av is None or bv is None:
        return
    if pct:
        print(f"  {label:<26}{av:>11.1%}{bv:>12.1%}{bv-av:>+12.1%}")
    else:
        print(f"  {label:<26}{av:>11.2f}{bv:>12.2f}{bv-av:>+12.2f}")


line("Recall@5", A["recall@5"], B["recall@5"])
line("Faithfulness", A["faithfulness"], B["faithfulness"])
line("Answer Correctness", A["answer_correctness"], B["answer_correctness"])
line("Citation Correctness", A["citation_correctness"], B["citation_correctness"])
line("拒答率（9 题无答案）", A["refusal_rate"], B["refusal_rate"])
print()
line("Faithfulness 均分", A["avg_scores"]["faithfulness"], B["avg_scores"]["faithfulness"], pct=False)
line("Correctness 均分", A["avg_scores"]["correctness"], B["avg_scores"]["correctness"], pct=False)
line("Citation 均分", A["avg_scores"]["citation_correctness"], B["avg_scores"]["citation_correctness"], pct=False)

print()
print("=" * 88)
print("按题型分解")
print("=" * 88)
for metric, label in (("correct", "Correctness"), ("faithful", "Faithfulness"), ("citation_ok", "Citation")):
    print(f"\n  【{label}】")
    print(f"    {'题型':<10}{'Pipeline':>12}{'ReAct':>12}{'差异':>12}")
    for t in ("single", "multi", "multihop", "summary", "compare"):
        av = rate(A["rows"], metric, t)
        bv = rate(B["rows"], metric, t)
        if av is None or bv is None:
            continue
        d = bv - av
        mark = "  ←" if abs(d) >= 0.10 else ""
        print(f"    {t:<10}{av:>11.1%}{bv:>12.1%}{d:>+12.1%}{mark}")

print()
print("=" * 88)
print("ReAct 工具使用（Pipeline 不具备的能力）")
print("=" * 88)
from collections import defaultdict
tot = defaultdict(int)
for r in B["rows"]:
    for k, v in (r.get("tool_use") or {}).items():
        tot[k] += v
s = sum(tot.values()) or 1
print(f"  总调用 {s} 次 / {len(B['rows'])} 题 = {s/len(B['rows']):.1f} 次/题")
for k, v in sorted(tot.items(), key=lambda x: -x[1]):
    print(f"    {k:<16}{v:>5}  ({v/s:.1%})")
print(f"  Pipeline 固定调用: 每轮 4 个检索器（按静态计划），无自主选择")
