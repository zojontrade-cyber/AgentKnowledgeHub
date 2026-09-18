"""
P4-D 前置测量：DeepSeek 上下文缓存的实际命中情况

背景（来自 DeepSeek 官方文档）
------------------------------
DeepSeek 自 2024-08 起提供**自动**磁盘上下文缓存，无需改代码：

  - 只有**从第 0 个 token 起完全相同的前缀**才算缓存命中
  - 缓存命中 $0.014/M tokens，未命中 $0.14/M  -> 命中省 90%
  - 响应 usage 中提供 prompt_cache_hit_tokens / prompt_cache_miss_tokens
  - 缓存条目在几小时到几天内自动清除
  - 缓存以 64 token 为存储单元，不足 64 token 不缓存

我们的抽取负载是官方列出的**典型受益场景**：
    "Q&A assistants with long preset prompts"
    同一段 675 字符 system prompt × 159 次调用

因此本脚本**直接测量真实命中率**，而不是假设。
若命中率高，则无需压缩 prompt —— 成本已经很低。

方法：连续发 N 次「相同 system prompt + 不同 chunk」的请求，
      读取每次的 cache hit/miss token 数。
"""

import asyncio
import json
import os
import sys
from pathlib import Path

_PY = Path(Path(__file__).resolve().parent.parent / "python")
os.chdir(_PY)
sys.path.insert(0, str(_PY))
sys.stdout.reconfigure(encoding="utf-8")

import httpx

from agents.knowledge_extract_agent import EXTRACTION_SYSTEM_PROMPT
from config import settings

N_CALLS = 8


async def one(cli, chunk_text: str, idx: int):
    """发一次抽取请求，返回 usage 中的缓存字段"""
    r = await cli.post(
        f"{settings.openai_base_url}/chat/completions",
        headers={"Authorization": f"Bearer {settings.openai_api_key}"},
        json={
            "model": settings.openai_model,
            "messages": [
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": f"请从以下文本中抽取知识：\n\n{chunk_text}"},
            ],
            "temperature": 0,
        },
    )
    r.raise_for_status()
    d = r.json()
    u = d.get("usage", {}) or {}
    return {
        "idx": idx,
        "prompt_tokens": u.get("prompt_tokens"),
        "hit": u.get("prompt_cache_hit_tokens"),
        "miss": u.get("prompt_cache_miss_tokens"),
        "completion": u.get("completion_tokens"),
    }


async def main():
    print("=" * 84)
    print("DeepSeek 上下文缓存命中测量")
    print("=" * 84)
    print(f"  system prompt 长度: {len(EXTRACTION_SYSTEM_PROMPT)} 字符")
    print(f"  模型: {settings.openai_model}")
    print(f"  连续发 {N_CALLS} 次（相同 system prompt + 不同 chunk）")
    print()

    chunks = [
        "## 2. 工作时间\n\n标准工作时间为周一至周五 9:00 至 18:00，午休 12:00 至 13:00。",
        "## 3. 打卡与迟到早退\n\n员工应在上班前和下班后各完成一次打卡。",
        "## 4. 请假流程\n\n请假使用 OA 系统填写申请单，按审批权限提交审批。",
        "## 5. 外勤与出差\n\n外勤前应在考勤系统中登记地点、事由和预计返回时间。",
        "## 6. 异常处理\n\n考勤记录与员工确认不一致时，应在次月 5 个工作日内提出申诉。",
        "## 2. 密级定义\n\n密级分为公开、内部、机密、绝密四级。",
        "## 3. 费用标准\n\n一线城市住宿 500 元/晚，省会城市 400，其他城市 300。",
        "## 4. 报销流程\n\n出差结束后 10 个工作日内提交报销单。",
    ]

    async with httpx.AsyncClient(timeout=120) as cli:
        rows = []
        for i in range(N_CALLS):
            try:
                r = await one(cli, chunks[i % len(chunks)], i + 1)
                rows.append(r)
                hit = r["hit"] or 0
                miss = r["miss"]
                tot = r["prompt_tokens"] or 1
                print(f"  {r['idx']}. prompt={tot:>5}  "
                      f"hit={hit:>5}  miss={str(miss):>5}  "
                      f"completion={r['completion']}")
            except Exception as e:
                print(f"  {i+1}. 失败: {type(e).__name__} {str(e)[:80]}")
            await asyncio.sleep(0.5)

    print()
    print("=" * 84)
    print("汇总")
    print("=" * 84)
    if not rows:
        print("  无有效样本")
        return

    hits = [r["hit"] or 0 for r in rows]
    misses = [r["miss"] or 0 for r in rows]
    totals = [r["prompt_tokens"] or 0 for r in rows]
    sum_hit, sum_miss = sum(hits), sum(misses)

    print(f"  样本数            {len(rows)}")
    print(f"  总 prompt token   {sum(totals):,}")
    print(f"  缓存命中 token    {sum_hit:,}  ({sum_hit/max(sum(totals),1):.1%})")
    print(f"  缓存未命中 token  {sum_miss:,}")
    print()
    print("  逐次命中率:")
    for r in rows:
        t = r["prompt_tokens"] or 1
        h = r["hit"] or 0
        print(f"    #{r['idx']}  {h/t:>6.1%}")

    print()
    print("=" * 84)
    print("成本对比（按 DeepSeek 定价：hit $0.014/M，miss $0.14/M）")
    print("=" * 84)
    if sum_hit + sum_miss > 0:
        # 无缓存：全部按 miss 计
        no_cache = sum(totals) / 1e6 * 0.14
        with_cache = (sum_hit / 1e6 * 0.014) + (sum_miss / 1e6 * 0.14)
        print(f"  无缓存（假设）    ${no_cache:.6f}")
        print(f"  实测（含缓存）    ${with_cache:.6f}")
        if no_cache > 0:
            print(f"  节省              {1 - with_cache/no_cache:.1%}")

    out = Path(Path(__file__).resolve().parent.parent / "data" / "eval" / "p4d_cache_measure.json")
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  明细写入 {out}")


asyncio.run(main())
