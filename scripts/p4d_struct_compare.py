"""
P4-D 假设验证：能否提高缓存命中率？

已测事实
--------
每次调用 prompt ≈305 token，其中 hit=128（恒定 41.9%）。
system prompt 约 450 token，说明**只有部分前缀被缓存**。

DeepSeek 缓存规则（官方文档）
----------------------------
- 只有从第 0 token 起**完全相同的前缀**才命中
- 以 **64 token 为存储单元**
- 不足 64 token 不缓存

本实验测试三种结构，看哪种命中率最高：

  S1  当前结构（baseline）
        [system: 675字符 prompt]
        [user: 指令 + chunk]

  S2  role 交换 —— 把大段固定文本放到 user 首条消息
        [user: 675字符 prompt + 指令]
        [user: chunk]

  S3  长固定前缀（在 prompt 前先放一段固定的角色说明，凑满缓存块）
        [system: 角色说明 + 675字符 prompt]
        [user: 指令 + chunk]

目的：找到能命中最多 token 的结构，而不是改 prompt 内容。
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

CHUNKS = [
    "## 2. 工作时间\n标准工作时间为周一至周五 9:00 至 18:00，午休 12:00 至 13:00。",
    "## 3. 打卡\n员工应在上班前和下班后各完成一次打卡，连续两日未打卡视为旷工。",
    "## 4. 请假流程\n请假使用 OA 系统填写申请单，按审批权限提交审批，获批后方可休假。",
    "## 5. 外勤\n外勤前应在考勤系统中登记地点、事由和预计返回时间。",
    "## 6. 异常处理\n考勤记录不一致时，应在次月 5 个工作日内提出申诉。",
    "## 2. 密级定义\n密级分为公开、内部、机密、绝密四级，按敏感程度递增。",
]

INSTR = "请从以下文本中抽取知识：\n\n"


def build(struct: str, chunk: str) -> list[dict]:
    if struct == "S1_current":
        return [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": f"{INSTR}{chunk}"},
        ]
    if struct == "S2_user_prefix":
        return [
            {"role": "user", "content": f"{EXTRACTION_SYSTEM_PROMPT}\n\n{INSTR}{chunk}"},
        ]
    if struct == "S3_shared_prefix":
        # 在 system 前追加一段固定角色说明（增大可缓存固定前缀）
        head = (
            "你是企业制度知识抽取引擎，服务于一个内部制度问答系统。"
            "你的输出会被写入结构化知识库，供后续检索与问答使用。"
            "请严格遵循下面的抽取规范，不要添加额外解释。\n\n"
        )
        return [
            {"role": "system", "content": head + EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": f"{INSTR}{chunk}"},
        ]
    raise ValueError(struct)


async def measure(cli, struct: str, n: int = 6):
    rows = []
    for i in range(n):
        try:
            r = await cli.post(
                f"{settings.openai_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                json={
                    "model": settings.openai_model,
                    "messages": build(struct, CHUNKS[i % len(CHUNKS)]),
                    "temperature": 0,
                },
            )
            r.raise_for_status()
            u = r.json().get("usage", {}) or {}
            rows.append({
                "prompt": u.get("prompt_tokens") or 0,
                "hit": u.get("prompt_cache_hit_tokens") or 0,
                "miss": u.get("prompt_cache_miss_tokens") or 0,
            })
        except Exception as e:
            print(f"     (失败 {type(e).__name__})")
        await asyncio.sleep(0.4)
    return rows


async def main():
    print("=" * 84)
    print("缓存命中率：三种消息结构对比")
    print("=" * 84)
    print()

    results = {}
    async with httpx.AsyncClient(timeout=120) as cli:
        for struct in ("S1_current", "S2_user_prefix", "S3_shared_prefix"):
            print(f"--- {struct} ---")
            rows = await measure(cli, struct)
            if not rows:
                print("     无有效样本")
                continue
            # 跳过第 1 次（缓存冷启动）
            warm = rows[1:] or rows
            th = sum(r["hit"] for r in warm)
            tm = sum(r["miss"] for r in warm)
            tp = sum(r["prompt"] for r in warm)
            results[struct] = {
                "n": len(warm),
                "avg_prompt": tp / len(warm),
                "avg_hit": th / len(warm),
                "avg_miss": tm / len(warm),
                "hit_rate": th / max(tp, 1),
            }
            print(f"     平均 prompt {tp/len(warm):.0f}  "
                  f"hit {th/len(warm):.0f}  miss {tm/len(warm):.0f}  "
                  f"命中率 {th/max(tp,1):.1%}")
            print()

    print("=" * 84)
    print("对比汇总（已排除首次冷启动）")
    print("=" * 84)
    print(f"  {'结构':<20}{'平均prompt':>12}{'hit':>8}{'miss':>8}{'命中率':>10}")
    for k, v in results.items():
        print(f"  {k:<20}{v['avg_prompt']:>12.0f}{v['avg_hit']:>8.0f}"
              f"{v['avg_miss']:>8.0f}{v['hit_rate']:>10.1%}")

    print()
    print("=" * 84)
    print("成本（每 1000 次抽取调用）")
    print("=" * 84)
    print(f"  {'结构':<20}{'命中成本':>14}{'未命中成本':>14}{'合计':>12}")
    for k, v in results.items():
        # 每 1000 次
        hit_cost = v["avg_hit"] * 1000 / 1e6 * 0.014
        miss_cost = v["avg_miss"] * 1000 / 1e6 * 0.14
        print(f"  {k:<20}{hit_cost:>13.4f}$ {miss_cost:>13.4f}$ {hit_cost+miss_cost:>11.4f}$")

    out = Path(Path(__file__).resolve().parent.parent / "data" / "eval" / "p4d_struct_compare.json")
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  写入 {out}")


asyncio.run(main())
