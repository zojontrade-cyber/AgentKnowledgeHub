"""
P4-B 核心实验：Answer-bearing Fact Recall（语义判定）

为什么重跑
----------
上一次 P4-B 只保存了聚合数字，没保存抽取结果，无法做事实级判定。
且当时的 fact recall 用**字符串匹配**（基线仅 46.3%，明显失效）。

本脚本：
  1. 三档 batch 重新抽取，**保存完整抽取结果**
  2. 用 LLM 判 "抽取结果是否表达了该事实"（允许同义/单位/格式变化）
  3. 0/1/2 三档评分：缺失 / 部分 / 完整
  4. Fact Recall = sum(score) / (2 * n_facts)

Fact 集：从 bench_v3 的 exact 题手工提炼"最小答案事实"
（只列回答该问题**必需**的信息点）
"""

import asyncio
import json
import os
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

_PY = Path(Path(__file__).resolve().parent.parent / "python")
os.chdir(_PY)
sys.path.insert(0, str(_PY))
sys.stdout.reconfigure(encoding="utf-8")

import chromadb
from openai import AsyncOpenAI

from agents.doc_parser_agent import DocType, DocumentChunk
from agents.knowledge_extract_agent import KnowledgeExtractAgent
from config import settings

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "eval"

# 每题的最小答案事实集（只列回答所必需的信息点）
FACTS: dict[str, list[str]] = {
    "上班时间是几点到几点？": ["9:00", "18:00", "午休 12:00-13:00"],
    "请假要走什么流程？": ["OA 系统填写申请单", "1 天以内直属主管审批", "3 天以上分管领导审批"],
    "外勤要提前报备吗？": ["外勤前登记地点、事由和预计返回时间"],
    "年假有多少天？": ["满1年不满10年 5 天", "满10年不满20年 10 天", "满20年 15 天"],
    "年假要提前多久申请？": ["提前 3 个工作日"],
    "年假没休完能跨年吗？": ["经审批后可跨 1 个年度"],
    "出差住宿费能报多少？": ["一线城市 500 元/晚", "省会城市 400 元/晚", "其他城市 300 元/晚"],
    "出差回来多久内要报销？": ["10 个工作日"],
    "保密级别分几级？": ["公开", "内部", "机密", "绝密"],
    "加班费怎么算？": ["工作日 150%", "周末 200%", "法定节假日 300%"],
    "加班要提前申请吗？": ["下班前 2 小时提交申请"],
    "用印要走什么流程？": ["填写申请单", "注明文件名称、份数、用途和审批人"],
    "劳动合同一般签几年？": ["首次合同期限一般为 3 年"],
    "工资几号发？": ["每月 10 日"],
    "离职要交接哪些内容？": ["在办事项、文档资料、账号权限", "设备和财务事项"],
    "招聘有哪些渠道？": ["招聘网站", "内推", "猎头", "校园招聘", "公司官网"],
    "培训费用能报销吗？": ["报销后应按约定服务期继续服务"],
    "绩效多久考核一次？": ["按季度考核"],
    "数据分了哪几个级别？": ["L1 公开", "L2 内部", "L3 敏感", "L4 高敏"],
    "发生安全事件怎么处置？": ["先止损、后取证、再恢复", "涉及客户数据 24 小时内通知"],
    "软件的许可证怎么管理？": ["许可到期前 60 天提醒续费"],
    "用开源代码要注意什么？": ["GPL、AGPL 等传染性许可须经法务评估"],
    "新供应商怎么准入？": ["提交营业执照、资质、案例和报价"],
    "固定资产报废怎么处理？": ["使用人提交报废申请", "涉密设备先完成数据清除"],
    "固定资产多久盘点一次？": ["每季度抽查", "每年 12 月全面盘点"],
    "哪些文件要归档？": ["行政、人事、财务、合同、研发、客户和知识产权档案"],
    "档案怎么借阅？": ["填写借阅单", "机密档案按密级审批"],
    "上线前要谁审批？": ["研发、测试、运维和安全负责人共同确认"],
    "代码提交有什么要求？": ["Conventional Commits 格式", "禁止提交敏感内容"],
    "代码评审怎么做？": ["至少一名评审人批准", "关注正确性、安全性、性能"],
    "分支怎么管理？": ["main 保持可发布", "禁止直接向 main 提交"],
    "客户资料能导出吗？": ["批量导出须审批", "导出文件加密"],
    "办公用品怎么领？": ["办公用品系统提交领用申请"],
    "会议室预订后能取消吗？": ["会议取消后应在系统中释放预订", "连续 3 次爽约限制预订 1 周"],
    "技术变更分几级？": ["低风险", "中风险", "高风险"],
    "工作时间能处理私事吗？": ["不从事与工作无关且影响效率的活动"],
    "远程办公怎么申请？": ["提前 3 个工作日提交申请"],
}

JUDGE = """你在做"抽取结果是否包含某条关键事实"的判定。

目标事实：
{fact}

抽取结果（知识抽取器从制度文档中提取的实体与关系）：
{extracted}

请判断抽取结果是否**表达了该事实**，允许同义表述、单位变化、格式差异。
例如：
- 事实"9:00" 与 抽取"标准工作时间为每日8小时" -> 部分表达（1）
- 事实"10 个工作日" 与 抽取"10 个工作日内提交报销单" -> 完整表达（2）
- 抽取结果完全没提这件事 -> 缺失（0）

只返回 JSON：
{{"score": 0 或 1 或 2, "reason": "一句话"}}"""


def norm(s):
    return re.sub(r"[\s\u3000]+", "", unicodedata.normalize("NFKC", s or ""))


async def main():
    client_ch = chromadb.PersistentClient(path=settings.chroma_path)
    col = client_ch.get_or_create_collection("knowledge_chunks")
    got = col.get(include=["documents", "metadatas"])

    by_doc: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for d, m in zip(got["documents"], got["metadatas"]):
        m = m or {}
        by_doc[m.get("title") or m.get("doc_id")].append((m.get("section_title"), d or ""))

    ex = KnowledgeExtractAgent()
    arms = {}

    # ── 1. 三档抽取（保存结果）────────────────────────────
    for arm, batch in (("A_1chunk", 1), ("B_2chunk", 2), ("C_3chunk", 3)):
        cache = OUT_DIR / f"p4b_extract_{arm}.json"
        if cache.exists():
            print(f"{arm}: 复用缓存 {cache.name}")
            arms[arm] = json.loads(cache.read_text(encoding="utf-8"))
            continue

        print(f"{arm}: 抽取中（{batch} chunk/call）...")
        t0 = time.perf_counter()
        # 断点续传：已完成的部分调用逐条落盘，网络抖动不丢进度
        part = OUT_DIR / f"p4b_part_{arm}.jsonl"
        done_keys: set[str] = set()
        per_call: list[dict] = []
        if part.exists():
            for line in part.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    per_call.append(rec)
                    done_keys.add(rec["key"])
            print(f"   续传：已有 {len(per_call)} 条")

        calls = len(per_call)
        with part.open("a", encoding="utf-8") as fh:
            for title, secs in by_doc.items():
                chunks = [
                    DocumentChunk(
                        content=c, doc_id=title, chunk_index=i,
                        doc_type=DocType.MARKDOWN,
                        metadata={"section_title": sec, "title": title},
                    )
                    for i, (sec, c) in enumerate(secs)
                ]
                for i in range(0, len(chunks), batch):
                    grp = chunks[i : i + batch]
                    key = f"{title}#{i}"
                    if key in done_keys:
                        continue
                    merged = "\n\n".join(
                        f"## {g.metadata.get('section_title','')}\n{g.content}" for g in grp
                    )
                    # 单块失败（重试耗尽）不应中断整轮——记空结果继续
                    try:
                        res = await ex.extract_single(merged, chunk_id=key)
                    except Exception as e:
                        print(f"   [跳过] {key} 抽取失败: {type(e).__name__}")
                        res = None
                    calls += 1
                    items = []
                    if res is not None:
                        for e in res.entities:
                            items.append(f"{e.name}（{e.type}）{e.description}")
                        for r in res.relations:
                            items.append(f"{r.head} -[{r.relation}]-> {r.tail}")
                    rec = {
                        "key": key, "doc": title,
                        "sections": [g.metadata.get("section_title") for g in grp],
                        "text": "\n".join(items),
                    }
                    per_call.append(rec)
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()

        arms[arm] = {"calls": calls, "ms": (time.perf_counter() - t0) * 1000,
                     "per_call": per_call}
        cache.write_text(json.dumps(arms[arm], ensure_ascii=False), encoding="utf-8")
        print(f"   调用 {calls}  {arms[arm]['ms']/1000:.0f}s  已缓存")

    # ── 2. 每题：找出该题所属文档的抽取文本 ───────────────
    bench = {json.loads(l)["question"]: json.loads(l)
             for l in (ROOT / "data" / "eval" / "bench_v3.jsonl").read_text(encoding="utf-8").splitlines()
             if l.strip()}

    async_client = AsyncOpenAI(
        api_key=settings.openai_api_key, base_url=settings.openai_base_url,
        timeout=settings.llm_timeout_seconds,
    )
    sem = asyncio.Semaphore(4)

    async def judge(fact, extracted):
        async with sem:
            try:
                rr = await async_client.chat.completions.create(
                    model=settings.openai_model,
                    messages=[{"role": "user", "content": JUDGE.format(
                        fact=fact, extracted=extracted[:3000])}],
                    temperature=0,
                    response_format={"type": "json_object"},
                )
                d = json.loads(rr.choices[0].message.content or "{}")
                return int(d.get("score", 0)), d.get("reason", "")
            except Exception as e:
                return 0, f"ERR {e}"

    print()
    print("=" * 84)
    print("Answer-bearing Fact Recall（语义判定，0/1/2 三档）")
    print("=" * 84)

    results = {}
    for arm in arms:
        blob_by_doc = defaultdict(list)
        for c in arms[arm]["per_call"]:
            blob_by_doc[c["doc"]].append(c["text"])

        tasks, meta = [], []
        for q, facts in FACTS.items():
            b = bench.get(q)
            if not b:
                continue
            doc = b.get("doc_title")
            blob = "\n".join(blob_by_doc.get(doc, []))
            if not blob:
                continue
            for f in facts:
                tasks.append(judge(f, blob))
                meta.append((q, f))

        out = await asyncio.gather(*tasks)
        scores = [s for s, _ in out]
        total = sum(scores)
        maxs = 2 * len(scores)
        recall = total / maxs if maxs else 0

        missing = [(q, f, r) for (q, f), (s, r) in zip(meta, out) if s == 0]
        partial = [(q, f) for (q, f), (s, _) in zip(meta, out) if s == 1]

        results[arm] = {
            "calls": arms[arm]["calls"],
            "facts": len(scores), "score_sum": total, "max": maxs,
            "fact_recall": recall,
            "n_missing": len(missing), "n_partial": len(partial),
            "missing": missing[:15],
        }
        print(f"\n{arm}  (调用 {arms[arm]['calls']})")
        print(f"  Fact Recall  {total}/{maxs} = {recall:.1%}")
        print(f"  完整 {sum(1 for s in scores if s==2)}  部分 {len(partial)}  缺失 {len(missing)}")
        if missing:
            print(f"  缺失样例:")
            for q, f, _ in missing[:6]:
                print(f"     [{q[:18]}] {f}")

    # ── 3. 汇总 ──────────────────────────────────────────
    print()
    print("=" * 84)
    print("汇总")
    print("=" * 84)
    base = results["A_1chunk"]["fact_recall"]
    print(f"  {'方案':<12}{'调用':>7}{'Fact Recall':>13}{'相对基线':>10}{'缺失':>7}{'部分':>7}")
    for arm, r in results.items():
        print(f"  {arm:<12}{r['calls']:>7}{r['fact_recall']:>13.1%}"
              f"{r['fact_recall']-base:>+10.1%}{r['n_missing']:>7}{r['n_partial']:>7}")

    print()
    print("  选择标准（用户指定）：Fact Recall ≥ baseline - 2pp 才可接受")
    for arm, r in results.items():
        drop = base - r["fact_recall"]
        verdict = "✅ 可接受" if drop <= 0.02 else f"❌ 超出（-{drop:.1%}）"
        print(f"    {arm:<12} {verdict}")

    out = OUT_DIR / "p4b_fact_recall.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写入 {out}")


asyncio.run(main())
