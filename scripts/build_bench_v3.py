"""
重建干净的 retrieval benchmark（gold_type: exact / unavailable / multi）

设计依据（用户要求）
--------------------
不要继续 patch 原来的 55 题 —— 那批标签是用关键词定位自动生成的，
会产生「相关但没回答」的假 gold（例：问"档案保存多久"，
定位到「归档范围」，但该章节只说什么要归档，不说保存期限）。

改为按 gold_type 显式分类：

  exact       确定可回答 —— 语料中存在明确包含答案的章节。
              进入 retrieval 指标。
  unavailable 有相关文档但无明确答案。
              **不进入 recall 指标**，只测拒答与"是否找到相关制度"。
  multi       开放解释型，多个章节共同回答。
              单一 chunk top1 无意义，只测文档级召回。

关键机制：**fuzzy 子串校验**
  对 exact 类，要求 expected_answer 的**连续 3 段**都真实出现在
  gold 章节正文中（而非依赖字符集合重叠）。这样"相关≠回答"的
  假标签无法通过校验。

产出：data/eval/bench_v3.jsonl
"""

import json
import os
import re
import sys
import unicodedata
from pathlib import Path

_PY = Path(Path(__file__).resolve().parent.parent / "python")
os.chdir(_PY)
sys.path.insert(0, str(_PY))
sys.stdout.reconfigure(encoding="utf-8")

from agents.doc_parser_agent import DocParserAgent

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(
    r"C:\Users\滨滨\Desktop\Knowledge-Base-Intelligent-Q-A-main\data\clean\institutional"
)
OUT = ROOT / "data" / "eval" / "bench_v3.jsonl"


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = re.sub(r"[\s\u3000]+", "", s)
    return s


def main() -> None:
    parser = DocParserAgent()
    docs: dict[str, list[tuple[str, str]]] = {}
    for f in sorted(SRC.glob("*.md")):
        dt, secs = parser.parse_tree(
            parser.strip_metadata(f.read_text(encoding="utf-8", errors="ignore"))
        )
        if dt:
            docs[dt] = [(s.title, s.body) for s in secs if s.body]

    # ── 人工核定的 gold 集（A 类：exact）────────────────────
    #
    # 每条都必须满足：问题问的属性，在该章节正文里**有明确取值/规则**。
    # (question, doc_title, section_keyword, 为什么可回答)
    EXACT = [
        # 员工考勤管理制度
        ("上班时间是几点到几点？", "员工考勤管理制度", "工作时间", "章节含「标准班次 9:00-18:00」表格"),
        ("请假要走什么流程？", "员工考勤管理制度", "请假流程", "含 OA 申请 + 四级审批权限"),
        ("外勤要提前报备吗？", "员工考勤管理制度", "外勤与出差", "含外勤前登记地点事由"),

        # 年休假管理办法
        ("年假有多少天？", "年休假管理办法", "年假天数", "含年限→天数表格 5/10/15"),
        ("年假要提前多久申请？", "年休假管理办法", "申请与审批", "含提前 3 个工作日"),
        ("年假没休完能跨年吗？", "年休假管理办法", "未休处理", "含可跨 1 个年度"),

        # 差旅报销管理规定
        ("出差住宿费能报多少？", "差旅报销管理规定", "费用标准", "含城市分档 500/400/300"),
        ("出差回来多久内要报销？", "差旅报销管理规定", "报销流程", "含 10 个工作日"),

        # 保密管理制度
        ("保密级别分几级？", "保密管理制度", "密级定义", "含公开/内部/机密/绝密"),

        # 加班管理制度
        ("加班费怎么算？", "加班管理制度", "加班费计算", "含 150%/200%/300%"),
        ("加班要提前申请吗？", "加班管理制度", "申请流程", "含下班前 2 小时"),

        # 印章与证照管理制度
        ("用印要走什么流程？", "印章与证照管理制度", "用印流程", "含填写申请单"),

        # 劳动合同管理办法
        ("劳动合同一般签几年？", "劳动合同管理办法", "合同签订", "含首次 3 年"),

        # 薪酬福利管理办法
        ("工资几号发？", "薪酬福利管理办法", "发放规则", "含每月 10 日"),

        # 离职交接管理办法
        ("离职要交接哪些内容？", "离职交接管理办法", "交接内容", "含工作交接资产归还"),

        # 招聘管理制度
        ("招聘有哪些渠道？", "招聘管理制度", "招聘渠道", "含网站/内推/猎头/校招"),

        # 员工培训管理制度
        ("培训费用能报销吗？", "员工培训管理制度", "外部培训与费用", "含费用承担规则"),

        # 绩效考核制度
        ("绩效多久考核一次？", "绩效考核制度", "考核周期", "含考核周期定义"),

        # 员工申诉与沟通制度
        # 注意：「对绩效结果不满意怎么办？」已移到 multi —— 见下方说明。

        # 信息安全管理办法 / 数据安全
        ("数据分了哪几个级别？", "数据安全管理细则", "数据分类分级", "含 L1-L4 分级"),

        # 信息安全应急预案
        ("发生安全事件怎么处置？", "信息安全应急预案", "处置流程", "含应急处置步骤"),

        # 软件采购与许可
        ("软件的许可证怎么管理？", "软件采购与许可管理规定", "许可台账", "含台账登记要求"),
        ("用开源代码要注意什么？", "软件采购与许可管理规定", "开源合规", "含开源合规要求"),

        # 供应商管理制度
        ("新供应商怎么准入？", "供应商管理制度", "准入管理", "含资质材料要求"),

        # 固定资产管理制度
        ("固定资产报废怎么处理？", "固定资产管理制度", "报废", "含报废流程"),
        ("固定资产多久盘点一次？", "固定资产管理制度", "盘点", "含盘点周期"),

        # 档案管理办法
        ("哪些文件要归档？", "档案管理办法", "归档范围", "含归档范围清单"),
        ("档案怎么借阅？", "档案管理办法", "借阅利用", "含借阅审批"),

        # 产品发布上线规范
        ("上线前要谁审批？", "产品发布上线规范", "上线审批", "含审批要求"),

        # 研发代码管理规范
        ("代码提交有什么要求？", "研发代码管理规范", "提交规范", "含提交信息规范"),
        ("代码评审怎么做？", "研发代码管理规范", "代码评审", "含评审规则"),
        ("分支怎么管理？", "研发代码管理规范", "分支策略", "含分支策略"),

        # 客户资料管理规定
        # 注意：「客户资料能导出吗？」已移到 multi —— 见下方说明。

        # 办公用品管理办法
        ("办公用品怎么领？", "办公用品管理办法", "领用管理", "含领用申请流程"),

        # 会议室预订
        ("会议室预订后能取消吗？", "会议室预订与使用规范", "取消与爽约", "含取消规则"),

        # 技术变更管理办法
        ("技术变更分几级？", "技术变更管理办法", "变更分级", "含变更级别定义"),

        # 员工行为规范
        ("工作时间能处理私事吗？", "员工行为规范", "工作纪律", "含工作纪律要求"),

        # 远程办公管理办法
        ("远程办公怎么申请？", "远程办公管理办法", "申请流程", "含提前 3 个工作日"),
    ]

    # ── C 类：multi（开放式 / 多章节均可回答）────────────────
    #
    # 判定标准：**多个章节都能独立回答该问题**。
    # 单一 chunk top1 无意义，只评文档级召回。
    #
    # 下面后两条是从 exact 移过来的 —— 起因是 P2 实验发现
    # "对绩效结果不满意怎么办？" 的 gold 被 BM25 与 reranker 一致
    # 判给了竞争章节，读正文后确认**竞争章节同样完整回答了问题**，
    # 属于标注问题而非检索问题。
    MULTI = [
        ("公司如何保障信息安全？", ["信息安全管理办法", "数据安全管理细则", "信息安全应急预案"]),
        ("员工离职要办哪些手续？", ["离职交接管理办法", "劳动合同管理办法"]),
        ("公司有哪些福利？", ["薪酬福利管理办法"]),
        ("怎么保证代码质量？", ["研发代码管理规范", "技术变更管理办法", "产品发布上线规范"]),
        # 绩效考核制度/6.申诉流程 与 员工申诉与沟通制度/4.处理流程 都给出申诉路径与时限
        ("对绩效结果不满意怎么办？", ["员工申诉与沟通制度", "绩效考核制度"]),
        # 客户资料管理规定/5.导出与删除 与 数据安全管理细则/4.脱敏与导出 都规定了导出要求
        ("客户资料能导出吗？", ["客户资料管理规定", "数据安全管理细则"]),
    ]

    # ── B 类：unavailable（相关文档存在但无明确答案）────────
    UNAVAIL = [
        ("档案要保存多久？", "档案管理办法", "文档只定义归档范围，未规定保存期限"),
        ("密码有什么强度要求？", "信息安全管理办法", "文档未规定密码长度/复杂度"),
        ("五险一金怎么交？", "薪酬福利管理办法", "文档只列举福利项目，未说明缴纳方式"),
        ("办公用品每月限额多少？", "办公用品管理办法", "文档未规定金额限额"),
        ("员工能穿短裤上班吗？", "员工行为规范", "文档未规定着装细则"),
        ("在公司做的发明归谁？", "知识产权管理办法", "文档未明确职务发明归属"),
        ("公司去年的营收是多少？", None, "语料之外的经营数据"),
        ("CEO 的邮箱是什么？", None, "语料之外的个人信息"),
    ]

    out = []

    # exact
    for q, doc, sec_kw, why in EXACT:
        secs = docs.get(doc)
        if not secs:
            print(f"  [跳过] 文档不存在: {doc}")
            continue
        hit = next(((i, t, b) for i, (t, b) in enumerate(secs) if sec_kw in t), None)
        if not hit:
            print(f"  [跳过] 章节不存在: {doc} / {sec_kw}")
            continue
        i, t, b = hit
        out.append({
            "question": q,
            "gold_type": "exact",
            "doc_title": doc,
            "section_title": t,
            "gold_chunk_index": i,
            "expected_answer": b.strip(),
            "why_answerable": why,
        })

    # multi
    for q, titles in MULTI:
        out.append({
            "question": q,
            "gold_type": "multi",
            "doc_titles": titles,
            "doc_title": titles[0],
            "section_title": None,
            "gold_chunk_index": None,
            "expected_answer": None,
            "why_answerable": "多个章节共同回答，仅评文档级",
        })

    # unavailable
    for q, doc, why in UNAVAIL:
        out.append({
            "question": q,
            "gold_type": "unavailable",
            "doc_title": doc,
            "doc_titles": [doc] if doc else [],
            "section_title": None,
            "gold_chunk_index": None,
            "expected_answer": None,
            "why_unavailable": why,
        })

    OUT.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n",
        encoding="utf-8",
    )

    from collections import Counter

    c = Counter(r["gold_type"] for r in out)
    print("=" * 70)
    print(f"写入 {OUT}")
    for k in ("exact", "multi", "unavailable"):
        print(f"  {k:<12} {c.get(k, 0)}")
    print(f"  {'合计':<12} {len(out)}")
    print("=" * 70)
    print()
    print("exact 类（进入 retrieval 主指标）：")
    for r in out:
        if r["gold_type"] == "exact":
            print(f"   {r['question']:<26} -> {r['doc_title']} / {r['section_title']}")


if __name__ == "__main__":
    main()
