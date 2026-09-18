"""
重建评测集 —— 修正上一版的致命缺陷

上一版的问题
------------
题目是用「文档标题 + 章节名」机械拼接的，例如：
    文档《员工考勤管理制度》+ 章节「工作时间」
    -> 题目"员工考勤工作时间是怎么规定的？"

题目里**直接包含了答案所在文档的标题词**，向量检索必然命中，
导致 Recall 恒为 100% —— 这是在测字符串匹配，不是语义检索。

这一版的规则
------------
1. **题目不得出现文档标题词**（如"考勤"、"保密"、"差旅报销"）
   除非该词是普通用户真会说的口语表达
2. 优先使用**用户真实会问的说法**（口语化、间接指代）
3. 答案依然从文档实际内容提取，保证可判定
4. 保留负样本

产出：data/eval/eval_set.jsonl（覆盖写入）
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
sys.stdout.reconfigure(encoding="utf-8")

from agents.doc_parser_agent import DocParserAgent

SRC = Path(r"C:\Users\滨滨\Desktop\Knowledge-Base-Intelligent-Q-A-main\data\clean\institutional")
OUT = Path(__file__).resolve().parent.parent / "data" / "eval"
OUT.mkdir(parents=True, exist_ok=True)

parser = DocParserAgent()


def load_docs():
    docs = {}
    for f in sorted(SRC.glob("*.md")):
        raw = f.read_text(encoding="utf-8", errors="ignore")
        clean = parser.strip_metadata(raw)
        title = parser.extract_title(clean)
        m = re.search(r"(YQ-INST-\d+)", raw)
        docs[title] = {
            "code": m.group(1) if m else "",
            "file": f.name,
            "text": clean,
        }
    return docs


def extract_section(text: str, keyword: str) -> str:
    """取出含关键词的章节内容"""
    parts = re.split(r"(?m)^##\s+(.+)$", text)
    for i in range(1, len(parts) - 1, 2):
        if keyword in parts[i]:
            return parts[i + 1].strip()
    return ""


def best_answer(body: str) -> str:
    """从章节内容里提取一个可判定的答案片段"""
    for line in body.split("\n"):
        s = line.strip()
        # 表格数据行
        if s.startswith("|") and not re.match(r"^\|[\s\-|:]+\|$", s):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if len(cells) >= 2 and not re.search(r"(密级|事项|级别|项目|环节|情形|类型|方式|对象)", cells[0]):
                return " | ".join(cells)
    for line in body.split("\n"):
        s = line.strip().lstrip("#-* ").strip()
        s = re.sub(r"^\d+[\.、]\s*", "", s)
        if 8 < len(s) < 150 and not s.startswith("|"):
            return s
    return body[:120]


# ══════════════════════════════════════════════════════════
#  人工设计的题目：题面不含文档标题词，用真实用户说法
# ══════════════════════════════════════════════════════════
QUESTIONS = [
    # (问题, 期望文档标题关键词, 文档内关键词[用于定位章节])
    ("上班时间是几点到几点？", "员工考勤", "工作时间"),
    ("迟到和早退是怎么界定的？", "员工考勤", "打卡"),
    ("忘了打卡怎么办？", "员工考勤", "打卡"),
    ("请假要走什么手续？", "员工考勤", "请假"),
    ("外勤需要提前报备吗？", "员工考勤", "外勤"),

    ("入职多久能休年假？", "年休假", "享受条件"),
    ("年假能分几次休？", "年休假", "申请"),
    ("年底没休完的假怎么办？", "年休假", "折算"),

    ("出差住宿费能报多少？", "差旅报销", "标准"),
    ("报销单多久内要提交？", "差旅报销", "报销"),
    ("出差坐飞机有什么要求？", "差旅报销", "交通"),

    ("加班费怎么计算？", "加班管理", "加班费"),
    ("周末加班有补贴吗？", "加班管理", "加班费"),
    ("加班需要提前申请吗？", "加班管理", "申请"),

    ("公司的保密级别分几级？", "保密管理", "密级"),
    ("机密文件能带出公司吗？", "保密管理", "访问"),
    ("离职后还要保密吗？", "保密管理", "保密"),

    ("个人信息泄露了该怎么做？", "信息安全管理办法", "事件"),
    ("密码有什么强度要求？", "信息安全管理办法", "密码"),
    ("电脑能装盗版软件吗？", "软件采购", "许可"),

    ("数据泄露事件怎么上报？", "数据安全管理细则", "分级"),
    ("客户数据能存在个人电脑上吗？", "数据安全", "存储"),

    ("固定资产报废怎么走流程？", "固定资产", "报废"),
    ("电脑坏了找谁修？", "固定资产", "维修"),
    ("资产盘点多久做一次？", "固定资产", "盘点"),

    ("招聘流程有哪几步？", "招聘管理", "流程"),
    ("试用期多久？", "招聘管理", "试用期"),
    ("内推有奖励吗？", "招聘管理", "内推"),

    ("培训费用能报销吗？", "员工培训", "费用"),
    ("新员工要参加哪些培训？", "员工培训", "入职"),

    ("考核多久做一次？", "绩效考核", "周期"),
    ("考核结果会影响什么？", "绩效考核", "结果"),
    ("绩效申诉怎么提？", "绩效考核", "申诉"),

    ("上班能穿短裤吗？", "员工行为", "着装"),
    ("工作时间能处理私事吗？", "员工行为", "行为"),
    ("同事之间能谈恋爱吗？", "员工行为", "关系"),

    ("劳动合同一般签几年？", "劳动合同", "期限"),
    ("试用期工资怎么算？", "劳动合同", "工资"),
    ("合同到期不续签怎么办？", "劳动合同", "终止"),

    ("工资什么时候发？", "薪酬福利", "发放"),
    ("工资条能给别人看吗？", "薪酬福利", "保密"),
    ("五险一金怎么交？", "薪酬福利", "保险"),

    ("办公用品怎么领？", "办公用品", "领用"),
    ("打印纸找谁要？", "办公用品", "领用"),

    ("公章怎么申请使用？", "印章", "使用"),
    ("营业执照原件能借出吗？", "印章", "外借"),

    ("会议室要提前多久预约？", "会议室", "预约"),
    ("会议室能吃东西吗？", "会议室", "使用"),

    ("在家办公怎么申请？", "远程办公", "申请"),
    ("远程办公要打卡吗？", "远程办公", "考勤"),

    ("辞职要提前多久说？", "离职交接", "离职"),
    ("工作交接给谁？", "离职交接", "交接"),
    ("离职证明什么时候给？", "离职交接", "证明"),

    ("对处理结果不服怎么办？", "员工申诉", "申诉"),
    ("申诉多久有回复？", "员工申诉", "反馈"),
    ("申诉可以匿名吗？", "员工申诉", "匿名"),

    ("安全事件怎么分级？", "信息安全应急预案", "分级"),
    ("出了安全事件谁负责处理？", "信息安全应急预案", "响应"),
    ("应急演练多久做一次？", "信息安全应急预案", "演练"),

    ("采购软件要走什么审批？", "软件采购", "审批"),
    ("开源软件能用吗？", "软件采购", "开源"),

    ("新供应商怎么合作？", "供应商管理", "准入"),
    ("供应商考核多久一次？", "供应商管理", "评估"),

    ("合同谁来审批？", "合同管理", "审批"),
    ("合同原件保存在哪？", "合同管理", "归档"),

    ("档案要保存多久？", "档案管理", "保存"),
    ("档案能借出来吗？", "档案管理", "借阅"),

    ("产品上线前要做什么检查？", "产品发布", "上线"),
    ("线上出故障怎么回滚？", "产品发布", "回滚"),

    ("代码提交有什么要求？", "研发代码", "提交"),
    ("代码需要几个人评审？", "研发代码", "评审"),
    ("分支怎么管理？", "研发代码", "分支"),

    ("线上变更需要审批吗？", "技术变更", "审批"),
    ("紧急变更怎么处理？", "技术变更", "紧急"),

    ("客户资料能导出吗？", "客户资料", "导出"),
    ("客户信息能对外提供吗？", "客户资料", "对外"),

    ("在公司做的发明归谁？", "知识产权", "归属"),
    ("专利申请流程是什么？", "知识产权", "申请"),
]


def main():
    docs = load_docs()
    print(f"=== 载入 {len(docs)} 篇文档 ===")

    cases = []
    matched = 0
    unmatched = []

    for q, title_kw, sec_kw in QUESTIONS:
        # 找到匹配的文档
        target = None
        for t in docs:
            if title_kw in t:
                target = t
                break
        if not target:
            unmatched.append((q, title_kw))
            continue

        info = docs[target]
        body = extract_section(info["text"], sec_kw)
        if not body:
            body = info["text"]  # 找不到章节就用全文
        ans = best_answer(body)

        cases.append({
            "question": q,
            "expected_answer": ans[:200],
            "expected_source": info["file"],
            "expected_title": target,
            "expected_code": info["code"],
            "in_corpus": True,
        })
        matched += 1

    # 负样本
    for q in [
        "公司去年的营收是多少？",
        "CEO 的邮箱是什么？",
        "竞争对手的定价策略是什么？",
        "员工持股计划的具体条款？",
        "未来三年的战略规划是什么？",
        "公司的股价今天是多少？",
        "办公室的 WiFi 密码是什么？",
    ]:
        cases.append({
            "question": q,
            "expected_answer": None,
            "expected_source": None,
            "expected_title": None,
            "expected_code": None,
            "in_corpus": False,
        })

    # 检查：题目是否泄露了答案标题
    leaked = []
    for c in cases:
        if not c["in_corpus"]:
            continue
        t = c["expected_title"]
        # 去掉通用后缀后，看标题主干是否出现在题目里
        stem = t.replace("管理制度", "").replace("管理办法", "") \
                .replace("管理规定", "").replace("管理规范", "") \
                .replace("管理细则", "").replace("制度", "").replace("办法", "")
        if len(stem) >= 3 and stem in c["question"]:
            leaked.append((c["question"], t))

    out_file = OUT / "eval_set.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    pos = [c for c in cases if c["in_corpus"]]
    neg = [c for c in cases if not c["in_corpus"]]

    print()
    print("=== 统计 ===")
    print(f"  总题数:   {len(cases)}")
    print(f"  正样本:   {len(pos)}")
    print(f"  负样本:   {len(neg)}")
    print(f"  覆盖文档: {len({c['expected_title'] for c in pos})} 篇")
    print(f"  已写入:   {out_file}")

    if unmatched:
        print()
        print(f"  ⚠ 未匹配到文档的题目 {len(unmatched)} 条:")
        for q, kw in unmatched[:10]:
            print(f"    {q}  (关键词: {kw})")

    if leaked:
        print()
        print(f"  ⚠ 仍可能泄露答案的题目 {len(leaked)} 条:")
        for q, t in leaked[:10]:
            print(f"    {q}  ->  标题《{t}》")
    else:
        print()
        print("  ✅ 未检测到题目泄露答案标题")

    print()
    print("  样例:")
    for c in cases[:8]:
        print(f"    Q: {c['question']}")
        print(f"    A: {str(c['expected_answer'])[:55]}")
        print(f"    来源: {c['expected_title']}")


main()
