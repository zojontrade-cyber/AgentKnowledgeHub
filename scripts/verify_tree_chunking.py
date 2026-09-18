"""验证树形解析 + 三段式 embedding + 质量闸门"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
sys.stdout.reconfigure(encoding="utf-8")

from agents.doc_parser_agent import DocParserAgent

SRC = Path(r"C:\Users\滨滨\Desktop\Knowledge-Base-Intelligent-Q-A-main\data\clean\institutional")
parser = DocParserAgent()


async def main():
    f = None
    for x in SRC.glob("*.md"):
        if "标准工作时间" in x.read_text(encoding="utf-8", errors="ignore"):
            f = x
            break

    raw = f.read_text(encoding="utf-8")
    clean = parser.strip_metadata(raw)

    print("=" * 74)
    print("1. 树形解析结果")
    print("=" * 74)
    doc_title, sections = parser.parse_tree(clean)
    print(f"  文档标题: {doc_title!r}")
    print(f"  章节数:   {len(sections)}")
    for s in sections:
        has = "标准工作时间" in s.body
        print(f"    [L{s.level}] {s.title!r:<28} body={len(s.body):>4} 字 "
              f"{'← 含答案' if has else ''}")

    print()
    print("=" * 74)
    print("2. 质量闸门（纯标题块应被拒绝）")
    print("=" * 74)
    tests = [
        ("员工考勤管理制度", False, "纯文档标题"),
        ("2. 工作时间", False, "纯章节标题"),
        ("薪酬福利管理办法", False, "纯文档标题"),
        ("标准工作时间为周一至周五 9:00 至 18:00，午休 12:00 至 13:00。", True, "有正文"),
        ("## 5. 信息安全\n远程办公使用公司 VPN 和受控终端，禁止使用公共电脑。", True, "有正文"),
    ]
    for text, expect, label in tests:
        got = parser.is_indexable(text)
        ok = got == expect
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:<14} "
              f"is_indexable={got} (期望 {expect})  {text[:36]!r}")

    print()
    print("=" * 74)
    print("3. 实际分块（embedding 三段式）")
    print("=" * 74)
    chunks = await parser.parse(str(f))
    print(f"  共 {len(chunks)} 块（旧实现为 7 块，含 1 个 8 字纯标题块）\n")
    for c in chunks:
        has = "标准工作时间" in c.content
        print(f"{'★' if has else ' '} chunk-{c.chunk_index} ({len(c.content)} 字) "
              f"section={c.section_title!r}")
        print(f"    ┌─── embedding 输入 ───")
        for line in c.content.split("\n")[:4]:
            print(f"    │ {line[:80]}")
        print(f"    └──────────────────────")
        print()

    # 检查是否还有纯标题块
    titles_only = [c for c in chunks
                   if not parser.is_indexable(c.content.split("\n")[-1])]
    print(f"  纯标题块数量: {len(titles_only)} （应为 0）")


asyncio.run(main())
