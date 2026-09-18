"""验证三层分离：embedding 输入 / 展示文本 / 父块"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
sys.stdout.reconfigure(encoding="utf-8")

from agents.doc_parser_agent import DocParserAgent

UPLOADS = Path(__file__).resolve().parent.parent / "python" / "uploads"
parser = DocParserAgent()


async def main():
    target = None
    for f in UPLOADS.glob("*.md"):
        c = f.read_text(encoding="utf-8", errors="ignore")
        if "标准工作时间" in c:
            target = f
            break

    print("=" * 66)
    print(f"文档: {target.name}")
    print("=" * 66)

    chunks = await parser.parse(str(target))
    print(f"\n共 {len(chunks)} 块\n")

    for c in chunks:
        has_ans = "标准工作时间" in c.content
        print(f"{'★ ' if has_ans else '  '}chunk-{c.chunk_index}")
        print(f"    章节标题:   {c.section_title!r}")
        print(f"    标题路径:   {c.title_path!r}")
        print(f"    ── embedding 输入（{len(c.content)} 字）──")
        print(f"    {c.content[:150]!r}")
        print(f"    ── 展示文本（{len(c.display)} 字）──")
        print(f"    {c.display[:100]!r}")
        print(f"    ── 父块（{len(c.parent_content)} 字，用于生成）──")
        print(f"    前 60 字: {c.parent_content[:60]!r}")
        print()

    # 关键检查
    print("=" * 66)
    print("关键检查")
    print("=" * 66)
    ans_chunks = [c for c in chunks if "标准工作时间" in c.content]
    print(f"  含答案的块数: {len(ans_chunks)}")
    if ans_chunks:
        c = ans_chunks[0]
        # embedding 输入是否干净（不含其他章节的通用套话）
        clean = "文档说明" not in c.content and "本制度规定" not in c.content
        print(f"  embedding 输入是否纯净（不含通用套话）: {'✓ 是' if clean else '✗ 否'}")
        print(f"  内容长度: {len(c.content)} 字（此前的大杂烩块为 337 字）")


asyncio.run(main())
