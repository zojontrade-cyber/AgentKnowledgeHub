"""
实验 1：对比三种 embedding 输入的区分度

按你的建议做对照：
  A. 当前实现      文档标题 + 章节：X + 正文
  B. 加强上下文    公司/类别/文档/章节/内容（五段式）
  C. 去套话        只保留章节标题 + 正文（完全不含制度套话）

用同一个 query，看 target 与 wrong 的分数差。
"""

import asyncio
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
sys.stdout.reconfigure(encoding="utf-8")

import chromadb
from langchain_openai import OpenAIEmbeddings

from agents.doc_parser_agent import DocParserAgent
from config import settings


def cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-9)


# 用**系统同一个 client 类型**（LangChain），避免上次的客户端不一致问题
emb = OpenAIEmbeddings(
    model=settings.embedding_model,
    api_key=settings.effective_embedding_api_key,
    base_url=settings.effective_embedding_base_url,
)

SRC = Path(r"C:\Users\滨滨\Desktop\Knowledge-Base-Intelligent-Q-A-main\data\clean\institutional")
parser = DocParserAgent()

QUERIES = [
    ("公司的标准工作时间是什么时候到什么时候", "工作时间"),
    ("请假要走什么流程", "请假"),
    ("保密级别分几级", "密级"),
    ("出差住宿费能报多少", "住宿"),
    ("加班费怎么算", "加班费"),
    ("年假有多少天", "年假"),
]


def build_forms(doc_title: str, sec_title: str, body: str) -> dict[str, str]:
    """构造三种 embedding 输入"""
    return {
        # A. 当前实现
        "A_当前": f"{doc_title}\n章节：{sec_title}\n{body}",
        # B. 加强上下文（五段式）
        "B_强上下文": (
            f"公司名称：云启科技\n\n"
            f"制度类别：企业管理制度\n\n"
            f"文档：{doc_title}\n\n"
            f"章节：{sec_title}\n\n"
            f"内容：{body}"
        ),
        # C. 去套话（只留章节标题 + 正文）
        "C_去套话": f"{sec_title}\n{body}",
    }


async def main():
    # 收集所有章节
    items = []          # (doc_title, sec_title, body)
    for f in sorted(SRC.glob("*.md")):
        raw = f.read_text(encoding="utf-8", errors="ignore")
        clean = parser.strip_metadata(raw)
        doc_title, sections = parser.parse_tree(clean)
        for s in sections:
            if s.body and len(s.body) >= 30:
                items.append((doc_title, s.title, s.body))

    print(f"共 {len(items)} 个章节\n")

    for form_name in ("A_当前", "B_强上下文", "C_去套话"):
        print("=" * 76)
        print(f"形态 {form_name}")
        print("=" * 76)

        # 批量编码所有章节
        texts = [build_forms(d, s, b)[form_name] for d, s, b in items]
        vecs = await emb.aembed_documents(texts)

        hit1 = 0
        for q, expect_kw in QUERIES:
            qv = await emb.aembed_query(q)
            scored = sorted(
                ((cos(qv, v), it) for v, it in zip(vecs, items)),
                key=lambda x: x[0], reverse=True,
            )
            rank = 0
            target_score = 0.0
            wrong_score = scored[0][0]
            for i, (sc, (dt, st, bd)) in enumerate(scored):
                if expect_kw in st or expect_kw in bd[:80]:
                    rank = i + 1
                    target_score = sc
                    break
            if rank == 1:
                hit1 += 1
            gap = target_score - wrong_score if rank else 0
            mark = "★" if rank == 1 else (" " if rank and rank <= 5 else "✗")
            print(f"  {mark} {q[:26]:<28} 排名={rank or '>10':<5} "
                  f"target={target_score:.4f} top1={wrong_score:.4f} 差={gap:+.4f} "
                  f"[{scored[0][1][0][:14]}/{scored[0][1][1][:10]}]")

        print(f"  → Top-1 命中: {hit1}/{len(QUERIES)}")
        print()


asyncio.run(main())
