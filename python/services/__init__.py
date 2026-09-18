"""
services 包

注意：本文件**故意不做** eager import。

原实现在此处 `from .knowledge_graph import ...`，而 knowledge_graph 又导入
`agents.knowledge_extract_agent`（反向依赖 agents 包），一旦某个 agents 模块
导入 `services.xxx`，就会形成循环：

    agents.doc_parser_agent
      → services.llm_factory
      → (触发 services/__init__) services.knowledge_graph
      → agents.knowledge_extract_agent
      → agents.doc_parser_agent   ← 仍在初始化中，ImportError

（原先的循环触发点已不存在，
 但「空包语义」的约定保留 —— 它避免的是同类反向依赖问题。）

因此这里保持空包语义，使用方按需直接导入具体子模块：

    from services.vector_store import VectorStoreService
    from services.bm25 import BM25Index
"""

__all__: list[str] = []
