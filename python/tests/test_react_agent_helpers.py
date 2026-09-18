"""
ReAct QA 链路回归测试

背景
----
`agents/react_qa_agent.py` 里 `_react_loop` 调用了两个**模块级 helper**
（`_fmt_args` / `_ctx_key`），但它们在某次重构中被删掉、调用点却留下了。
后果是 `qa_mode=react` 在第一次工具调用时必抛 NameError，整条链路不可用；
因为默认是 `qa_mode=pipeline`，这个缺陷一直没暴露。

因此这里加两道守卫：
  1. 静态扫描 —— 模块内**不能存在"被调用但未定义"的名字**（能拦住同类问题）
  2. 语义断言 —— `_ctx_key` 必须是**块级**去重键，不能用文档级 parent_id
     （那会把同一文档的所有章节折叠成一条，静默丢掉命中章节）
"""

import ast
import builtins
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from agents import react_qa_agent  # noqa: E402
from agents.qa_agent import RetrievedContext  # noqa: E402

SOURCE = Path(react_qa_agent.__file__)


def _module_scope_names(tree: ast.Module) -> tuple[set[str], set[str]]:
    """返回 (已定义的名字, 已导入的名字)"""
    defined: set[str] = set()
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defined.update(a.arg for a in node.args.args)
                defined.update(a.arg for a in node.args.kwonlyargs)
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                    defined.add(sub.id)
                if isinstance(sub, ast.arg):
                    defined.add(sub.arg)
                if isinstance(sub, ast.ExceptHandler) and sub.name:
                    defined.add(sub.name)
                if isinstance(sub, ast.comprehension):
                    for t in ast.walk(sub.target):
                        if isinstance(t, ast.Name):
                            defined.add(t.id)
        elif isinstance(node, ast.Import):
            for a in node.names:
                imported.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                imported.add(a.asname or a.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    defined.add(t.id)
    return defined, imported


def test_no_undefined_names_in_react_module():
    """模块内不允许存在"被调用但未定义"的名字

    这是 #1 缺陷的直接守卫：`_fmt_args` / `_ctx_key` 曾被调用却未定义，
    而静态扫描能在不联网、不起服务的前提下抓到它。
    """
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    defined, imported = _module_scope_names(tree)
    known = defined | imported | set(dir(builtins))

    missing: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in known:
                missing.setdefault(node.func.id, node.lineno)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in known:
                missing.setdefault(node.id, node.lineno)

    assert not missing, f"存在未定义的名字: {missing}"


@pytest.mark.parametrize("name", ["_fmt_args", "_ctx_key", "_build_tools"])
def test_helpers_exist(name):
    assert hasattr(react_qa_agent, name), f"{name} 缺失 —— react 链路会抛 NameError"
    assert callable(getattr(react_qa_agent, name))


def test_fmt_args_is_compact_single_line():
    out = react_qa_agent._fmt_args({"query": "年假 天数", "top_k": 5})
    assert "\n" not in out
    assert "query=" in out and "top_k=" in out


def test_fmt_args_handles_empty():
    assert react_qa_agent._fmt_args({}) == ""


def test_ctx_key_is_chunk_level_not_document_level():
    """去重键必须是 doc_id#chunk_index

    用 parent_id（文档级）去重会把同一文档的所有章节折叠成一条，
    静默丢掉命中章节 —— 这是早前 gold 章节"检索不到"的根因之一。
    """
    a = RetrievedContext("甲", "s", 0.9, "vector",
                         metadata={"doc_id": "D1", "chunk_index": 3,
                                   "parent_id": "D1#doc"})
    b = RetrievedContext("乙", "s", 0.8, "vector",
                         metadata={"doc_id": "D1", "chunk_index": 4,
                                   "parent_id": "D1#doc"})
    assert react_qa_agent._ctx_key(a) == "D1#3"
    assert react_qa_agent._ctx_key(b) == "D1#4"
    # 同一文档的不同章节不能被判为同一条
    assert react_qa_agent._ctx_key(a) != react_qa_agent._ctx_key(b)


def test_ctx_key_falls_back_without_chunk_index():
    """缺 chunk_index 时仍要给出可用且稳定的键（不返回空串）"""
    c = RetrievedContext("内容", "/x/doc.md", 0.5, "vector",
                         metadata={"doc_id": "D2", "section_title": "第二章"})
    key = react_qa_agent._ctx_key(c)
    assert key and key.startswith("D2#")


def test_ctx_key_never_uses_parent_id_alone():
    """显式反例：不能把 parent_id 当作键（否则 a/b 会相等）"""
    a = RetrievedContext("甲", "s", 0.9, "vector",
                         metadata={"doc_id": "D1", "chunk_index": 3,
                                   "parent_id": "D1#doc"})
    assert react_qa_agent._ctx_key(a) != "D1#doc"
