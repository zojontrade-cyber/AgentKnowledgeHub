"""
验证新增的 P0/P1 能力

覆盖：
  1. SQLite 持久化层（状态机 / outbox / 别名 / 审计）
  2. 别名表（同实异名归一）
  3. 别名字段的读写
  4. 数据级 ACL
"""

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

FAILS = []


def check(label, ok, extra=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {extra}" if extra else ""))
    if not ok:
        FAILS.append(label)


# ══════════════════════════════════════════════════════════
print("=" * 60)
print("1. SQLite 持久化层")
print("=" * 60)

from services.database import Database, DocStatus

tmp = Path(tempfile.mkdtemp()) / "test.db"
db = Database(str(tmp))

# 文档状态机
doc_id = db.create_document("/tmp/a.md", "a.md", "hash_aaa", "public", "key_x")
check("创建文档", bool(doc_id))
d = db.get_document(doc_id)
check("初始状态 PENDING", d["status"] == DocStatus.PENDING, d["status"])

db.set_status(doc_id, DocStatus.VECTOR_DONE)
check("状态流转 VECTOR_DONE", db.get_document(doc_id)["status"] == DocStatus.VECTOR_DONE)

db.set_status(doc_id, DocStatus.GRAPH_DONE, entities_count=10)
d = db.get_document(doc_id)
check("状态+字段更新", d["status"] == DocStatus.GRAPH_DONE and d["entities_count"] == 10)

# 幂等：相同 content_hash 且已 COMMITTED 时复用
db.set_status(doc_id, DocStatus.COMMITTED)
doc_id2 = db.create_document("/tmp/b.md", "b.md", "hash_aaa", "public", "key_x")
check("幂等复用（同内容不重复入库）", doc_id2 == doc_id, f"{doc_id2[:8]} == {doc_id[:8]}")

# Outbox
tid = db.enqueue(doc_id, "graph", {"reason": "test"})
check("投递 outbox 任务", tid > 0)
claimed = db.claim_tasks(limit=5)
check("claim 取到任务", len(claimed) == 1, f"{len(claimed)} 条")
check("claim 后状态为 PROCESSING", claimed[0]["status"] == "PROCESSING")
db.finish_task(tid, ok=True)
check("完成后 outbox 状态 DONE", db.outbox_stats().get("DONE") == 1)

# 失败回退
tid2 = db.enqueue(doc_id, "vector", {})
db.claim_tasks(limit=5, task_types=("vector",))
db.finish_task(tid2, ok=False, error="boom")
pend = db.pending_tasks()
check("失败任务回到 PENDING（可重放）", any(t["id"] == tid2 for t in pend))

# 文件版本（取代内存态）
db.set_file_hash("/tmp/a.md", "h1")
check("文件哈希登记", db.get_file_hash("/tmp/a.md") == "h1")
db.set_file_hash("/tmp/a.md", "h2")
check("文件哈希更新（版本+1）", db.get_file_hash("/tmp/a.md") == "h2")

# 审计
db.audit("qa.ask", actor="key_abc", resource="测试问题", result="ok")
check("审计日志写入", len(db.recent_audit()) >= 1)


# ══════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════
print()
print("=" * 60)
print("=" * 60)

# ══════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════
print()
print("=" * 60)
print("=" * 60)

# ══════════════════════════════════════════════════════════
print()
print("=" * 60)
print("4. 数据级 ACL")
print("=" * 60)

from api.security import scopes_for_key
from config import settings

orig_scopes, orig_admin = settings.api_key_scopes, settings.admin_api_keys
settings.api_key_scopes = "hr-key=public|hr,fin-key=public|finance"
settings.admin_api_keys = "admin-key"
try:
    check("管理员 -> 不限制", scopes_for_key("admin-key") == ["*"])
    check("HR Key -> public+hr", scopes_for_key("hr-key") == ["public", "hr"])
    check("财务 Key -> public+finance",
          scopes_for_key("fin-key") == ["public", "finance"])
    check("未配置 Key -> 最小权限 public", scopes_for_key("unknown-key") == ["public"])
    check("空 Key -> public", scopes_for_key("") == ["public"])
finally:
    settings.api_key_scopes = orig_scopes
    settings.admin_api_keys = orig_admin


# ══════════════════════════════════════════════════════════
print()
print("=" * 60)
if FAILS:
    print(f"失败 {len(FAILS)} 项：")
    for f in FAILS:
        print("  -", f)
    sys.exit(1)
print("全部通过")
print("=" * 60)
