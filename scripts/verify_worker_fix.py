"""
验证 worker 修复：能否看到并处理那 20 个被卡住的 PENDING 文档。

不启动完整 API（避免长跑），只验证：
  1. 新的 list_pending_documents() 能否取到之前取不到的文档
  2. reclaim_stuck_documents() 能否复位卡死的 PROCESSING
"""

import os
import sys
from pathlib import Path

_PY = Path(Path(__file__).resolve().parent.parent / "python")
os.chdir(_PY)
sys.path.insert(0, str(_PY))
sys.stdout.reconfigure(encoding="utf-8")

from config import settings
from services.database import DocStatus, get_db, init_db

init_db(settings.sqlite_path)
db = get_db()

print("=" * 84)
print("修复前 vs 修复后：worker 能否看到 PENDING")
print("=" * 84)

# 旧逻辑（复刻 bug）
old = [d for d in db.list_documents(limit=5, acl_scopes=None)
       if d["status"] == DocStatus.PENDING]
print(f"  旧逻辑 [list_documents(5) 后筛]  取到 {len(old)} 篇")

# 新逻辑
new = db.list_pending_documents(limit=5)
print(f"  新逻辑 [list_pending_documents]  取到 {len(new)} 篇")
print()
for d in new:
    print(f"     {str(d['source_path'])[-46:]}  ({d['status']})")

print()
print("=" * 84)
print("卡死的 PROCESSING 文档")
print("=" * 84)
stuck = [d for d in db.list_documents(limit=200) if d["status"] == DocStatus.PROCESSING]
for d in stuck:
    import time
    age = time.time() - (d.get("updated_at") or 0)
    print(f"  {str(d['source_path'])[-46:]}")
    print(f"     已卡住 {age/60:.1f} 分钟   (updated_at={d.get('updated_at')})")

print()
print("=" * 84)
print("执行回收（阈值 30 分钟）")
print("=" * 84)
n = db.reclaim_stuck_documents(timeout_seconds=1800)
print(f"  回收 {n} 篇")
n2 = db.reclaim_stuck_documents(timeout_seconds=1800)
print(f"  再次调用（应为 0，幂等）: {n2}")

print()
print("=" * 84)
print("回收后状态")
print("=" * 84)
import sqlite3
conn = sqlite3.connect(str(Path(settings.sqlite_path)))
for r in conn.execute("SELECT status, COUNT(*) FROM documents GROUP BY status"):
    print(f"  {r[0]:<14} {r[1]}")
conn.close()
