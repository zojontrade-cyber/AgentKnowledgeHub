"""查看持久化层状态"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
sys.stdout.reconfigure(encoding="utf-8")

from config import settings
from services.database import init_db

db = init_db(settings.sqlite_path)

print("=== 审计日志（最近 10 条）===")
for r in db.recent_audit(10):
    ts = time.strftime("%H:%M:%S", time.localtime(r["ts"]))
    actor = (r["actor"] or "-")[:16]
    res = (r["resource"] or "-")[:24]
    print(f"  {ts}  {r['action']:<12} {actor:<18} {res:<26} {r['result']}")

print()
print("=== 文档状态分布 ===")
for r in db.fetchall("SELECT status, COUNT(*) c FROM documents GROUP BY status"):
    print(f"  {r['status']:<14} {r['c']}")

print()
print("=== 实体别名表（消歧记录）===")
aliases = db.all_aliases()
print(f"  共 {len(aliases)} 条")
for r in aliases[:12]:
    cn = str(r["canonical_name"])[:20]
    al = str(r["alias"])[:20]
    print(f"  {cn:<22} <- {al:<22} [{r['entity_type']}]")

print()
print("=== Outbox 状态 ===")
stats = db.outbox_stats()
print(f"  {stats if stats else '(空)'}")
