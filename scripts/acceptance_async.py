"""端到端验收：异步上传 + 任务查询 + 一致性恢复 + ACL"""
import json
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

BASE = "http://127.0.0.1:8080"
KEY = "dev-key-1"
ADMIN = "dev-admin-key-1"


def call(path, method="GET", data=None, key=KEY, timeout=180):
    r = urllib.request.Request(BASE + path, method=method)
    r.add_header("X-API-Key", key)
    if data is not None:
        r.add_header("Content-Type", "application/json")
        r.data = json.dumps(data).encode()
    try:
        resp = urllib.request.urlopen(r, timeout=timeout)
        return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode()[:200]}
    except Exception as e:
        return None, {"error": str(e)[:150]}


print("=" * 62)
print("端到端验收：异步架构")
print("=" * 62)

# ── 1. 异步上传 ─────────────────────────────────────────
print("\n[1] 异步上传（应立刻返回 202 + doc_id）")
doc = """# 测试用供应商制度

## 1. 准入条件

供应商需提供营业执照、税务登记证和近三年财务报表。

## 2. 评估周期

每季度对供应商进行一次绩效评估，评分低于 60 分启动整改。

## 3. 黑名单机制

连续两次评估不合格的供应商列入黑名单，两年内不得重新合作。
"""
boundary = "----BoundaryASYNC"
body = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="file"; filename="test_supplier.md"\r\n'
    f"Content-Type: text/markdown\r\n\r\n{doc}\r\n--{boundary}--\r\n"
).encode()

r = urllib.request.Request(BASE + "/api/ingest/upload", data=body, method="POST")
r.add_header("X-API-Key", KEY)
r.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
t0 = time.time()
try:
    resp = urllib.request.urlopen(r, timeout=60)
    elapsed = time.time() - t0
    d = json.loads(resp.read().decode())
    print(f"  HTTP {resp.status}  耗时 {elapsed:.2f} 秒")
    print(f"  doc_id: {d['doc_id'][:16]}...")
    print(f"  状态:   {d['status']}")
    print(f"  消息:   {d['message']}")
    doc_id = d["doc_id"]
    print(f"  ✓ {'响应及时（<3s），不再是同步阻塞' if elapsed < 3 else '⚠ 响应偏慢'}")
except Exception as e:
    print(f"  失败: {str(e)[:150]}")
    sys.exit(1)

# ── 2. 轮询任务进度 ─────────────────────────────────────
print("\n[2] 轮询任务进度（后台 worker 处理）")
final = None
for i in range(40):
    st, d = call(f"/api/jobs/{doc_id}")
    if st != 200:
        print(f"  查询失败: {d}")
        break
    print(f"  [{i*2:>2}s] status={d['status']:<14} progress={d['progress']:>3}%  "
          f"chunks={d['chunks_count']} entities={d['entities_count']} "
          f"relations={d['relations_count']} {d['error'][:40]}")
    if d["status"] in ("COMMITTED", "FAILED"):
        final = d
        break
    time.sleep(2)

if final and final["status"] == "COMMITTED":
    print(f"  ✓ 后台处理完成，耗时约 {i*2} 秒")
else:
    print(f"  ⚠ 未在预期时间内完成: {final}")

# ── 3. 问答验证（新文档是否可检索）──────────────────────
print("\n[3] 问答验证（新入库文档能否被检索）")
for q in ["供应商准入需要什么材料？", "供应商多久评估一次？"]:
    st, d = call("/api/qa/ask", "POST", {"question": q})
    if st == 200:
        print(f"\n  Q: {q}")
        print(f"  A: {d['answer'][:150]}")
        srcs = [s.get("type") for s in d.get("sources", [])]
        print(f"  引用 {len(d['sources'])} 条  构成: {srcs}")
    else:
        print(f"  Q: {q} -> {d}")

# ── 4. ACL 验证 ─────────────────────────────────────────
print("\n[4] 数据级 ACL")
st_admin, _ = call("/api/jobs", key=ADMIN)
st_user, d_user = call("/api/jobs", key=KEY)
print(f"  管理员访问 /api/jobs  -> HTTP {st_admin}")
print(f"  普通 Key 访问 /api/jobs -> HTTP {st_user} {'(正确拒绝)' if st_user == 403 else ''}")

# ── 5. 幂等验证 ─────────────────────────────────────────
print("\n[5] 幂等验证（重复上传同一内容）")
r2 = urllib.request.Request(BASE + "/api/ingest/upload", data=body, method="POST")
r2.add_header("X-API-Key", KEY)
r2.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
try:
    resp2 = urllib.request.urlopen(r2, timeout=60)
    d2 = json.loads(resp2.read().decode())
    print(f"  doc_id: {d2['doc_id'][:16]}...  duplicate={d2['duplicate']}")
    print(f"  消息:   {d2['message']}")
    print(f"  {'✓ 未重复消耗 LLM' if d2['duplicate'] else '⚠ 未识别为重复'}")
except Exception as e:
    print(f"  失败: {str(e)[:120]}")

# ── 6. 审计日志 ─────────────────────────────────────────
print("\n[6] 审计日志")
import subprocess
code = (
    "import sys; sys.path.insert(0,'.');"
    "sys.stdout.reconfigure(encoding='utf-8');"
    "from services.database import init_db;"
    "from config import settings;"
    "db=init_db(settings.sqlite_path);"
    "[print('   ', r['action'], '|', (r['actor'] or '')[:14], '|', (r['resource'] or '')[:28], '|', r['result'])"
    " for r in db.recent_audit(8)]"
)
out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                     cwd="E:\\agent-knowledge-hub-main\\python")
for line in (out.stdout or "").splitlines():
    print(line)

print()
print("=" * 62)
print("验收完成")
print("=" * 62)
