"""端到端验收：模拟用户真实操作流程"""
import json
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")

BASE = "http://127.0.0.1:8080"
KEY = "dev-key-1"


def req(path, method="GET", data=None, timeout=180):
    r = urllib.request.Request(BASE + path, method=method)
    r.add_header("X-API-Key", KEY)
    if data is not None:
        r.add_header("Content-Type", "application/json")
        r.data = json.dumps(data).encode()
    try:
        resp = urllib.request.urlopen(r, timeout=timeout)
        return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode()[:200]}
    except Exception as e:
        return None, {"error": str(e)[:200]}


print("=" * 60)
print("端到端验收测试")
print("=" * 60)

# 1. 概览
print("\n[1] 系统概览")
st, d = req("/api/ui/overview")
if st == 200:
    print(f"  向量数:   {d['vector_store'].get('total_vectors')}")
    print(f"  已入库文件: {d['uploaded_files']}")
else:
    print(f"  失败: {d}")

# 2. 上传新文档（验证解析器改动）
print("\n[2] 上传新文档（验证语义切分 + 小大分块）")
import urllib.request as ur
boundary = "----WebKitFormBoundaryTEST"
doc = """# 测试用差旅制度

## 1. 适用范围

本制度适用于公司全体正式员工的国内出差活动。

## 2. 交通标准

飞机仅限经济舱，高铁仅限二等座。部门负责人及以上可乘坐商务座。

## 3. 住宿标准

一线城市每晚不超过 500 元，二线城市不超过 350 元，其他城市不超过 250 元。

## 4. 报销时限

出差结束后 10 个工作日内提交报销单，附行程单、发票和审批记录。
"""
body = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="file"; filename="test_travel.md"\r\n'
    f"Content-Type: text/markdown\r\n\r\n"
    f"{doc}\r\n"
    f"--{boundary}--\r\n"
).encode("utf-8")

r = ur.Request(BASE + "/api/ingest/upload", data=body, method="POST")
r.add_header("X-API-Key", KEY)
r.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
try:
    resp = ur.urlopen(r, timeout=180)
    d = json.loads(resp.read().decode())
    print(f"  上传成功: {d['file_name']}")
    print(f"    文档块: {d['chunks_count']}  实体: {d['entities_count']}  关系: {d['relations_count']}")
except Exception as e:
    print(f"  上传失败: {str(e)[:150]}")

# 3. 提问（验证检索 + 生成）
print("\n[3] 智能问答")
questions = [
    "出差住宿费能报多少？",
    "报销单多久内要提交？",
    "出差坐飞机有什么限制？",
    "公司去年的营收是多少？",   # 负样本
]
for q in questions:
    st, d = req("/api/qa/ask", "POST", {"question": q})
    if st == 200:
        print(f"\n  Q: {q}")
        print(f"  意图: {d['intent']}  置信度: {d['confidence']:.2f}  引用: {len(d['sources'])} 条")
        print(f"  A: {d['answer'][:180]}")
        for s in d["reasoning_steps"]:
            print(f"     · {s}")
    else:
        print(f"\n  Q: {q}  -> 失败 {d}")

print()
print("=" * 60)
print("验收完成")
print("=" * 60)
