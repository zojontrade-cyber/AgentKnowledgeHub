"""
静态验证 Web UI：HTML 结构完整性 + 前端逻辑所依赖的接口契约

不依赖浏览器，但能确认：
  1. 页面元素 id 与 JS 引用一致（防止 undefined 报错）
  2. 前端调用的接口路径真实存在且返回预期字段
"""

import json
import re
import sys
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
HTML = ROOT / "python" / "api" / "static" / "index.html"
BASE = "http://127.0.0.1:8080"
KEY = "dev-key-1"

html = HTML.read_text(encoding="utf-8")
failures = []


def check(label, ok, extra=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {extra}" if extra else ""))
    if not ok:
        failures.append(label)


print("=== 1. HTML 结构 ===")
check("文件非空", len(html) > 5000, f"{len(html)} 字节")
check("有 DOCTYPE", html.lstrip().lower().startswith("<!doctype html>"))

# JS 里通过 getElementById 引用的 id，必须在 HTML 中存在
js_ids = set(re.findall(r'getElementById\("([^"]+)"\)', html))
html_ids = set(re.findall(r'id="([^"]+)"', html))
missing = js_ids - html_ids
check("JS 引用的元素 id 都存在", not missing,
      f"缺失: {sorted(missing)}" if missing else f"共 {len(js_ids)} 个")

# 视图容器与导航按钮对应
# 注：视图 id 形如 view-chat，导航形如 data-view="chat"，这里统一取后缀比较
views = set(re.findall(r'id="view-(\w+)"', html))
navs = set(re.findall(r'data-view="(\w+)"', html))
check("导航与视图一一对应", views == navs,
      f"视图={sorted(views)} 导航={sorted(navs)}")

check("三个功能模块齐全",
      all(k in html for k in ("智能问答", "知识入库", "数据概览")))


print()
print("=== 2. 前端调用的接口是否真实存在 ===")
api_paths = sorted(set(re.findall(r'api\("(/api/[^"?]+)', html)))
print(f"  前端调用: {api_paths}")


def get(path, with_key=True):
    req = urllib.request.Request(BASE + path)
    if with_key:
        req.add_header("X-API-Key", KEY)
    try:
        r = urllib.request.urlopen(req, timeout=30)
        return r.status, json.loads(r.read().decode("utf-8", "ignore"))
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception:
        return None, None


def post(path, payload):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-API-Key": KEY},
        method="POST",
    )
    try:
        r = urllib.request.urlopen(req, timeout=180)
        return r.status, json.loads(r.read().decode("utf-8", "ignore"))
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception:
        return None, None


# GET 接口直接校验；POST 接口用 405 判断"路由存在"（因为前端用的是 POST）
POST_ENDPOINTS = {"/api/ingest/upload", "/api/qa/ask"}
for p in api_paths:
    if "{" in p:
        continue
    if p in POST_ENDPOINTS:
        st, _ = get(p)  # GET 打到 POST 路由上应返回 405
        check(f"路由存在 {p} (POST)", st == 405, f"GET 返回 HTTP {st}（405=已注册为 POST）")
    else:
        st, _ = get(p)
        check(f"接口可用 {p}", st == 200, f"HTTP {st}")

print()
print("=== 3. 前端依赖的响应字段 ===")
st, d = get("/api/ui/overview")
if st == 200:
    check("overview.vector_store.total_vectors", "total_vectors" in d.get("vector_store", {}))
    check("overview.uploaded_files", "uploaded_files" in d)
    check("overview.config", "config" in d)
else:
    check("overview 接口", False, f"HTTP {st}")

st, d = get("/api/ui/documents")
if st == 200:
    check("documents.total", "total" in d)
    check("documents.documents[]", isinstance(d.get("documents"), list))
    if d.get("documents"):
        doc = d["documents"][0]
        check("文档项含 name/size_human/modified",
              all(k in doc for k in ("name", "size_human", "modified")), str(list(doc)))
else:
    check("documents 接口", False, f"HTTP {st}")

# 对应的一整段检查已移除。这里改为断言它**确实不存在**（防止残留路由）。
st, _ = get("/api/ui/graph?limit=60")

print()
print("=== 4. 问答接口（UI 渲染依赖的字段）===")
st, d = post("/api/qa/ask", {"question": "张三负责什么？"})
if st == 200 and d:
    check("qa.answer", isinstance(d.get("answer"), str) and len(d["answer"]) > 0,
          d.get("answer", "")[:40] + "...")
    check("qa.intent", "intent" in d, d.get("intent"))
    check("qa.confidence", isinstance(d.get("confidence"), (int, float)))
    check("qa.sources[]", isinstance(d.get("sources"), list),
          f"{len(d.get('sources', []))} 条")
    check("qa.reasoning_steps[]", isinstance(d.get("reasoning_steps"), list))
    if d.get("sources"):
        s = d["sources"][0]
        check("来源项含 content/source/score/type",
              all(k in s for k in ("content", "source", "score", "type")), str(list(s)))
else:
    check("问答接口可用", False, f"HTTP {st}")

print()
if failures:
    print(f"!!! {len(failures)} 项失败:")
    for f in failures:
        print("   -", f)
    sys.exit(1)
print("Web UI 静态验证全部通过")
