"""对话历史（展示保留）端到端验证 —— 走真实 HTTP

验证的是**真实 API 路径**，不是直接调函数：
  1. 带 session_id 连问 2 轮 → 历史能读回 2 轮，且含答案与出处
  2. **匿名单次问答（不传 session_id）不写历史**
  3. **actor 隔离**：换一个 Key 读同一 session_id → 0 轮
  4. 历史快照的引用是**命中章节**（不是父文档全文）
  5. 历史里带用量（恢复后的 readout 才有数字可显示）
  6. 历史**不注入 prompt**：同一个 session 内第 2 轮与全新会话问同一问题，
     生成的 prompt token 数不应因"有历史"而系统性变大

用法: python scripts/verify_chat_history.py [--url http://127.0.0.1:8080]
"""

import argparse
import sys
import uuid

import httpx

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8080")
ap.add_argument("--key", default="dev-key-1")
ap.add_argument("--other-key", default="dev-admin-key-1")
args = ap.parse_args()

URL = args.url.rstrip("/")
H = {"X-API-Key": args.key}
H_OTHER = {"X-API-Key": args.other_key}
FAILS = []

# trust_env=False 必须：本机系统代理会让 127.0.0.1 的请求也走代理而挂死
CLI = httpx.Client(timeout=300, trust_env=False)


def check(label, ok, extra=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {extra}" if extra else ""))
    if not ok:
        FAILS.append(label)


def ask(question, session_id=None, headers=None):
    body = {"question": question}
    if session_id:
        body["session_id"] = session_id
    r = CLI.post(f"{URL}/api/qa/ask", headers=headers or H, json=body)
    r.raise_for_status()
    return r.json()


def history(session_id, headers=None, limit=50):
    r = CLI.get(f"{URL}/api/ui/qa-history", headers=headers or H,
                params={"session_id": session_id, "limit": limit})
    r.raise_for_status()
    return r.json()


print("=" * 80)
print(f"目标 {URL}")
print("=" * 80)

# ── 1. 连问 2 轮 → 历史 2 轮 ─────────────────────────────────
print("1. 连问 2 轮，历史应能读回 2 轮")
print("=" * 80)
sid = uuid.uuid4().hex[:16]
r1 = ask("标准工作时间是几点到几点？", sid)
r2 = ask("年假最多有多少天？", sid)
print(f"     第1轮 turn={r1.get('turn')} 第2轮 turn={r2.get('turn')}")

d = history(sid)
turns = d.get("turns") or []
print(f"     total={d.get('total')} returned={d.get('returned')} "
      f"truncated={d.get('truncated')}")
for t in turns:
    print(f"       turn {t.get('turn')}: {str(t.get('question'))[:20]} "
          f"→ 答案 {len(str(t.get('answer')))} 字，出处 {len(t.get('sources') or [])} 条，"
          f"tokens={((t.get('usage') or {}).get('total_tokens'))}")

check("历史返回 2 轮", len(turns) == 2, str(len(turns)))
check("total=2 且未截断", d.get("total") == 2 and d.get("truncated") is False)
check("轮次升序", [t.get("turn") for t in turns] == [1, 2],
      str([t.get("turn") for t in turns]))
check("第 1 轮问题一致", turns[0].get("question") == "标准工作时间是几点到几点？")
check("答案非空", all(len(str(t.get("answer") or "")) > 0 for t in turns))
check("每轮都有出处", all((t.get("sources") or []) for t in turns))
check("出处带 title 与 content",
      all(s.get("title") and s.get("content") for t in turns for s in t["sources"]))
check("历史带用量（恢复后的 readout 才有数字）",
      all((t.get("usage") or {}).get("total_tokens") for t in turns))
check("历史带推理步骤", all(t.get("reasoning_steps") for t in turns))

# ── 2. 引用契约：存的是命中章节而不是父文档 ──────────────────
print()
print("=" * 80)
print("2. 引用契约：历史里的出处必须是「命中章节」")
print("=" * 80)
first_contents = [s["content"][:24] for t in turns for s in t["sources"]][:4]
print(f"     前几条出处开头: {first_contents}")
bad = [s for t in turns for s in t["sources"]
       if "文档说明" in s.get("content", "")[:12] and "章节" not in s.get("content", "")[:12]]
check("没有'每条引用都以文档说明开头'的退化",
      not any(s.get("content", "").startswith("## 1. 文档说明")
              for t in turns for s in t["sources"]),
      f"{len(bad)} 条可疑")

# ── 3. 匿名请求不写历史 ─────────────────────────────────────
print()
print("=" * 80)
print("3. 匿名单次问答（不传 session_id）不应写入历史")
print("=" * 80)
anon = ask("密码长度要求多少位？")
print(f"     匿名响应 session_id={anon.get('session_id')!r} turn={anon.get('turn')}")
check("匿名响应不交出 session_id", anon.get("session_id") is None)

# 匿名请求在库里的 session_id 形如 anon-xxx，客户端拿不到；
# 用任意 anon- 前缀去查也应为空（actor 隔离之外，也确认我们没往可查询空间写）
probe = history("anon-000000000000")
check("anon-* 命名空间查不到任何东西", (probe.get("total") or 0) == 0,
      str(probe.get("total")))

# ── 4. actor 隔离 ───────────────────────────────────────────
print()
print("=" * 80)
print("4. actor 隔离：换 Key 读同一 session_id")
print("=" * 80)
other = history(sid, headers=H_OTHER)
print(f"     另一个 Key: total={other.get('total')} returned={other.get('returned')}")
check("另一个 Key 读不到（total=0）", (other.get("total") or 0) == 0,
      str(other.get("total")))

# ── 5. 历史不注入 prompt（单轮语义不变）─────────────────────
print()
print("=" * 80)
print("5. 历史不注入 prompt：已有 2 轮历史的会话再问，prompt 不应因历史变大")
print("=" * 80)
r3 = ask("工资每月几号发放？", sid)          # 同一会话的第 3 轮
fresh_sid = uuid.uuid4().hex[:16]
r4 = ask("工资每月几号发放？", fresh_sid)    # 全新会话的第 1 轮

u3, u4 = r3.get("usage") or {}, r4.get("usage") or {}
p3, p4 = u3.get("prompt_tokens") or 0, u4.get("prompt_tokens") or 0
print(f"     有历史(第3轮) prompt={p3}  全新会话 prompt={p4}  差={p3 - p4}")
# 允许检索结果不同带来的正常波动；若把 2 轮历史拼进 prompt，增幅会远大于此
check("prompt 未因历史而显著变大（单轮语义不变）", abs(p3 - p4) < 1500,
      f"diff={p3 - p4}")
check("第 3 轮 turn=3", r3.get("turn") == 3, str(r3.get("turn")))

d3 = history(sid)
check("历史现在 3 轮", d3.get("total") == 3, str(d3.get("total")))

# ── 6. limit 与 truncated ───────────────────────────────────
print()
print("=" * 80)
print("6. limit 截断时返回最近 N 轮并置 truncated")
print("=" * 80)
d_lim = history(sid, limit=1)
print(f"     limit=1: total={d_lim.get('total')} returned={d_lim.get('returned')} "
      f"truncated={d_lim.get('truncated')} turns={[t.get('turn') for t in d_lim['turns']]}")
check("只返回 1 轮", d_lim.get("returned") == 1)
check("total 仍是 3", d_lim.get("total") == 3)
check("truncated=True", d_lim.get("truncated") is True)
check("返回的是最近一轮（turn=3）",
      [t.get("turn") for t in d_lim["turns"]] == [3],
      str([t.get("turn") for t in d_lim["turns"]]))

print()
print("=" * 80)
print(f"结果：{'全部通过' if not FAILS else '失败 ' + str(FAILS)}")
print("=" * 80)
sys.exit(1 if FAILS else 0)
