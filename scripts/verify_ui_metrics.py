"""UI 验证：对话指标是否真的渲染出来

用 CDP 驱动无头 Chrome，走**真实浏览器路径**（不是读源码猜），依次验证：
  1. 首屏 chat 视图可见（此前 app.js 没有引导代码，.view{display:none} 导致空白）
  2. 提一个问题后，答案卡片 readout 出现 8 格，其中含 轮次/Tokens/缓存命中/耗时
  3. 侧栏「本次会话」显示 对话轮数 / 累计 Tokens / 缓存命中
  4. 再问一次 -> 轮次与累计递增（服务端计号）
  5. 刷新页面 -> 侧栏数字从 /api/ui/qa-session 回填（持久化生效）
  6. 出处抽屉里有「模型调用」明细，按 stage 展开

用法: python scripts/verify_ui_metrics.py [--port 8080]
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
CDP_PORT = 9225

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8080)
args = ap.parse_args()
BASE = f"http://127.0.0.1:{args.port}/"

FAILS = []


def check(label, ok, extra=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {extra}" if extra else ""))
    if not ok:
        FAILS.append(label)


proc = subprocess.Popen(
    [CHROME, f"--remote-debugging-port={CDP_PORT}", "--headless=new", "--disable-gpu",
     "--no-first-run", "--remote-allow-origins=*", "--window-size=1440,1000",
     "--user-data-dir=" + str(Path.home() / ".chrome-ui-metrics"), "about:blank"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
time.sleep(6)

try:
    tabs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{CDP_PORT}/json", timeout=10).read())
    ws_url = next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page")

    from websocket import create_connection
    ws = create_connection(ws_url, timeout=120)
    mid = [0]

    def send(method, params=None, wait=True):
        mid[0] += 1
        cur = mid[0]
        ws.send(json.dumps({"id": cur, "method": method, "params": params or {}}))
        if not wait:
            return cur
        end = time.time() + 180
        while time.time() < end:
            try:
                msg = json.loads(ws.recv())
            except Exception:
                return {}
            if msg.get("id") == cur:
                if "error" in msg:
                    print("   CDP error:", msg["error"])
                return msg.get("result", {})
        return {}

    def js(expr, timeout_s=30):
        """求值并返回 JSON 化结果（awaitPromise 以便等待轮询函数）"""
        r = send("Runtime.evaluate", {
            "expression": expr, "returnByValue": True,
            "awaitPromise": True, "timeout": timeout_s * 1000,
        })
        return (r.get("result") or {}).get("value")

    send("Page.enable")
    send("Runtime.enable")

    # ── 1. 首屏 ──────────────────────────────────────────────
    print("=" * 78)
    print("1. 首屏渲染（app.js 此前无引导代码，chat 视图默认 display:none）")
    print("=" * 78)
    send("Page.navigate", {"url": BASE})
    time.sleep(5)

    info = js("""(() => {
      const v = document.getElementById('view-chat');
      return {
        active: !!v && v.classList.contains('is-active'),
        display: v ? getComputedStyle(v).display : 'missing',
        status: (document.getElementById('rail-status-text')||{}).textContent,
        docs: (document.getElementById('lg-docs')||{}).textContent,
        tokens: (document.getElementById('lg-tokens')||{}).textContent,
        turns: (document.getElementById('lg-turns')||{}).textContent,
      };
    })()""")
    print(f"     {info}")
    check("chat 视图已激活且可见", bool(info) and info.get("active")
          and info.get("display") != "none", str(info.get("display") if info else None))
    check("侧栏状态不再是「正在检查服务」",
          bool(info) and info.get("status") not in (None, "正在检查服务"),
          str(info.get("status") if info else None))

    # ── 2. 提问 ──────────────────────────────────────────────
    print()
    print("=" * 78)
    print("2. 提问后答案卡片的用量 readout")
    print("=" * 78)
    js("""(() => {
      const q = document.getElementById('question');
      q.value = '年假最多有多少天？';
      ask();
      return true;
    })()""")

    res = js("""(async () => {
      const t0 = Date.now();
      while (Date.now() - t0 < 150000) {
        const dl = document.querySelector('.qa .readout');
        if (dl) {
          const cells = [...dl.querySelectorAll('.cell')].map(c => ({
            k: c.querySelector('dt').textContent.trim(),
            v: c.querySelector('dd').textContent.trim(),
            title: c.querySelector('dd').getAttribute('title') || '',
          }));
          const rail = {
            turns: document.getElementById('lg-turns').textContent.trim(),
            tokens: document.getElementById('lg-tokens').textContent.trim(),
            cache: document.getElementById('lg-cache').textContent.trim(),
          };
          return {cells, rail};
        }
        if (document.querySelector('.qa .answer.is-error')) {
          return {error: document.querySelector('.qa .answer.is-error').textContent};
        }
        await new Promise(r => setTimeout(r, 800));
      }
      return {error: 'timeout waiting for readout'};
    })()""", timeout_s=180)

    if not res or res.get("error"):
        check("答案返回", False, str(res))
    else:
        keys = [c["k"] for c in res["cells"]]
        print(f"     readout 列: {keys}")
        for c in res["cells"]:
            print(f"       {c['k']:<8} = {c['v']:<14} title={c['title'][:60]}")
        print(f"     侧栏: {res['rail']}")
        check("readout 8 格", len(res["cells"]) == 8, str(len(res["cells"])))
        for want in ("轮次", "Tokens", "缓存命中", "耗时"):
            check(f"含「{want}」格", want in keys)
        check("轮次=第 1 轮", res["cells"][keys.index("轮次")]["v"] == "第 1 轮",
              res["cells"][keys.index("轮次")]["v"])
        tok = res["cells"][keys.index("Tokens")]["v"]
        check("Tokens 显示具体数字", tok not in ("—", ""), tok)
        cache = res["cells"][keys.index("缓存命中")]["v"]
        check("缓存命中显示百分比或 —", cache.endswith("%") or cache == "—", cache)
        check("缓存格 tooltip 说明是 Token 口径",
              "Hit / (Hit + Miss)" in res["cells"][keys.index("缓存命中")]["title"])
        check("侧栏对话轮数已更新", res["rail"]["turns"].endswith("轮"), res["rail"]["turns"])
        check("侧栏累计 Tokens 已更新", res["rail"]["tokens"] not in ("—", ""),
              res["rail"]["tokens"])
        check("侧栏缓存命中已更新", res["rail"]["cache"] not in ("—", ""),
              res["rail"]["cache"])

    # ── 3. 第二次提问 -> 递增 ────────────────────────────────
    print()
    print("=" * 78)
    print("3. 第二次提问：轮次与累计递增（服务端计号）")
    print("=" * 78)
    js("""(() => {
      document.getElementById('question').value = '密码长度要求多少位？';
      ask();
      return true;
    })()""")

    res2 = js("""(async () => {
      const t0 = Date.now();
      while (Date.now() - t0 < 150000) {
        const cards = document.querySelectorAll('.qa .readout');
        if (cards.length >= 2) {
          const dl = cards[cards.length - 1];
          const cells = [...dl.querySelectorAll('.cell')].map(c => ({
            k: c.querySelector('dt').textContent.trim(),
            v: c.querySelector('dd').textContent.trim(),
          }));
          return {cells, rail: {
            turns: document.getElementById('lg-turns').textContent.trim(),
            tokens: document.getElementById('lg-tokens').textContent.trim(),
            cache: document.getElementById('lg-cache').textContent.trim(),
          }};
        }
        await new Promise(r => setTimeout(r, 800));
      }
      return {error: 'timeout'};
    })()""", timeout_s=180)

    if not res2 or res2.get("error"):
        check("第二次答案返回", False, str(res2))
    else:
        keys2 = [c["k"] for c in res2["cells"]]
        print(f"     第二条 readout 轮次 = {res2['cells'][keys2.index('轮次')]['v']}")
        print(f"     侧栏: {res2['rail']}")
        check("轮次=第 2 轮", res2["cells"][keys2.index("轮次")]["v"] == "第 2 轮",
              res2["cells"][keys2.index("轮次")]["v"])
        check("侧栏轮数=2 轮", res2["rail"]["turns"].startswith("2"),
              res2["rail"]["turns"])

    # ── 4. 出处抽屉的调用明细 ────────────────────────────────
    print()
    print("=" * 78)
    print("4. 出处抽屉「模型调用」明细")
    print("=" * 78)
    calls = js("""(() => {
      const box = document.getElementById('prov-body');
      const t = box.querySelector('.call-table');
      if (!t) return {found:false, html: box.innerHTML.slice(0,200)};
      return {
        found: true,
        head: [...t.querySelectorAll('th')].map(x => x.textContent.trim()),
        rows: [...t.querySelectorAll('tbody tr')].map(tr =>
          [...tr.querySelectorAll('td')].map(td => td.textContent.trim().replace(/\\s+/g,' '))),
        note: (box.querySelector('.calls-note')||{}).textContent || '',
        h3: (box.querySelector('.calls h3')||{}).textContent || '',
      };
    })()""")
    if calls and calls.get("found"):
        print(f"     {calls['h3']}")
        print(f"     表头 {calls['head']}")
        for r in calls["rows"]:
            print(f"       {r}")
        print(f"     注: {calls['note']}")
        check("明细表存在", True)
        check("明细 3 行（intent/rewrite/generate）", len(calls["rows"]) == 3,
              str(len(calls["rows"])))
        check("stage 列有值",
              {r[1] for r in calls["rows"]} == {"intent", "rewrite", "generate"},
              str([r[1] for r in calls["rows"]]))
    else:
        check("调明明细表存在", False, str(calls)[:200])

    # ── 5. 用量拆解（必须在刷新之前查：刷新后抽屉是空的）────────
    print()
    print("=" * 78)
    print("5. 出处抽屉「用量拆解」（① 总命中 / ② 稳定前缀 / ③ 检索上下文）")
    print("=" * 78)
    bd = js("""(() => {
      const box = document.getElementById('prov-body');
      const el = box.querySelector('.breakdown');
      if (!el) return {found:false, html: box.innerHTML.slice(0,300)};
      return {
        found: true,
        title: (el.querySelector('h3')||{}).textContent || '',
        rows: [...el.querySelectorAll('.breakdown-row')].map(r => [
          (r.querySelector('dt')||{}).textContent,
          (r.querySelector('dd')||{}).textContent,
        ]),
      };
    })()""")
    if bd and bd.get("found"):
        print(f"     {bd['title']}")
        for k, v in bd["rows"]:
            print(f"       {k:<26} = {v}")
        keys = [k for k, _ in bd["rows"]]
        check("用量拆解块存在", True)
        check("含 ① 总命中率", any("①" in k for k in keys))
        check("含 ② 稳定前缀复用", any("②" in k for k in keys))
        check("含 ③ 检索上下文复用", any("③" in k for k in keys))
        check("缓存写入侧未显示为 0（应为「未提供」）",
              not any("缓存写入" in k and v.strip() == "0" for k, v in bd["rows"]),
              str([v for k, v in bd["rows"] if "缓存写入" in k]))
    else:
        check("用量拆解块存在", False, str(bd)[:200])

    # ── 6. 刷新后：侧栏回填 + 历史恢复 ───────────────────────
    print()
    print("=" * 78)
    print("6. 刷新页面：侧栏回填 + 历史问答卡片恢复")
    print("=" * 78)
    send("Page.navigate", {"url": BASE})
    time.sleep(7)
    after = js("""(async () => {
      const t0 = Date.now();
      // 等侧栏数字与历史卡片都就位
      while (Date.now() - t0 < 30000) {
        const turns = document.getElementById('lg-turns').textContent.trim();
        const cards = document.querySelectorAll('.qa').length;
        if (turns !== '—' && cards > 0) break;
        await new Promise(r => setTimeout(r, 500));
      }
      const cards = [...document.querySelectorAll('.qa')];
      return {
        turns: document.getElementById('lg-turns').textContent.trim(),
        tokens: document.getElementById('lg-tokens').textContent.trim(),
        cache: document.getElementById('lg-cache').textContent.trim(),
        cards: cards.length,
        sid: sessionStorage.getItem('akh_session'),
        welcome: !!document.getElementById('welcome'),
        cellCounts: cards.map(c => c.querySelectorAll('.readout .cell').length),
        citeCounts: cards.map(c => c.querySelectorAll('.cite').length),
        turnLabels: cards.map(c => {
          const cells = [...c.querySelectorAll('.readout .cell')];
          const t = cells.find(x => x.querySelector('dt').textContent.trim() === '轮次');
          return t ? t.querySelector('dd').textContent.trim() : null;
        }),
        provItems: document.querySelectorAll('#prov-body .prov-item').length,
        breakdown: !!document.querySelector('#prov-body .breakdown'),
      };
    })()""")
    print(f"     {after}")
    check("刷新后侧栏回填轮数=2", str(after.get("turns", "")).startswith("2"),
          str(after.get("turns")))
    check("刷新后侧栏回填 Tokens", after.get("tokens") not in ("—", "", None),
          str(after.get("tokens")))
    check("刷新后侧栏回填缓存命中", after.get("cache") not in ("—", "", None),
          str(after.get("cache")))
    check("sessionStorage 持有会话 id", bool(after.get("sid")))
    check("刷新后恢复出 2 张历史卡片", after.get("cards") == 2,
          str(after.get("cards")))
    check("历史卡片 readout 仍是 8 格",
          after.get("cellCounts") == [8, 8], str(after.get("cellCounts")))
    check("历史卡片轮次正确", after.get("turnLabels") == ["第 1 轮", "第 2 轮"],
          str(after.get("turnLabels")))
    check("历史卡片有出处按钮",
          all(n > 0 for n in (after.get("citeCounts") or [])),
          str(after.get("citeCounts")))
    check("welcome 已随历史恢复移除", after.get("welcome") is False)
    check("抽屉恢复为最后一轮的出处", (after.get("provItems") or 0) > 0,
          str(after.get("provItems")))
    check("抽屉含用量拆解", after.get("breakdown") is True)

    print()
    print("=" * 78)
    print(f"UI 结果：{'全部通过' if not FAILS else '失败 ' + str(FAILS)}")
    print("=" * 78)
finally:
    proc.terminate()

sys.exit(1 if FAILS else 0)
