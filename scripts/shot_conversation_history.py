"""截图：刷新后恢复的对话历史

用同一个 sessionStorage 会话先问两轮，再刷新页面，截取"历史被恢复"的效果。
产物：docs/screenshots/08-conversation-history.png
"""

import base64
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
CDP_PORT = 9227
BASE = "http://127.0.0.1:8080/"
OUT = Path(__file__).resolve().parent.parent / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)

proc = subprocess.Popen(
    [CHROME, f"--remote-debugging-port={CDP_PORT}", "--headless=new", "--disable-gpu",
     "--no-first-run", "--remote-allow-origins=*", "--window-size=1600,1100",
     "--user-data-dir=" + str(Path.home() / ".chrome-history-shot"), "about:blank"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
time.sleep(6)

try:
    tabs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{CDP_PORT}/json", timeout=10).read())
    ws_url = next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page")
    from websocket import create_connection
    ws = create_connection(ws_url, timeout=180)
    mid = [0]

    def send(method, params=None):
        mid[0] += 1
        cur = mid[0]
        ws.send(json.dumps({"id": cur, "method": method, "params": params or {}}))
        end = time.time() + 240
        while time.time() < end:
            try:
                msg = json.loads(ws.recv())
            except Exception:
                return {}
            if msg.get("id") == cur:
                return msg.get("result", {})
        return {}

    def js(expr, timeout_s=240):
        r = send("Runtime.evaluate", {"expression": expr, "returnByValue": True,
                                      "awaitPromise": True, "timeout": timeout_s * 1000})
        return (r.get("result") or {}).get("value")

    send("Page.enable")
    send("Runtime.enable")
    send("Page.navigate", {"url": BASE})
    time.sleep(5)

    # 问两轮
    for q in ("产品上线流程中有哪些关键审批节点？", "年假最多有多少天？"):
        js(f"""(() => {{
          document.getElementById('question').value = {json.dumps(q, ensure_ascii=False)};
          ask(); return true;
        }})()""")
        js("""(async () => {
          const before = document.querySelectorAll('.qa .readout').length;
          const t0 = Date.now();
          while (Date.now() - t0 < 180000) {
            if (document.querySelectorAll('.qa .readout').length > before) break;
            await new Promise(r => setTimeout(r, 700));
          }
          return document.querySelectorAll('.qa .readout').length;
        })()""")
        time.sleep(1)

    # 刷新 —— 卡片应当被恢复
    send("Page.navigate", {"url": BASE})
    time.sleep(9)
    js("""(async () => {
      const t0 = Date.now();
      while (Date.now() - t0 < 40000) {
        if (document.querySelectorAll('.qa').length >= 2) break;
        await new Promise(r => setTimeout(r, 500));
      }
      return document.querySelectorAll('.qa').length;
    })()""")
    js("""(() => { const s = document.getElementById('ask-scroll');
                    s.scrollTop = s.scrollHeight; return true; })()""")
    time.sleep(1)

    shot = send("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
    data = shot.get("data")
    if not data:
        print("截图失败：CDP 未返回数据")
        sys.exit(1)
    path = OUT / "08-conversation-history.png"
    path.write_bytes(base64.b64decode(data))
    print(f"已保存 {path}")
finally:
    proc.terminate()
