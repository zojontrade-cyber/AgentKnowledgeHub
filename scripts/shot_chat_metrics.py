"""截图：对话指标（轮次 / Tokens / 缓存命中）在 UI 上的呈现

问一个问题后截图，便于离线查看结果。产物：docs/screenshots/07-chat-metrics.png
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
CDP_PORT = 9226
BASE = "http://127.0.0.1:8080/"
OUT = Path(__file__).resolve().parent.parent / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)

proc = subprocess.Popen(
    [CHROME, f"--remote-debugging-port={CDP_PORT}", "--headless=new", "--disable-gpu",
     "--no-first-run", "--remote-allow-origins=*", "--window-size=1600,1100",
     "--user-data-dir=" + str(Path.home() / ".chrome-qa-shot"), "about:blank"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
time.sleep(6)

try:
    tabs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{CDP_PORT}/json", timeout=10).read())
    ws_url = next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page")
    from websocket import create_connection
    ws = create_connection(ws_url, timeout=120)
    mid = [0]

    def send(method, params=None):
        mid[0] += 1
        cur = mid[0]
        ws.send(json.dumps({"id": cur, "method": method, "params": params or {}}))
        end = time.time() + 180
        while time.time() < end:
            try:
                msg = json.loads(ws.recv())
            except Exception:
                return {}
            if msg.get("id") == cur:
                return msg.get("result", {})
        return {}

    def js(expr, timeout_s=180):
        r = send("Runtime.evaluate", {"expression": expr, "returnByValue": True,
                                      "awaitPromise": True, "timeout": timeout_s * 1000})
        return (r.get("result") or {}).get("value")

    send("Page.enable")
    send("Runtime.enable")
    send("Page.navigate", {"url": BASE})
    time.sleep(5)

    js("""(() => {
      document.getElementById('question').value = '年假最多有多少天？';
      ask(); return true;
    })()""")
    js("""(async () => {
      const t0 = Date.now();
      while (Date.now() - t0 < 150000) {
        if (document.querySelector('.qa .readout')) break;
        await new Promise(r => setTimeout(r, 800));
      }
      return document.querySelectorAll('.qa .readout').length;
    })()""")
    # 展开出处抽屉，并滚到底部，让「模型调用」与「用量拆解」都进画面
    js("(() => { const b = document.querySelector('.cite-open'); if (b) b.click(); return true; })()")
    time.sleep(2)
    js("(() => { const b = document.getElementById('prov-body'); if (b) b.scrollTop = b.scrollHeight; return true; })()")
    time.sleep(1)

    shot = send("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
    data = shot.get("data")
    if not data:
        print("截图失败：CDP 未返回数据")
        sys.exit(1)
    path = OUT / "07-chat-metrics.png"
    path.write_bytes(base64.b64decode(data))
    print(f"已保存 {path}")
finally:
    proc.terminate()
