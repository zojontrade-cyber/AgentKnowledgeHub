"""用 Chrome DevTools Protocol 截图 Web UI 各页面"""

import base64
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PORT = 9224
BASE = "http://127.0.0.1:8080/"
OUT = Path(__file__).resolve().parent.parent / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)

proc = subprocess.Popen(
    [CHROME, f"--remote-debugging-port={PORT}", "--headless=new", "--disable-gpu",
     "--no-first-run", "--remote-allow-origins=*", "--window-size=1440,900",
     "--user-data-dir=" + str(Path.home() / ".chrome-shot"), "about:blank"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
time.sleep(6)

try:
    tabs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json", timeout=10).read())
    ws_url = next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page")

    from websocket import create_connection
    ws = create_connection(ws_url, timeout=60)
    mid = [0]

    def send(method, params=None, wait=True):
        mid[0] += 1
        cur = mid[0]
        ws.send(json.dumps({"id": cur, "method": method, "params": params or {}}))
        if not wait:
            return cur
        end = time.time() + 30
        while time.time() < end:
            try:
                msg = json.loads(ws.recv())
                if msg.get("id") == cur:
                    return msg.get("result", {})
            except Exception:
                break
        return {}

    send("Page.enable")
    send("Emulation.setDeviceMetricsOverride",
         {"width": 1440, "height": 900, "deviceScaleFactor": 1, "mobile": False})
    send("Page.navigate", {"url": BASE})
    time.sleep(6)

    def shot(name):
        r = send("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
        data = r.get("data")
        if data:
            (OUT / name).write_bytes(base64.b64decode(data))
            print(f"  [OK] {name}")
        else:
            print(f"  [FAIL] {name}")

    # 1. 问答页
    shot("01-chat.png")

    # 2. 上传页
    send("Runtime.evaluate", {
        "expression": "[...document.querySelectorAll('nav button')].find(b=>b.dataset.view==='upload').click()"
    })
    time.sleep(3)
    shot("02-upload.png")

    # 3. 概览页
    send("Runtime.evaluate", {
        "expression": "[...document.querySelectorAll('nav button')].find(b=>b.dataset.view==='overview').click()"
    })
    time.sleep(4)
    shot("03-overview.png")


    # 5. 问答演示
    send("Runtime.evaluate", {
        "expression": """
        (() => {
          [...document.querySelectorAll('nav button')].find(b=>b.dataset.view==='chat').click();
          document.getElementById('question').value = '张三负责什么？';
          document.getElementById('ask-btn').click();
          return 'ok';
        })()
        """
    })
    time.sleep(16)
    shot("05-answer.png")

    ws.close()
finally:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()

print()
print(f"截图输出目录: {OUT}")
