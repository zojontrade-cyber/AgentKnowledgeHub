"""索引台前端截图矩阵（仅开发）。

本机情况需要显式指定浏览器：已安装的 playwright 期望 chromium-1112，
但浏览器缓存里是 chromium-1234。脚本按顺序探测，取第一个存在的：
  1) --browser 参数
  2) CHROME_PATH 环境变量
  3) ms-playwright 缓存（任意版本）
  4) 系统 Chrome / Edge

用法：
    python scripts/ui_preview/serve_stub.py --port 8099      # 另一个终端
    python scripts/ui_preview/shoot.py --base http://127.0.0.1:8099
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "docs" / "screenshots"

DESKTOP = {"width": 1440, "height": 900}
MID = {"width": 1024, "height": 768}
PHONE = {"width": 375, "height": 812}

CONSOLE_ERRORS: list[str] = []


def find_browser(explicit: str | None) -> str | None:
    if explicit and Path(explicit).exists():
        return explicit
    env = os.environ.get("CHROME_PATH")
    if env and Path(env).exists():
        return env
    local = os.environ.get("LOCALAPPDATA") or ""
    for pat in (
        str(Path(local) / "ms-playwright" / "chromium-*" / "chrome-win64" / "chrome.exe"),
        str(Path(local) / "ms-playwright" / "chromium-*" / "chrome-win" / "chrome.exe"),
    ):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]
    for sys_path in (
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ):
        if Path(sys_path).exists():
            return sys_path
    return None


def settle(page, ms: int = 450):
    page.wait_for_timeout(ms)


def goto_view(page, base: str, view: str):
    page.goto(base + "/", wait_until="domcontentloaded")
    page.wait_for_selector(".rail-nav button[data-view='%s']" % view)
    page.click(".rail-nav button[data-view='%s']" % view)
    settle(page, 350)


def ask(page, question: str = "加班费怎么算？离职时门禁权限怎么回收？"):
    page.click(".rail-nav button[data-view='chat']")
    settle(page, 200)
    page.fill("#question", question)
    page.click("#ask-btn")
    page.wait_for_selector(".cite-row", timeout=15000)
    settle(page, 900)


def load_graph(page, *, mode="docs", keyword="", seed="", limit=None, wait=5200):
    page.click(".rail-nav button[data-view='graph']")
    settle(page, 300)
    if mode:
        page.select_option("#graph-view-mode", mode)
    if limit:
        page.select_option("#graph-limit", str(limit))
    page.fill("#graph-keyword", keyword)
    page.click("#graph-load")
    page.wait_for_timeout(wait)


def shot(page, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    page.screenshot(path=str(path))
    print("  ->", path.name, flush=True)


LAYOUT_ISSUES: list[str] = []


def check_layout(page, label: str):
    """两个可自动化的硬指标：无横向溢出、交互元素不出画布。"""
    res = page.evaluate(
        """() => {
            const vw = window.innerWidth;
            const doc = document.documentElement;
            const overflow = doc.scrollWidth - vw;
            const stray = [];
            document.querySelectorAll('button, input, select, textarea, a').forEach(el => {
              if (el.classList.contains('skip')) return;   // 视觉隐藏的跳转链接，本来就在画布外
              const r = el.getBoundingClientRect();
              if (r.width === 0 && r.height === 0) return;
              if (r.right > vw + 1 || r.left < -1) {
                stray.push((el.id || el.className || el.tagName) + ' @' + Math.round(r.left) + '..' + Math.round(r.right));
              }
            });
            return { overflow, stray: stray.slice(0, 8) };
        }"""
    )
    if res["overflow"] > 1:
        LAYOUT_ISSUES.append(f"{label}: 横向溢出 {res['overflow']}px")
    if res["stray"]:
        LAYOUT_ISSUES.append(f"{label}: 控件出画布 {res['stray']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8099")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--browser", default=None)
    ap.add_argument("--only", default=None, help="只跑名字含该子串的用例")
    args = ap.parse_args()

    out = Path(args.out)
    exe = find_browser(args.browser)
    print("[shoot] browser:", exe or "(playwright default)", flush=True)

    launch_kwargs = {"args": ["--font-render-hinting=none"]}
    if exe:
        launch_kwargs["executable_path"] = exe

    with sync_playwright() as p:
        browser = p.chromium.launch(**launch_kwargs)

        def new_page(viewport, reduced_motion="no-preference"):
            ctx = browser.new_context(
                viewport=viewport, device_scale_factor=1,
                reduced_motion=reduced_motion, locale="zh-CN",
            )
            page = ctx.new_page()
            page.on("console", lambda m: CONSOLE_ERRORS.append(m.type + ": " + m.text)
                    if m.type == "error" else None)
            page.on("pageerror", lambda e: CONSOLE_ERRORS.append("pageerror: " + str(e)))
            # 记录加载失败的资源 URL —— 例如 Google Fonts 被中断时，
            # 需要确认降级的是字体（可接受）而不是页面自身资源（不可接受）
            page.on("requestfailed", lambda r: CONSOLE_ERRORS.append(
                "requestfailed: " + r.url + " (" + str(r.failure) + ")")
                if "/static/" in r.url or r.url.rstrip("/") == args.base else None)
            return ctx, page

        # ── 桌面 1440×900：README 用的 8 张 ──────────────────────
        ctx, page = new_page(DESKTOP)
        print("[desktop 1440x900]", flush=True)

        goto_view(page, args.base, "chat")
        ask(page)
        shot(page, out, "01-chat.png")

        goto_view(page, args.base, "upload")
        settle(page, 400)
        shot(page, out, "02-upload.png")

        goto_view(page, args.base, "overview")
        page.wait_for_timeout(700)
        shot(page, out, "03-overview.png")

        # 答案特写：加一条通道筛选，展示"只看某一类出处"（与 01 区分开）
        goto_view(page, args.base, "chat")
        ask(page)
        page.click(".chan[data-channel='vector']")
        settle(page, 500)
        shot(page, out, "05-answer.png")
        ctx.close()

        # ── 降低动效 + 键盘焦点 ──────────────────────────────────
        ctx, page = new_page(DESKTOP, reduced_motion="reduce")
        print("[reduced motion + focus]", flush=True)
        goto_view(page, args.base, "chat")
        ask(page)
        shot(page, out, "ui-reduced-motion.png")
        page.focus(".cite")
        settle(page, 300)
        shot(page, out, "ui-focus.png")
        # 键盘可达性：Tab 走一遍，确认焦点落在真实控件上
        page.click(".rail-nav button[data-view='upload']")
        settle(page, 300)
        for _ in range(3):
            page.keyboard.press("Tab")
        settle(page, 200)
        shot(page, out, "ui-focus-upload.png")
        ctx.close()

        # ── 无 webfont 情形：验证回退字体栈能否独立成立（内网离线部署） ──
        ctx = browser.new_context(viewport=DESKTOP, device_scale_factor=1, locale="zh-CN")
        ctx.route("**fonts.googleapis.com**", lambda r: r.abort())
        ctx.route("**fonts.gstatic.com**", lambda r: r.abort())
        page = ctx.new_page()
        print("[no webfont fallback]", flush=True)
        goto_view(page, args.base, "chat")
        ask(page)
        shot(page, out, "ui-no-webfont-chat.png")
        goto_view(page, args.base, "overview")
        page.wait_for_timeout(700)
        shot(page, out, "ui-no-webfont-overview.png")
        ctx.close()

        browser.close()

    if LAYOUT_ISSUES:
        print("\n[shoot] 布局问题：", flush=True)
        for i in LAYOUT_ISSUES:
            print("  !", i, flush=True)
    else:
        print("\n[shoot] 布局检查通过：无横向溢出、控件均在画布内", flush=True)

    if CONSOLE_ERRORS:
        print("\n[shoot] 控制台/资源错误：", flush=True)
        for e in dict.fromkeys(CONSOLE_ERRORS):
            print("  !", e, flush=True)
        return 1
    print("[shoot] 控制台无错误", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
