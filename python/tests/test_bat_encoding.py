"""
Windows 批处理脚本（.bat / .cmd）的编码与格式守卫

为什么需要这个测试
------------------
这套规则是踩出来的，不是洁癖：

1. **UTF-8 BOM 会让首行失效**
   cmd.exe 把 BOM 字节当成命令名的一部分，`@echo off` 变成
   `\ufeff@echo off`，于是报 `'锘緻echo' 不是内部或外部命令`
   （`EF BB BF` 在 GBK 下正好显示为「锘緻」）。
   成因：Windows PowerShell 5.1 的 `Set-Content -Encoding UTF8` **默认写 BOM**。

2. **.bat 里不要写中文注释**
   cmd.exe 默认代码页是 936(GBK)，看不懂 UTF-8 中文；误码后的乱码里可能
   混入 `<` `>` `&` `|` 这类 cmd 保留字符，把一行**截断成多条命令**，
   报出 `'geHub' 不是内部或外部命令` 这种莫名其妙的碎片。
   要中文提示就在 Python 侧打印，或给 .bat 显式 `chcp 65001` 后再用。

3. **行尾应为 CRLF**
   LF 目前能跑，但 .bat 在 Windows 上的惯例是 CRLF，混用是隐患。

这个测试会扫描仓库里所有 .bat/.cmd，任何一条不满足就失败。
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

BAT_FILES = sorted(
    p for p in REPO_ROOT.rglob("*")
    if p.suffix.lower() in (".bat", ".cmd")
    and ".git" not in p.parts
)


def test_bat_files_are_found():
    """守卫本身要有效：必须真的扫到脚本，否则这个测试是空转"""
    assert BAT_FILES, f"未在 {REPO_ROOT} 下找到任何 .bat/.cmd —— 守卫失效"


@pytest.mark.parametrize("path", BAT_FILES, ids=lambda p: p.name)
def test_no_utf8_bom(path: pathlib.Path):
    """BOM 会让首行命令失效（'锘緻echo' 不是内部或外部命令）"""
    raw = path.read_bytes()
    assert raw[:3] != b"\xef\xbb\xbf", (
        f"{path.name} 带 UTF-8 BOM —— cmd.exe 会把 BOM 当成首行命令的一部分。"
        f"写入时不要用 PowerShell 的 Set-Content -Encoding UTF8（它会加 BOM）。"
    )


@pytest.mark.parametrize("path", BAT_FILES, ids=lambda p: p.name)
def test_ascii_only(path: pathlib.Path):
    """非 ASCII（中文）在 GBK 代码页下会被误读，可能把一行截断成多条命令"""
    raw = path.read_bytes()
    bad = [
        (i, ln) for i, ln in enumerate(raw.decode("utf-8", "replace").splitlines(), 1)
        if any(ord(c) > 127 for c in ln)
    ]
    assert not bad, (
        f"{path.name} 含非 ASCII 字符（第 {[i for i, _ in bad]} 行）：\n"
        + "\n".join(f"    L{i}: {ln!r}" for i, ln in bad[:5])
        + "\n  cmd.exe 默认代码页是 GBK，会把它们误读；误码后若混入 < > & | "
          "会把一行截断成多条命令。请改成英文，或给脚本加 chcp 65001。"
    )


@pytest.mark.parametrize("path", BAT_FILES, ids=lambda p: p.name)
def test_crlf_line_endings(path: pathlib.Path):
    """Windows 批处理应为 CRLF 行尾"""
    raw = path.read_bytes()
    lf = raw.count(b"\n")
    crlf = raw.count(b"\r\n")
    assert lf > 0, f"{path.name} 没有换行符？"
    assert crlf == lf, (
        f"{path.name} 行尾不是全 CRLF（CRLF={crlf}, LF={lf}）—— "
        f"其中有 {lf - crlf} 行是裸 LF"
    )


@pytest.mark.parametrize("path", BAT_FILES, ids=lambda p: p.name)
def test_is_valid_utf8(path: pathlib.Path):
    """无 BOM 的 UTF-8 仍应是合法编码（纯 ASCII 自然满足）"""
    raw = path.read_bytes()
    raw.decode("utf-8")  # 抛异常即失败
