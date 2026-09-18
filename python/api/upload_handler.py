"""
安全上传处理 —— 文件名净化 + 路径遏制 + 大小限制 + 扩展名白名单

背景
----
重构前 `api/main.py:108` 直接用客户端可控的 `file.filename` 拼路径：

    save_path = os.path.join(settings.upload_dir, file.filename or "unknown")

三个可利用点：
  1. 路径穿越：filename="../../../../x" 逃逸 uploads 目录
  2. 绝对路径覆盖：os.path.join("uploads", "/etc/passwd") == "/etc/passwd"
     （os.path.join 遇绝对路径会丢弃前缀）→ 任意文件写入
  3. 无大小限制：shutil.copyfileobj(file.file, f) 无 length → 磁盘耗尽

另外 `doc_parser_agent.SUPPORTED_EXTENSIONS` 看着像白名单，但 `_classify`
返回 UNKNOWN 后直接落进 `_parse_text`（doc_parser_agent.py:96-97），
**该字典从未用于拒绝任何文件**。本模块把它真正用作准入策略。
"""

from __future__ import annotations

import hashlib
import logging
import os
import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile, status

from agents.doc_parser_agent import DocParserAgent
from config import settings

logger = logging.getLogger(__name__)

# 扩展名 → 允许的安全后缀（由白名单推导，永不采用客户端给的文件名）
ALLOWED_EXTENSIONS: set[str] = set(DocParserAgent.SUPPORTED_EXTENSIONS.keys())

# 分块读取大小（流式，避免一次性读入内存）
_CHUNK = 1024 * 1024


def _safe_extension(filename: str) -> str:
    """
    从客户端文件名中只提取**扩展名**，并校验在白名单内。

    文件名本体一律丢弃 —— 服务端用 UUID 生成新名字。但这里仍显式拒绝
    含路径分隔符或 .. 的输入，作为纵深防御（不依赖下游的 realpath 断言
    单独兜底）。
    """
    name = filename or ""

    # 纵深防御：文件名里出现路径成分即拒绝，不等下游断言
    if any(sep in name for sep in ("/", "\\", "\x00")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="文件名不得包含路径分隔符",
        )

    ext = Path(name).suffix.lower()

    # 无扩展名或非白名单扩展名都拒绝（白名单是真正的准入策略，
    # 而非仅作类型映射 —— 原实现的 SUPPORTED_EXTENSIONS 从未用于拒绝）
    if not ext or ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"不支持的文件类型: {ext or '(无扩展名)'}；"
                f"允许的类型: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )
    return ext


def _assert_within(base: Path, target: Path) -> None:
    """
    路径遏制断言 —— 解析符号链接与 .. 之后必须仍在 upload_dir 内。

    这是防御路径穿越的最终闸门：即便前面的拼接逻辑被绕过，这里也会拦住。
    """
    base_real = base.resolve()
    target_real = target.resolve()
    if base_real != target_real and base_real not in target_real.parents:
        logger.error(
            "检测到路径穿越尝试",
            extra={"extra_fields": {"base": str(base_real), "target": str(target_real)}},
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="非法文件路径"
        )


async def save_upload_safely(file: UploadFile) -> tuple[str, str, str]:
    """
    安全落盘上传文件。

    返回 (保存路径, 原始文件名, 内容 sha256)

    安全措施：
      1. 扩展名白名单校验（真正拒绝，而非仅映射）
      2. 服务端 UUID 生成文件名 —— 客户端文件名永不参与路径构造
      3. realpath 遏制断言
      4. 流式写入 + 累计字节数上限，超限即删除并报错
    """
    original_name = file.filename or "unknown"
    ext = _safe_extension(original_name)

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    # 服务端生成文件名：UUID + 白名单扩展名
    safe_name = f"{uuid.uuid4().hex}{ext}"
    dest = upload_dir / safe_name

    # 二次确认目标在受管目录内
    _assert_within(upload_dir, dest)

    written = 0
    digest = hashlib.sha256()
    try:
        with open(dest, "wb") as f:
            while True:
                chunk = await file.read(_CHUNK)
                if not chunk:
                    break
                written += len(chunk)
                if written > settings.max_upload_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=(
                            f"文件超过大小上限 "
                            f"{settings.max_upload_bytes // (1024 * 1024)}MB"
                        ),
                    )
                digest.update(chunk)
                f.write(chunk)
    except HTTPException:
        # 超限：清掉半截文件，不留垃圾
        dest.unlink(missing_ok=True)
        raise
    except Exception:
        dest.unlink(missing_ok=True)
        logger.exception("写入上传文件失败")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="文件保存失败"
        )

    logger.info(
        "上传已落盘",
        extra={"extra_fields": {
            "saved_as": safe_name,
            "original_name": original_name,
            "bytes": written,
        }},
    )
    return str(dest), original_name, digest.hexdigest()


def resolve_managed_path(file_path: str) -> Path:
    """
    解析来自客户端的路径参数（/api/admin/update 的 file_path），
    确保它落在受管目录内。

    重构前该端点接受**任意路径**，可读取系统任意文件并入库。
    """
    upload_dir = Path(settings.upload_dir)
    candidate = Path(file_path)

    # 相对路径按 upload_dir 解析；绝对路径直接用（随后由遏制断言校验）
    target = candidate if candidate.is_absolute() else (upload_dir / candidate)
    _assert_within(upload_dir, target)

    if not target.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="目标文件不存在"
        )
    return target
