"""
彻底重建 Chroma：删除整个数据目录，从零开始

背景：我反复 delete_collection + 重建了 5 次以上，怀疑底层存储
      残留了旧向量（ID 形式相同但内容不同），导致
      self-cosine(存储向量, 重新编码) = 0.0988 这种异常。
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
sys.stdout.reconfigure(encoding="utf-8")

from config import settings

path = Path(settings.chroma_path).resolve()
print(f"=== Chroma 数据目录: {path} ===")

if path.exists():
    # 统计
    files = list(path.rglob("*"))
    total_mb = sum(f.stat().st_size for f in files if f.is_file()) / 1024 / 1024
    print(f"  文件数: {len([f for f in files if f.is_file()])}")
    print(f"  总大小: {total_mb:.1f} MB")
    print(f"  子目录: {[d.name for d in path.iterdir() if d.is_dir()]}")

    # 备份后删除
    backup = path.parent / f"{path.name}_backup"
    if backup.exists():
        shutil.rmtree(backup)
    shutil.copytree(path, backup)
    print(f"  已备份到: {backup}")

    shutil.rmtree(path)
    print(f"  ✓ 已删除数据目录")
else:
    print("  目录不存在")

print()
print("=== 下一步 ===")
print("  运行 rebuild_all_index.py 从零重建")
