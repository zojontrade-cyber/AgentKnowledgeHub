"""导出语料全文，供出题时核对真实内容（避免编造答案）。"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(r"C:\Users\滨滨\Desktop\Knowledge-Base-Intelligent-Q-A-main\data\clean")

for sub in ("institutional", "products"):
    print("#" * 100)
    print(f"# {sub}")
    print("#" * 100)
    for f in sorted((BASE / sub).glob("*.md")):
        print()
        print("=" * 100)
        print(f"FILE: {f.name}")
        print("=" * 100)
        print(f.read_text(encoding="utf-8", errors="ignore"))
