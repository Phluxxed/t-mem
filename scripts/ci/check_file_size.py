#!/usr/bin/env python3
"""
File size CI check. Fails if any source file exceeds the line limit.
"""
import sys
from pathlib import Path


SOURCE_EXTENSIONS = {".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs"}
LINE_LIMIT = 400
EXEMPT_DIRS = {"node_modules", ".git", "__pycache__", ".venv", "venv", "dist", "build"}


def check_file_sizes(root: Path) -> list[str]:
    errors = []
    for f in root.rglob("*"):
        if not f.is_file():
            continue
        rel = f.relative_to(root)
        if any(part in EXEMPT_DIRS for part in rel.parts):
            continue
        if f.suffix not in SOURCE_EXTENSIONS:
            continue
        lines = len(f.read_text(encoding="utf-8").splitlines())
        if lines > LINE_LIMIT:
            errors.append(f"{rel}: {lines} lines (limit {LINE_LIMIT})")
    return errors


def main() -> None:
    root = Path.cwd()
    errors = check_file_sizes(root)
    if errors:
        print("File size check FAILED:")
        for e in errors:
            print(f"  \u2717 {e}")
        sys.exit(1)
    print("File size check PASSED")


if __name__ == "__main__":
    main()
