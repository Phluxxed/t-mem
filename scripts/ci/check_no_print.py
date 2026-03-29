#!/usr/bin/env python3
"""
No-debug-print CI check. Fails if debug print statements are found in source files.
"""
import sys
from pathlib import Path


PRINT_PATTERNS: dict[str, tuple[str, list[str]]] = {
    "python": ("print(", [".py"]),
    "node": ("console.log(", [".ts", ".tsx", ".js", ".jsx"]),
    "go": ("fmt.Print", [".go"]),
    "rust": ("println!", [".rs"]),
}

STACK_MARKERS: dict[str, list[str]] = {
    "python": ["pyproject.toml", "requirements.txt", "setup.py"],
    "node": ["package.json"],
    "go": ["go.mod"],
    "rust": ["Cargo.toml"],
}

EXEMPT_DIRS = {"node_modules", ".git", "__pycache__", ".venv", "venv", "dist", "build"}
EXEMPT_PATTERNS = {"test_", "_test.", ".test.", ".spec."}


def detect_stack(root: Path) -> str | None:
    for stack, markers in STACK_MARKERS.items():
        if any((root / m).exists() for m in markers):
            return stack
    return None


def is_test_file(path: Path) -> bool:
    return any(p in path.name for p in EXEMPT_PATTERNS)


def check_no_print(root: Path) -> list[str]:
    stack = detect_stack(root)
    if stack not in PRINT_PATTERNS:
        return []
    pattern, extensions = PRINT_PATTERNS[stack]
    errors = []
    for f in root.rglob("*"):
        if not f.is_file():
            continue
        rel = f.relative_to(root)
        if any(part in EXEMPT_DIRS for part in rel.parts):
            continue
        if f.suffix not in extensions:
            continue
        if is_test_file(f):
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if pattern in line:
                errors.append(f"{rel}:{i}: {pattern} found")
    return errors


def main() -> None:
    root = Path.cwd()
    errors = check_no_print(root)
    if errors:
        print("No-print check FAILED:")
        for e in errors:
            print(f"  \u2717 {e}")
        sys.exit(1)
    print("No-print check PASSED")


if __name__ == "__main__":
    main()
