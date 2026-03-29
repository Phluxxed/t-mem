#!/usr/bin/env python3
"""
Doc-gardening CI check. Validates the knowledge base is internally consistent.
Exits non-zero if any check fails.
"""
import re
import sys
from pathlib import Path


def check_links(root: Path) -> list[str]:
    """Validate cross-links in CLAUDE.md and ARCHITECTURE.md point to real files."""
    errors = []
    for md_file in [root / "CLAUDE.md", root / "ARCHITECTURE.md"]:
        if not md_file.exists():
            continue
        content = md_file.read_text(encoding="utf-8")
        for match in re.finditer(r'\[([^\]]+)\]\(([^)]+)\)', content):
            text, href = match.group(1), match.group(2)
            if href.startswith("http") or href.startswith("#"):
                continue
            if href.startswith("/"):
                continue  # absolute paths are not local relative links
            target = (root / href).resolve()
            if not target.exists():
                errors.append(f"{md_file.name}: broken link [{text}]({href})")
    return errors


def check_exec_plan_naming(root: Path) -> list[str]:
    """Validate exec-plans/active/ files follow YYYY-MM-DD-<name>.md convention."""
    errors = []
    active_dir = root / "docs" / "exec-plans" / "active"
    if not active_dir.exists():
        return errors
    pattern = re.compile(r"^\d{4}-\d{2}-\d{2}-.+\.md$")
    for f in active_dir.iterdir():
        if f.name == ".gitkeep" or not f.is_file():
            continue
        if not pattern.match(f.name):
            errors.append(
                f"{f.relative_to(root)}: "
                "does not match YYYY-MM-DD-<name>.md"
            )
    return errors


def check_orphaned_docs(root: Path) -> list[str]:
    """Validate all docs/ markdown files are reachable from CLAUDE.md."""
    errors = []
    claude_md = root / "CLAUDE.md"
    if not claude_md.exists():
        return errors
    content = claude_md.read_text(encoding="utf-8")
    referenced = set()
    for match in re.finditer(r'\[([^\]]+)\]\(([^)]+)\)', content):
        href = match.group(2)
        if href.startswith("http") or href.startswith("#"):
            continue
        # Strip fragment
        href = href.split("#")[0]
        # Normalise ./docs/... to docs/...
        if href.startswith("./"):
            href = href[2:]
        if href.endswith(".md"):
            referenced.add(href)
    exempt_prefixes = ("docs/exec-plans/", "docs/plans/", "docs/references/")
    for f in (root / "docs").rglob("*.md"):
        rel = str(f.relative_to(root))
        if any(rel.startswith(p) for p in exempt_prefixes):
            continue
        if rel not in referenced:
            errors.append(f"{rel}: not reachable from CLAUDE.md")
    return errors


def run_checks(root: Path) -> list[str]:
    errors = []
    errors.extend(check_links(root))
    errors.extend(check_exec_plan_naming(root))
    errors.extend(check_orphaned_docs(root))
    return errors


def main() -> None:
    root = Path.cwd()
    errors = run_checks(root)
    if errors:
        print("Doc-gardening check FAILED:")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    print("Doc-gardening check PASSED")


if __name__ == "__main__":
    main()
