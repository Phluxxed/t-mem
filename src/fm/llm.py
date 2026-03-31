from __future__ import annotations

import subprocess
import sys


def call_claude(prompt: str, *, model: str = "sonnet", timeout: int = 120) -> str | None:
    """Call the claude CLI with a prompt, return stdout or None on failure."""
    try:
        result = subprocess.run(
            ["claude", "--print", "--model", model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None

    if result.returncode != 0:
        print(f"Warning: claude CLI returned {result.returncode}: {result.stderr[:200]}", file=sys.stderr)
        return None

    if not result.stdout.strip():
        print("Warning: claude CLI returned empty output", file=sys.stderr)
        return None

    return result.stdout
