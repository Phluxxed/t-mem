from __future__ import annotations

import json
import re
import subprocess
import sys

from fm.models import Tip, Turn
from fm.prompts.extract import build_extraction_prompt


def _parse_tips_json(raw: str, session_id: str, project: str) -> list[Tip]:
    """Parse the LLM's JSON response into Tip objects."""
    json_match = re.search(r"\{[\s\S]*\}", raw)
    if not json_match:
        return []

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return []

    tips_data = data.get("tips", [])
    if not isinstance(tips_data, list):
        return []

    tips = []
    for item in tips_data:
        try:
            tip = Tip(
                category=item["category"],
                content=item["content"],
                purpose=item.get("purpose", ""),
                steps=item.get("steps", []),
                trigger=item.get("trigger", ""),
                negative_example=item.get("negative_example"),
                priority=item.get("priority", "medium"),
                source_session_id=session_id,
                source_project=project,
                task_context=item.get("task_context"),
            )
            tips.append(tip)
        except (KeyError, ValueError):
            continue

    return tips


def extract_tips_from_session(
    turns: list[Turn],
    *,
    session_id: str,
    project: str,
    model: str = "sonnet",
) -> list[Tip]:
    """Extract tips from parsed session turns using the claude CLI."""
    if not turns:
        return []

    prompt = build_extraction_prompt(turns, session_id=session_id, project=project)

    try:
        result = subprocess.run(
            [
                "claude",
                "-p", prompt,
                "--model", model,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []

    if result.returncode != 0:
        print(
            f"Warning: claude CLI returned {result.returncode}: {result.stderr}",
            file=sys.stderr,
        )
        return []

    return _parse_tips_json(result.stdout, session_id, project)
