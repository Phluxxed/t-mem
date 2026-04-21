from __future__ import annotations

import json
import re
import sys

from fm.llm import call_claude
from fm.models import Subtask, Turn
from fm.prompts.segment import build_segmentation_prompt


def _parse_segmentation(raw: str, turns: list[Turn], *, session_id: str) -> list[Subtask]:
    """Parse the LLM segmentation response into Subtask objects. Falls back to single subtask on failure."""
    json_match = re.search(r"\[[\s\S]*\]", raw)
    if not json_match:
        return [_whole_session_subtask(turns, session_id)]

    try:
        items = json.loads(json_match.group())
    except json.JSONDecodeError:
        return [_whole_session_subtask(turns, session_id)]

    if not isinstance(items, list) or not items:
        return [_whole_session_subtask(turns, session_id)]

    subtasks = []
    for item in items:
        try:
            indices = item["turn_indices"]
            subtask_turns = [turns[i] for i in indices if 0 <= i < len(turns)]
            subtasks.append(
                Subtask(
                    id=item["subtask_id"],
                    session_id=session_id,
                    raw_description=item["raw_description"],
                    generalized_description=item["generalized_description"],
                    turns=subtask_turns,
                )
            )
        except (KeyError, TypeError, IndexError) as e:
            print(f"segmenter: skipping malformed subtask item {item!r:.80}: {e}", file=sys.stderr)
            continue

    if not subtasks:
        return [_whole_session_subtask(turns, session_id)]

    return subtasks


def summarize_session(turns: list[Turn], *, model: str = "haiku") -> str:
    """Produce a 1-2 sentence synthesized summary of what a session accomplished.

    Used as the generalized_description for task-level tip extraction.
    """
    snippets = "\n".join(
        f"[Turn {i}] {t.user_prompt[:150]}"
        + (f" → tools: {[a.tool_name for a in t.actions[:4]]}" if t.actions else "")
        for i, t in enumerate(turns[:30])
    )
    prompt = f"""\
Summarize what this Claude Code session accomplished in 1-2 sentences.
Be specific about what was built, fixed, debugged, or investigated.
Use active voice. Focus on outcomes, not individual steps.
Do not mention turn numbers, tool names, or file paths.

Generalize all entity names so the summary transfers to other contexts:
- Project names → "the project" or "the target system"
- Company or product names → omit or use "the application"
- Specific file names → "the configuration file", "the database", etc.
- Specific error messages or IDs → describe the error type generically

## Session (truncated to first 30 turns)
{snippets}

Return only the summary — no preamble, no extra text."""

    result = call_claude(prompt, model=model)
    if result is None:
        return turns[0].user_prompt[:200] if turns else "Agent session"
    return result.strip()


def _whole_session_subtask(turns: list[Turn], session_id: str) -> Subtask:
    """Fallback: treat the entire session as a single subtask."""
    return Subtask(
        id="s1",
        session_id=session_id,
        raw_description="Full session (segmentation failed or skipped)",
        generalized_description="Agent completes a multi-step task",
        turns=turns,
    )


def segment_session(
    turns: list[Turn],
    *,
    session_id: str,
    model: str = "sonnet",
) -> list[Subtask]:
    """Segment a session's turns into logical subtasks with generalised descriptions."""
    if not turns:
        return []

    prompt = build_segmentation_prompt(turns)
    raw = call_claude(prompt, model=model)

    if raw is None:
        return [_whole_session_subtask(turns, session_id)]

    return _parse_segmentation(raw, turns, session_id=session_id)
