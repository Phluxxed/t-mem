from __future__ import annotations

import json
import re

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
        except (KeyError, TypeError, IndexError):
            continue

    if not subtasks:
        return [_whole_session_subtask(turns, session_id)]

    return subtasks


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
