from __future__ import annotations

import json
import re
import sys

from fm.attribution import extract_attribution
from fm.intelligence import extract_intelligence
from fm.llm import call_claude
from fm.models import Subtask, SubtaskAttribution, SubtaskIntelligence, Tip, Turn
from fm.prompts.tips_gen import build_tips_generation_prompt
from fm.segmenter import segment_session


def _parse_tips_json(
    raw: str,
    *,
    session_id: str,
    project: str,
    subtask: Subtask,
) -> list[Tip]:
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
                subtask_id=subtask.id,
                subtask_description=subtask.generalized_description,
            )
            tips.append(tip)
        except (KeyError, ValueError):
            continue

    return tips


def _extract_tips_from_subtask(
    subtask: Subtask,
    intelligence: SubtaskIntelligence,
    attribution: SubtaskAttribution,
    *,
    session_id: str,
    project: str,
    model: str,
) -> list[Tip]:
    prompt = build_tips_generation_prompt(
        subtask, intelligence, attribution, project=project
    )
    raw = call_claude(prompt, model=model)
    if raw is None:
        return []
    return _parse_tips_json(raw, session_id=session_id, project=project, subtask=subtask)


def extract_tips_from_session(
    turns: list[Turn],
    *,
    session_id: str,
    project: str,
    model: str = "sonnet",
) -> list[Tip]:
    """Extract tips from a session using the full three-stage pipeline per subtask."""
    if not turns:
        return []

    subtasks = segment_session(turns, session_id=session_id, model=model)

    all_tips: list[Tip] = []
    for subtask in subtasks:
        intelligence = extract_intelligence(subtask, model=model)
        if intelligence is None:
            print(f"  Warning: intelligence extraction failed for subtask {subtask.id}", file=sys.stderr)
            continue

        attribution = extract_attribution(subtask, intelligence, model=model)
        if attribution is None:
            print(f"  Warning: attribution failed for subtask {subtask.id}", file=sys.stderr)
            continue

        tips = _extract_tips_from_subtask(
            subtask, intelligence, attribution,
            session_id=session_id, project=project, model=model,
        )
        all_tips.extend(tips)

    return all_tips
