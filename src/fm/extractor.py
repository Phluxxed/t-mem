from __future__ import annotations

import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        except (KeyError, ValueError) as e:
            print(f"extractor: dropping malformed tip from LLM response: {e} — item={item!r:.120}", file=sys.stderr)
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


def _process_subtask(
    subtask: Subtask,
    *,
    session_id: str,
    project: str,
    model: str,
    counter: list[int],
    total: int,
    lock: threading.Lock,
) -> list[Tip]:
    """Run the full 3-stage pipeline for a single subtask. Safe to call concurrently."""
    desc = subtask.generalized_description[:60]
    with lock:
        counter[0] += 1
        n = counter[0]
    print(f"  [{n}/{total}] {desc}...", file=sys.stderr)

    intelligence = extract_intelligence(subtask, model=model)
    if intelligence is None:
        print(f"  [{n}/{total}] Warning: intelligence extraction failed for subtask {subtask.id}", file=sys.stderr)
        return []

    attribution = extract_attribution(subtask, intelligence, model=model)
    if attribution is None:
        print(f"  [{n}/{total}] Warning: attribution failed for subtask {subtask.id}", file=sys.stderr)
        return []

    tips = _extract_tips_from_subtask(
        subtask, intelligence, attribution,
        session_id=session_id, project=project, model=model,
    )
    print(f"  [{n}/{total}] done — {len(tips)} tip(s)", file=sys.stderr)
    return tips


def extract_tips_from_session(
    turns: list[Turn],
    *,
    session_id: str,
    project: str,
    model: str = "sonnet",
    max_workers: int = 4,
) -> list[Tip]:
    """Extract tips from a session using the full three-stage pipeline per subtask."""
    if not turns:
        return []

    print("  Segmenting session into subtasks...", file=sys.stderr)
    subtasks = segment_session(turns, session_id=session_id, model=model)
    print(f"  Found {len(subtasks)} subtask(s), processing (max {max_workers} concurrent)...", file=sys.stderr)

    counter: list[int] = [0]
    lock = threading.Lock()
    total = len(subtasks)

    all_tips: list[Tip] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_subtask, subtask,
                session_id=session_id, project=project, model=model,
                counter=counter, total=total, lock=lock,
            ): subtask
            for subtask in subtasks
        }
        for future in as_completed(futures):
            all_tips.extend(future.result())

    return all_tips
