from __future__ import annotations

import json
import re

from fm.llm import call_claude
from fm.models import Subtask, SubtaskIntelligence
from fm.prompts.intelligence import build_intelligence_prompt


def _parse_intelligence(raw: str) -> SubtaskIntelligence | None:
    json_match = re.search(r"\{[\s\S]*\}", raw)
    if not json_match:
        return None
    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return None

    try:
        return SubtaskIntelligence(
            reasoning_categories=data.get("reasoning_categories", {
                "analytical": [], "planning": [], "validation": [], "reflection": []
            }),
            cognitive_patterns=data.get("cognitive_patterns", []),
            outcome=data.get("outcome", "clean_success"),
        )
    except (KeyError, TypeError):
        return None


def extract_intelligence(
    subtask: Subtask,
    *,
    model: str = "sonnet",
) -> SubtaskIntelligence | None:
    """Run the Trajectory Intelligence Extractor on a subtask."""
    prompt = build_intelligence_prompt(subtask)
    raw = call_claude(prompt, model=model)
    if raw is None:
        return None
    return _parse_intelligence(raw)
