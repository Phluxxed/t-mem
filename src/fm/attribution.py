from __future__ import annotations

import json
import re

from fm.llm import call_claude
from fm.models import Subtask, SubtaskAttribution, SubtaskIntelligence
from fm.prompts.attribution import build_attribution_prompt


def _parse_attribution(raw: str) -> SubtaskAttribution | None:
    json_match = re.search(r"\{[\s\S]*\}", raw)
    if not json_match:
        return None
    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return None

    try:
        return SubtaskAttribution(
            root_causes=data.get("root_causes", []),
            contributing_factors=data.get("contributing_factors", []),
            causal_chain=data.get("causal_chain", []),
        )
    except (KeyError, TypeError):
        return None


def extract_attribution(
    subtask: Subtask,
    intelligence: SubtaskIntelligence,
    *,
    model: str = "sonnet",
) -> SubtaskAttribution | None:
    """Run the Decision Attribution Analyzer on a subtask."""
    prompt = build_attribution_prompt(subtask, intelligence)
    raw = call_claude(prompt, model=model)
    if raw is None:
        return None
    return _parse_attribution(raw)
