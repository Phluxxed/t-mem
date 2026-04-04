from __future__ import annotations

from fm.models import Subtask, SubtaskAttribution, SubtaskIntelligence

_TIP_SCHEMA = """\
{
  "tips": [
    {
      "title": "5-8 word imperative phrase summarising the tip",
      "category": "strategy" | "recovery" | "optimization",
      "content": "Actionable guidance in imperative form",
      "purpose": "Why this tip exists — what problem it prevents",
      "steps": ["Step 1", "Step 2"],
      "trigger": "Specific condition under which this applies",
      "negative_example": "What NOT to do — null if not applicable",
      "priority": "critical" | "high" | "medium" | "low",
      "task_context": "Domain context — null if generic"
    }
  ]
}"""


def build_tips_generation_prompt(
    subtask: Subtask,
    intelligence: SubtaskIntelligence,
    attribution: SubtaskAttribution,
    *,
    project: str,
) -> str:
    patterns_str = ", ".join(intelligence.cognitive_patterns) if intelligence.cognitive_patterns else "none"
    root_causes_str = "\n".join(f"  - {c}" for c in attribution.root_causes) if attribution.root_causes else "  (none)"
    causal_chain_str = "\n".join(f"  {s}" for s in attribution.causal_chain) if attribution.causal_chain else "  (none)"

    tip_count_guidance = {
        "clean_success": "1-2 strategy tips capturing what worked. If execution was unremarkable, return [].",
        "inefficient_success": "1-2 optimization tips identifying the more efficient approach.",
        "recovery": "1-3 recovery tips focused on the root cause and how to avoid or recover from it.",
        "failure": "1-2 recovery tips on the failure mode, if a clear pattern exists.",
    }.get(intelligence.outcome, "2-4 tips as appropriate.")

    return f"""\
You are generating structured tips from a subtask analysis in project "{project}".

## Subtask (generalised)
{subtask.generalized_description}

## Outcome
{intelligence.outcome}

## Cognitive Patterns
{patterns_str}

## Root Causes
{root_causes_str}

## Causal Chain
{causal_chain_str}

## Task
Generate concrete, actionable tips from this subtask. Tip count guidance: {tip_count_guidance}

Requirements:
- **Actionable**: Stated in imperative form ("When X, do Y")
- **Generalized**: No project-specific names, file paths, or IDs — must transfer to other contexts
- **Non-obvious**: Skip tips for basic programming practices everyone already knows
- **Prioritized**: critical=prevents failure/data loss, high=significant improvement, medium=useful, low=minor

Priority guide for outcomes:
- recovery tips from failures → high or critical
- optimization tips → medium
- strategy tips from clean success → medium or high

## Output Format

Return ONLY a JSON object matching this schema — no other text:
{_TIP_SCHEMA}

If no genuine non-obvious tips exist, return {{"tips": []}}.
"""
