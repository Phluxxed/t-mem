from __future__ import annotations

from fm.models import Subtask, Turn


def _format_turn(turn: Turn, index: int) -> str:
    parts = [f"[Turn {index}] User: {turn.user_prompt[:400]}"]
    if turn.thinking:
        parts.append("  Thinking:")
        for t in turn.thinking[:3]:
            parts.append(f"    - {t[:300]}")
    if turn.actions:
        for a in turn.actions:
            parts.append(f"  Tool: {a.tool_name} (success={a.success})")
            if a.result_stderr:
                parts.append(f"    Error: {a.result_stderr[:200]}")
    if turn.response_text:
        parts.append(f"  Response: {turn.response_text[:400]}")
    return "\n".join(parts)


def build_intelligence_prompt(subtask: Subtask) -> str:
    formatted = "\n\n".join(_format_turn(t, i) for i, t in enumerate(subtask.turns))

    return f"""\
You are performing Trajectory Intelligence Extraction on a subtask from a Claude Code session.

## Subtask
{subtask.generalized_description}

## Task
Analyze the turns below and produce a structured intelligence report.

### 1. Reasoning Categories
Classify each piece of agent thinking/reasoning into one of four cognitive types:
- **analytical**: assessing the situation, reading context, understanding current state
- **planning**: deciding what actions to take, sequencing steps
- **validation**: checking assumptions, verifying results match expectations
- **reflection**: reconsidering approach, recognizing a mistake, changing strategy

### 2. Cognitive Patterns
Identify which of these behavioral patterns are present (list only those that apply):
- `validation_pattern`: agent checks results before proceeding to next step
- `reflection_pattern`: agent reconsiders and changes approach mid-task
- `self_correction`: agent recognizes and fixes its own mistake
- `error_recognition`: agent detects an error (tool failure, unexpected output, wrong assumption)
- `api_discovery`: agent discovers how an API or tool works through trial and exploration
- `efficiency_awareness`: agent recognizes a more efficient approach exists

### 3. Outcome
Classify the subtask outcome as exactly one of:
- `clean_success`: task completed directly without any failures or unnecessary steps
- `inefficient_success`: task completed but with unnecessary steps, repeated operations, or a slower path
- `recovery`: task initially failed but agent recognized the failure and successfully recovered
- `failure`: task did not complete successfully

## Output Format

Return ONLY a JSON object — no other text:
{{
  "reasoning_categories": {{
    "analytical": ["thought 1", ...],
    "planning": ["thought 1", ...],
    "validation": ["thought 1", ...],
    "reflection": ["thought 1", ...]
  }},
  "cognitive_patterns": ["pattern1", "pattern2"],
  "outcome": "recovery"
}}

## Turns

{formatted}
"""
