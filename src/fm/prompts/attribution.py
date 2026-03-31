from __future__ import annotations

from fm.models import Subtask, SubtaskIntelligence


def build_attribution_prompt(subtask: Subtask, intelligence: SubtaskIntelligence) -> str:
    turns_summary = "\n".join(
        f"[Turn {i}] {t.user_prompt[:200]}"
        + (f" → tools: {[a.tool_name for a in t.actions]}" if t.actions else "")
        + (f" → errors: {[a.result_stderr[:100] for a in t.actions if a.result_stderr]}" if any(a.result_stderr for a in t.actions) else "")
        for i, t in enumerate(subtask.turns)
    )

    patterns_str = ", ".join(intelligence.cognitive_patterns) if intelligence.cognitive_patterns else "none"
    reflection_thoughts = intelligence.reasoning_categories.get("reflection", [])
    reflection_str = "\n".join(f"  - {t}" for t in reflection_thoughts[:5]) if reflection_thoughts else "  (none)"

    return f"""\
You are performing Decision Attribution Analysis on a subtask from a Claude Code session.

## Subtask
{subtask.generalized_description}

## Outcome
{intelligence.outcome}

## Cognitive Patterns Observed
{patterns_str}

## Agent Reflection Thoughts
{reflection_str}

## Turn Summary
{turns_summary}

## Task
Trace backward through the agent's decisions to identify the causal chain that led to this outcome.

For failures and recoveries: identify what went wrong and why.
For clean successes: identify what decisions were critical to the success.
For inefficient successes: identify what choices caused unnecessary work.

Return ONLY a JSON object — no other text:
{{
  "root_causes": ["the fundamental decision or assumption that drove the outcome"],
  "contributing_factors": ["conditions that amplified or enabled the root cause"],
  "causal_chain": ["1: first decision/event", "2: consequence", "3: final outcome"]
}}
"""
