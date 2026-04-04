from __future__ import annotations

from fm.models import Tip


def _format_tip(tip: Tip) -> str:
    steps = "\n".join(f"    {i+1}. {s}" for i, s in enumerate(tip.steps)) if tip.steps else "    (none)"
    neg = f"\n  negative_example: {tip.negative_example}" if tip.negative_example else ""
    return (
        f"  id: {tip.id[:8]}\n"
        f"  category: {tip.category}\n"
        f"  priority: {tip.priority}\n"
        f"  content: {tip.content}\n"
        f"  purpose: {tip.purpose}\n"
        f"  trigger: {tip.trigger}\n"
        f"  steps:\n{steps}"
        f"{neg}"
    )


def build_consolidation_prompt(tips: list[Tip]) -> str:
    tips_block = "\n\n".join(f"--- Tip {i+1} ---\n{_format_tip(t)}" for i, t in enumerate(tips))

    return f"""\
You are reviewing a set of tips extracted from AI agent execution trajectories to determine \
whether they should be merged into a single canonical tip.

TIPS TO REVIEW:

{tips_block}

TASK:
Decide whether these tips are genuinely saying the same thing or are meaningfully distinct.

Rules:
- Be CONSERVATIVE. If in doubt, keep separate. Only merge if the tips are giving the same \
actionable advice in the same situation.
- Tips can have similar vocabulary but apply in different contexts — keep them separate.
- Tips about the same general topic but with different triggers or steps are distinct — keep them separate.
- Only merge if a single canonical tip would genuinely replace all of them without losing signal.

If merging: synthesise a canonical tip that is MORE GENERAL than any individual tip, \
preserves all important steps/triggers/nuance from the originals, and uses the HIGHEST \
priority among the merged tips.

Respond with JSON only, no other text:

If keeping separate:
{{"action": "keep", "reasoning": "one sentence explaining why these are distinct"}}

If merging:
{{
  "action": "merge",
  "reasoning": "one sentence explaining why these are genuinely the same advice",
  "category": "strategy" | "recovery" | "optimization",
  "priority": "critical" | "high" | "medium" | "low",
  "content": "Synthesised actionable guidance in imperative form",
  "purpose": "Why this tip exists — what problem it prevents",
  "trigger": "Specific condition under which this applies",
  "steps": ["Step 1", "Step 2"],
  "negative_example": "What NOT to do, or null if not applicable"
}}"""
