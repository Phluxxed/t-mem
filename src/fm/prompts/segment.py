from __future__ import annotations

from fm.models import Turn


def _format_turn(turn: Turn, index: int) -> str:
    parts = [f"[Turn {index}] User: {turn.user_prompt[:300]}"]
    if turn.actions:
        tools = [a.tool_name for a in turn.actions]
        successes = [a.success for a in turn.actions]
        parts.append(f"  Tools used: {', '.join(tools)}")
        parts.append(f"  Outcomes: {successes}")
    if turn.response_text:
        parts.append(f"  Response: {turn.response_text[:300]}")
    return "\n".join(parts)


def build_segmentation_prompt(turns: list[Turn]) -> str:
    formatted = "\n\n".join(_format_turn(t, i) for i, t in enumerate(turns))

    return f"""\
You are segmenting a Claude Code session into logical subtasks.

A subtask is a self-contained unit of work with a clear goal and outcome. Typical subtask types:
- Project/environment setup
- Implementing a specific feature
- Debugging or fixing a specific error
- Refactoring or reorganising code
- Running and fixing tests
- Configuring a tool or integration
- Answering a technical question

## Instructions

Given the session turns below, identify the logical subtask boundaries and produce a JSON array.

For each subtask, provide:
1. `subtask_id`: sequential string — "s1", "s2", etc.
2. `raw_description`: what actually happened in this session (may include specific file names, project names, errors)
3. `generalized_description`: abstract version that transfers to other contexts. Apply these transformations:
   - **Entity abstraction**: replace project names, file names, URLs, usernames, error IDs with generic placeholders
     - "future_memory" → "the target project"
     - "tips.db" → "the database file"
     - "voyage-4-lite" → "the embedding model"
   - **Action normalization**: use canonical verbs (configure, implement, debug, verify, refactor)
   - **Context removal**: strip task-specific qualifiers, keep the core operation
   - Examples:
     - "Fix SSL cert error for Voyage API calls in future_memory" → "Agent debugs SSL certificate verification failure in HTTP client requests"
     - "Add click CLI command fm extract-all" → "Agent implements a CLI command to batch-process items"
4. `turn_indices`: list of 0-based turn indices belonging to this subtask. Every turn must appear in exactly one subtask.

## Output Format

Return ONLY a JSON array — no other text:
[
  {{
    "subtask_id": "s1",
    "raw_description": "...",
    "generalized_description": "...",
    "turn_indices": [0, 1, 2]
  }}
]

## Session Turns

{formatted}
"""
