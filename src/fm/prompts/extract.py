from __future__ import annotations

import json
from fm.models import Turn

_MAX_TOOL_OUTPUT_CHARS = 2000

_TIP_SCHEMA = """\
{
  "tips": [
    {
      "category": "strategy" | "recovery" | "optimization",
      "content": "The actionable guidance — what to do",
      "purpose": "Why this tip exists — what problem it prevents or what benefit it provides",
      "steps": ["Concrete step 1", "Concrete step 2"],
      "trigger": "When this tip should be applied",
      "negative_example": "What NOT to do (optional, null if not applicable)",
      "priority": "critical" | "high" | "medium" | "low",
      "task_context": "Domain or application context (optional, null if generic)"
    }
  ]
}"""

_FEW_SHOT_EXAMPLES = """\
Example 1 — Strategy tip from a clean success:
{
  "category": "strategy",
  "content": "When performing operations that have prerequisites, systematically verify all prerequisites before beginning the main operation.",
  "purpose": "Prevents failures caused by missing prerequisites, reducing wasted steps and error recovery.",
  "steps": [
    "Identify all prerequisites for the target operation",
    "Verify each prerequisite is satisfied before proceeding",
    "Only begin the main operation once all checks pass"
  ],
  "trigger": "When task involves multi-step operations with dependencies",
  "negative_example": "Do not assume prerequisites are met without checking — even if they were met in a previous session.",
  "priority": "high",
  "task_context": null
}

Example 2 — Recovery tip from a failure-then-recovery:
{
  "category": "recovery",
  "content": "When a file edit fails because the old_string doesn't match, re-read the file to get the current content before retrying the edit.",
  "purpose": "File contents may have changed since last read. Editing with stale content causes failures.",
  "steps": [
    "Recognise the edit failure error message",
    "Re-read the target file to get current content",
    "Retry the edit with the correct old_string from the fresh read"
  ],
  "trigger": "When an Edit tool call fails with a 'string not found' error",
  "negative_example": "Do not retry the same edit with the same old_string — it will fail again.",
  "priority": "high",
  "task_context": null
}

Example 3 — Optimization tip from an inefficient success:
{
  "category": "optimization",
  "content": "When searching for a specific pattern across the codebase, use Grep with a targeted glob pattern instead of reading files one by one.",
  "purpose": "Reading files individually is slow and token-expensive. Grep searches all matching files in one call.",
  "steps": [
    "Identify the search pattern needed",
    "Use Grep with an appropriate glob filter (e.g. '*.py')",
    "Process the results instead of manually reading each candidate file"
  ],
  "trigger": "When task requires finding code patterns across multiple files",
  "negative_example": "Do not open files one at a time to search for a pattern — use Grep.",
  "priority": "medium",
  "task_context": null
}"""


def _format_turn(turn: Turn, index: int) -> str:
    """Format a single turn for inclusion in the extraction prompt."""
    parts = [f"### Turn {index + 1}"]
    parts.append(f"**User:** {turn.user_prompt}")

    if turn.thinking:
        parts.append("**Thinking:**")
        for thought in turn.thinking:
            truncated = thought[:_MAX_TOOL_OUTPUT_CHARS]
            if len(thought) > _MAX_TOOL_OUTPUT_CHARS:
                truncated += " [truncated]"
            parts.append(f"  - {truncated}")

    if turn.actions:
        parts.append("**Actions:**")
        for action in turn.actions:
            input_str = json.dumps(action.tool_input, indent=None)
            if len(input_str) > _MAX_TOOL_OUTPUT_CHARS:
                input_str = input_str[:_MAX_TOOL_OUTPUT_CHARS] + " [truncated]"
            parts.append(f"  - Tool: {action.tool_name}")
            parts.append(f"    Input: {input_str}")
            if action.result_stdout:
                stdout = action.result_stdout[:_MAX_TOOL_OUTPUT_CHARS]
                if len(action.result_stdout) > _MAX_TOOL_OUTPUT_CHARS:
                    stdout += " [truncated]"
                parts.append(f"    Output: {stdout}")
            if action.result_stderr:
                stderr = action.result_stderr[:_MAX_TOOL_OUTPUT_CHARS]
                if len(action.result_stderr) > _MAX_TOOL_OUTPUT_CHARS:
                    stderr += " [truncated]"
                parts.append(f"    Error: {stderr}")
            parts.append(f"    Success: {action.success}")

    if turn.response_text:
        response = turn.response_text[:_MAX_TOOL_OUTPUT_CHARS]
        if len(turn.response_text) > _MAX_TOOL_OUTPUT_CHARS:
            response += " [truncated]"
        parts.append(f"**Response:** {response}")

    return "\n".join(parts)


def build_extraction_prompt(
    turns: list[Turn],
    *,
    session_id: str,
    project: str,
) -> str:
    """Build the tip extraction prompt from parsed turns."""
    formatted_turns = "\n\n".join(
        _format_turn(turn, i) for i, turn in enumerate(turns)
    )

    return f"""\
You are analysing an agent execution trajectory to extract actionable learning tips.

The trajectory below is from a Claude Code session in project "{project}" (session: {session_id}).
It contains the user's requests, the agent's reasoning (thinking), the actions taken (tool calls and results), and the agent's responses.

## Your Task

Analyse this trajectory and extract structured tips in three categories:

1. **Strategy tips**: Effective patterns from successful execution — what worked well and should be replicated.
2. **Recovery tips**: Patterns from failure-then-recovery sequences — what went wrong, how it was recognised, and how it was fixed.
3. **Optimization tips**: Efficiency improvements from successful but suboptimal execution — what was done inefficiently and what the better alternative is.

For each tip, trace back to the specific decision or reasoning that led to the outcome. Tips must be:
- **Actionable**: The agent can directly apply them
- **Specific**: Concrete actions, not vague advice
- **Generalised**: Remove entity-specific details (file paths, variable names) so tips transfer to similar situations

## Output Format

Return a JSON object matching this schema:
{_TIP_SCHEMA}

## Examples

{_FEW_SHOT_EXAMPLES}

## Trajectory

{formatted_turns}

## Instructions

- Extract 2-6 tips from this trajectory. Quality over quantity — skip if there's nothing worth learning.
- If the session was entirely straightforward with no notable patterns, return {{"tips": []}}.
- Focus on patterns that would help in FUTURE sessions, not just this one.
- Return ONLY the JSON object, no other text."""
