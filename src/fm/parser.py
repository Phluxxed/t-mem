from __future__ import annotations

import json
import re
from pathlib import Path
from typing import overload

from fm.models import Action, Turn


def _strip_noise(text: str) -> str:
    """Remove ANSI escape codes and system-reminder tags."""
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    text = re.sub(r"<system-reminder>.*?</system-reminder>", "", text, flags=re.DOTALL)
    return text.strip()


def _extract_user_prompt(content: str | list) -> str | None:
    """Extract the user's prompt text from message content.

    Returns None if the content is a tool result or otherwise not a user prompt.
    """
    if isinstance(content, str):
        return _strip_noise(content)
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                return None
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block["text"])
        return _strip_noise(" ".join(texts)) if texts else None
    return None


def _build_tree(entries: list[dict]) -> dict[str | None, list[dict]]:
    """Build a parent -> children mapping from JSONL entries."""
    tree: dict[str | None, list[dict]] = {}
    for entry in entries:
        parent = entry.get("parentUuid")
        tree.setdefault(parent, []).append(entry)
    return tree


def _walk_tree(
    tree: dict[str | None, list[dict]],
    node_uuid: str | None,
) -> list[dict]:
    """Walk the tree depth-first from a given node, returning entries in order."""
    result = []
    for child in tree.get(node_uuid, []):
        result.append(child)
        child_uuid = child.get("uuid")
        if child_uuid:
            result.extend(_walk_tree(tree, child_uuid))
    return result


def _is_user_prompt(entry: dict) -> bool:
    """Check if an entry is an actual user prompt (not meta, not tool result)."""
    if entry.get("type") != "user":
        return False
    if entry.get("isMeta"):
        return False
    if "toolUseResult" in entry:
        return False
    content = entry.get("message", {}).get("content")
    return _extract_user_prompt(content) is not None


@overload
def parse_session(jsonl_path: Path, *, return_session_id: bool = False) -> list[Turn]: ...
@overload
def parse_session(jsonl_path: Path, *, return_session_id: bool = True) -> tuple[str, list[Turn]]: ...


def parse_session(
    jsonl_path: Path,
    *,
    return_session_id: bool = False,
) -> list[Turn] | tuple[str, list[Turn]]:
    """Parse a Claude Code JSONL session file into a list of Turns."""
    entries = []
    session_id = ""

    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            entries.append(entry)
            if not session_id and entry.get("sessionId"):
                session_id = entry["sessionId"]

    skip_types = {"file-history-snapshot"}
    skip_subtypes = {"stop_hook_summary", "turn_duration", "compact_boundary", "local_command"}

    filtered = []
    for entry in entries:
        if entry.get("type") in skip_types:
            continue
        if entry.get("type") == "system" and entry.get("subtype") in skip_subtypes:
            continue
        filtered.append(entry)

    tree = _build_tree(filtered)
    ordered = _walk_tree(tree, None)

    turns: list[Turn] = []
    current_turn: Turn | None = None
    pending_actions: dict[str, Action] = {}

    for entry in ordered:
        entry_type = entry.get("type")

        if entry_type == "user" and _is_user_prompt(entry):
            content = entry.get("message", {}).get("content")
            prompt = _extract_user_prompt(content) or ""
            current_turn = Turn(
                user_prompt=prompt,
                timestamp=entry.get("timestamp", ""),
                cwd=entry.get("cwd", ""),
            )
            turns.append(current_turn)

        elif entry_type == "assistant" and current_turn is not None:
            content = entry.get("message", {}).get("content", [])
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "thinking":
                        thinking_text = block.get("thinking", "")
                        if thinking_text:
                            current_turn.thinking.append(thinking_text)
                    elif block.get("type") == "text":
                        text = _strip_noise(block.get("text", ""))
                        if text:
                            if current_turn.response_text:
                                current_turn.response_text += "\n" + text
                            else:
                                current_turn.response_text = text
                    elif block.get("type") == "tool_use":
                        action = Action(
                            tool_name=block.get("name", ""),
                            tool_input=block.get("input", {}),
                        )
                        current_turn.actions.append(action)
                        tool_use_id = block.get("id")
                        if tool_use_id:
                            pending_actions[tool_use_id] = action

        elif entry_type == "user" and "toolUseResult" in entry:
            result = entry["toolUseResult"]
            content = entry.get("message", {}).get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        tool_use_id = block.get("tool_use_id")
                        if tool_use_id and tool_use_id in pending_actions:
                            action = pending_actions[tool_use_id]
                            action.result_stdout = result.get("stdout")
                            action.result_stderr = result.get("stderr")
                            action.success = not result.get("interrupted", False)

    result_turns = [t for t in turns if t.user_prompt]

    if return_session_id:
        return session_id, result_turns
    return result_turns
