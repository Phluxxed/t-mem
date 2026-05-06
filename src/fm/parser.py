from __future__ import annotations

import json
import re
from pathlib import Path
from typing import overload, Literal

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
def parse_session(jsonl_path: Path, *, return_session_id: Literal[False] = ...) -> list[Turn]: ...
@overload
def parse_session(jsonl_path: Path, *, return_session_id: Literal[True]) -> tuple[str, list[Turn]]: ...


def parse_session(
    jsonl_path: Path,
    *,
    return_session_id: bool = False,
) -> list[Turn] | tuple[str, list[Turn]]:
    """Parse a Claude Code JSONL session file into a list of Turns."""
    entries = []
    session_id = ""

    bad_lines = 0
    with open(jsonl_path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                import sys
                print(f"parser: skipping malformed JSON at line {lineno} of {jsonl_path.name}: {e}", file=sys.stderr)
                bad_lines += 1
                continue
            entries.append(entry)
            if not session_id and entry.get("sessionId"):
                session_id = entry["sessionId"]
    if bad_lines:
        import sys
        print(f"parser: {bad_lines} malformed line(s) skipped in {jsonl_path.name}", file=sys.stderr)

    skip_types = {"file-history-snapshot"}
    skip_subtypes = {"stop_hook_summary", "turn_duration", "compact_boundary", "local_command"}

    def _should_skip(entry: dict) -> bool:
        if entry.get("type") in skip_types:
            return True
        if entry.get("type") == "system" and entry.get("subtype") in skip_subtypes:
            return True
        return False

    # Build tree from ALL entries so compact_boundary UUIDs remain reachable
    # as parent references for their children (post-compaction entries).
    # Skipping compact_boundary before tree build would orphan those children.
    tree = _build_tree(entries)
    ordered = [e for e in _walk_tree(tree, None) if not _should_skip(e)]

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
                            is_error = block.get("is_error", False)
                            if isinstance(result, dict):
                                action.result_stdout = result.get("stdout")
                                action.result_stderr = result.get("stderr")
                                action.success = not (is_error or result.get("interrupted", False))
                            else:
                                action.success = not is_error

    result_turns = [t for t in turns if t.user_prompt]

    if return_session_id:
        return session_id, result_turns
    return result_turns


# Tool-input keys retained by the lean parser. Anything outside this set is
# dropped at parse time. Keep this in sync with whatever fm.baseline metrics
# actually read off `Action.tool_input`.
_LEAN_TOOL_INPUT_KEYS = ("file_path", "offset", "limit")

# Max bytes per JSONL line in the lean path. A single line over this size means
# the session contains a giant tool_result (typically megabytes of stdout) that
# would peak at 5-50MB during json.loads — enough to OOM when iterating over
# hundreds of files. p99 max-line in our corpus is ~135KB, so 200KB cuts only
# the long-tail outliers (~5-10 files out of ~3000).
_MAX_LINE_BYTES = 200_000


def _prune_entry_for_metrics(entry: dict) -> None:
    """Strip an entry in-place to retain only fields metric computation needs.

    This is the heart of the memory savings: tool results carry stdout/stderr
    blobs that can be megabytes each. We drop them before they hit the
    `entries` list, keeping memory bounded per-file regardless of result size.
    """
    # Tool results: keep only the metadata flags we need
    if "toolUseResult" in entry:
        tur = entry.get("toolUseResult")
        if isinstance(tur, dict):
            entry["toolUseResult"] = {"interrupted": tur.get("interrupted", False)}

    msg = entry.get("message")
    if not isinstance(msg, dict):
        return
    content = msg.get("content")
    if isinstance(content, str):
        # User prompts can be long; we only need a hint to satisfy
        # _is_user_prompt's content check, so truncate aggressively
        msg["content"] = content[:200]
        return
    if not isinstance(content, list):
        return

    lean_content: list[dict] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        bt = block.get("type")
        if bt == "tool_use":
            ti = block.get("input") or {}
            lean_input = {k: ti[k] for k in _LEAN_TOOL_INPUT_KEYS if k in ti}
            lean_content.append({
                "type": "tool_use",
                "id": block.get("id"),
                "name": block.get("name"),
                "input": lean_input,
            })
        elif bt == "tool_result":
            lean_content.append({
                "type": "tool_result",
                "tool_use_id": block.get("tool_use_id"),
                "is_error": block.get("is_error", False),
            })
        elif bt == "text":
            txt = block.get("text", "")
            if isinstance(txt, str):
                lean_content.append({"type": "text", "text": txt[:200]})
        # Drop "thinking" blocks and anything else — metrics don't read them
    msg["content"] = lean_content


def parse_session_lean(jsonl_path: Path) -> list[Turn]:
    """Memory-efficient parser for metric computation.

    Returns the same Turn/Action structure as parse_session, but with empty
    response_text/thinking and pruned tool_input. Use this when iterating
    over hundreds of session files in fm.baseline aggregation.
    """
    entries: list[dict] = []
    bad_lines = 0
    with open(jsonl_path) as f:
        for lineno, line in enumerate(f, 1):
            # Reject the whole file if any line exceeds the size cap — partial
            # parsing would leave Action.success flags wrong for that line's
            # tool_result, biasing error_rate. Whole-file rejection keeps every
            # included session's metrics complete.
            if len(line) > _MAX_LINE_BYTES:
                raise ValueError(
                    f"line {lineno} of {jsonl_path.name} is {len(line)} bytes "
                    f"(>{_MAX_LINE_BYTES}) — skipping whole file"
                )
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                import sys
                print(f"parser: skipping malformed JSON at line {lineno} of {jsonl_path.name}: {e}", file=sys.stderr)
                bad_lines += 1
                continue
            _prune_entry_for_metrics(entry)
            entries.append(entry)
    if bad_lines:
        import sys
        print(f"parser: {bad_lines} malformed line(s) skipped in {jsonl_path.name}", file=sys.stderr)

    skip_types = {"file-history-snapshot"}
    skip_subtypes = {"stop_hook_summary", "turn_duration", "compact_boundary", "local_command"}

    def _should_skip(entry: dict) -> bool:
        if entry.get("type") in skip_types:
            return True
        if entry.get("type") == "system" and entry.get("subtype") in skip_subtypes:
            return True
        return False

    tree = _build_tree(entries)
    ordered = [e for e in _walk_tree(tree, None) if not _should_skip(e)]

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
                    if block.get("type") == "tool_use":
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
                            is_error = block.get("is_error", False)
                            if isinstance(result, dict):
                                action.success = not (is_error or result.get("interrupted", False))
                            else:
                                action.success = not is_error

    return [t for t in turns if t.user_prompt]
