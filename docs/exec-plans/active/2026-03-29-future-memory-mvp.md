# future_memory MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI tool that extracts learning tips from Claude Code session logs and injects relevant ones into future sessions via a hook.

**Architecture:** Python CLI (`fm`) with four modules: parser (JSONL → turns), extractor (turns → tips via `claude` CLI), store (SQLite + embeddings), retriever (cosine similarity → formatted tips). Hook integration via `UserPromptSubmit`.

**Tech Stack:** Python 3.12+, SQLite, Voyage AI embeddings (HuggingFace fallback), `claude` CLI for extraction, `click` for CLI framework.

**Spec:** `docs/design-docs/2026-03-29-future-memory-mvp-design.md`

---

## File Structure

```
future_memory/
├── src/
│   └── fm/
│       ├── __init__.py           # Package init, version
│       ├── cli.py                # Click CLI entrypoints
│       ├── models.py             # Dataclasses: Turn, Action, Tip
│       ├── parser.py             # JSONL session parser
│       ├── extractor.py          # Tip extraction via claude CLI
│       ├── store.py              # SQLite tip store
│       ├── embeddings.py         # Embedding provider chain (Voyage → HF → None)
│       ├── retriever.py          # Cosine similarity retrieval
│       └── prompts/
│           └── extract.py        # Extraction prompt templates
├── tests/
│   ├── conftest.py               # Shared fixtures (sample JSONL, temp DB)
│   ├── test_models.py
│   ├── test_parser.py
│   ├── test_store.py
│   ├── test_embeddings.py
│   ├── test_retriever.py
│   ├── test_extractor.py
│   └── test_cli.py
├── pyproject.toml                # Project config, dependencies, [project.scripts] entry
└── ...
```

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/fm/__init__.py`
- Create: `src/fm/models.py`
- Create: `tests/conftest.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "future-memory"
version = "0.1.0"
description = "Trajectory-informed memory generation for self-improving agent systems"
requires-python = ">=3.12"
dependencies = [
    "click>=8.0",
    "voyageai>=0.3.0",
    "requests>=2.31",
    "numpy>=1.26",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
]

[project.scripts]
fm = "fm.cli:main"

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 2: Create package init**

```python
# src/fm/__init__.py
__version__ = "0.1.0"
```

- [ ] **Step 3: Create models.py with dataclasses**

```python
# src/fm/models.py
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Action:
    tool_name: str
    tool_input: dict
    result_stdout: str | None = None
    result_stderr: str | None = None
    success: bool = True


@dataclass
class Turn:
    user_prompt: str
    thinking: list[str] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)
    response_text: str = ""
    timestamp: str = ""
    cwd: str = ""


@dataclass
class Tip:
    category: str  # "strategy" | "recovery" | "optimization"
    content: str
    purpose: str
    steps: list[str]
    trigger: str
    priority: str  # "critical" | "high" | "medium" | "low"
    source_session_id: str
    source_project: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    negative_example: str | None = None
    task_context: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    VALID_CATEGORIES: tuple[str, ...] = ("strategy", "recovery", "optimization")
    VALID_PRIORITIES: tuple[str, ...] = ("critical", "high", "medium", "low")

    def __post_init__(self) -> None:
        if self.category not in self.VALID_CATEGORIES:
            raise ValueError(
                f"Invalid category: {self.category}. "
                f"Must be one of {self.VALID_CATEGORIES}"
            )
        if self.priority not in self.VALID_PRIORITIES:
            raise ValueError(
                f"Invalid priority: {self.priority}. "
                f"Must be one of {self.VALID_PRIORITIES}"
            )
```

- [ ] **Step 4: Write tests for models**

```python
# tests/test_models.py
import pytest

from fm.models import Action, Tip, Turn


class TestAction:
    def test_create_minimal(self) -> None:
        action = Action(tool_name="Bash", tool_input={"command": "ls"})
        assert action.tool_name == "Bash"
        assert action.success is True
        assert action.result_stdout is None

    def test_create_with_results(self) -> None:
        action = Action(
            tool_name="Read",
            tool_input={"file_path": "/tmp/f.txt"},
            result_stdout="file contents",
            result_stderr=None,
            success=True,
        )
        assert action.result_stdout == "file contents"


class TestTurn:
    def test_create_minimal(self) -> None:
        turn = Turn(user_prompt="fix the bug")
        assert turn.user_prompt == "fix the bug"
        assert turn.thinking == []
        assert turn.actions == []
        assert turn.response_text == ""

    def test_create_full(self) -> None:
        turn = Turn(
            user_prompt="fix the bug",
            thinking=["I should look at the error first"],
            actions=[Action(tool_name="Bash", tool_input={"command": "ls"})],
            response_text="Done.",
            timestamp="2026-03-29T01:00:00Z",
            cwd="/home/user/project",
        )
        assert len(turn.actions) == 1
        assert turn.cwd == "/home/user/project"


class TestTip:
    def test_create_valid(self) -> None:
        tip = Tip(
            category="strategy",
            content="Always check prerequisites before proceeding",
            purpose="Prevents failures from missing prerequisites",
            steps=["Check step 1", "Check step 2"],
            trigger="When task involves multi-step operations",
            priority="high",
            source_session_id="abc123",
            source_project="my-project",
        )
        assert tip.category == "strategy"
        assert tip.id  # auto-generated
        assert tip.created_at  # auto-generated

    def test_invalid_category_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid category"):
            Tip(
                category="bad",
                content="x",
                purpose="x",
                steps=[],
                trigger="x",
                priority="high",
                source_session_id="x",
                source_project="x",
            )

    def test_invalid_priority_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid priority"):
            Tip(
                category="strategy",
                content="x",
                purpose="x",
                steps=[],
                trigger="x",
                priority="urgent",
                source_session_id="x",
                source_project="x",
            )
```

- [ ] **Step 5: Create conftest with shared fixtures**

```python
# tests/conftest.py
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture
def sample_jsonl(tmp_path: Path) -> Path:
    """Minimal but realistic Claude Code session JSONL."""
    session_id = "test-session-001"
    entries = [
        {
            "type": "file-history-snapshot",
            "messageId": "msg-0",
            "snapshot": {"messageId": "msg-0", "trackedFileBackups": {}},
            "isSnapshotUpdate": False,
        },
        {
            "parentUuid": None,
            "isSidechain": False,
            "type": "user",
            "message": {"role": "user", "content": "Fix the login bug"},
            "isMeta": False,
            "uuid": "u1",
            "timestamp": "2026-03-29T01:00:00Z",
            "sessionId": session_id,
            "cwd": "/home/user/project",
        },
        {
            "parentUuid": "u1",
            "isSidechain": False,
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "I should check the auth module"},
                    {
                        "type": "tool_use",
                        "id": "tool1",
                        "name": "Read",
                        "input": {"file_path": "/home/user/project/auth.py"},
                    },
                ],
            },
            "uuid": "a1",
            "timestamp": "2026-03-29T01:00:01Z",
            "sessionId": session_id,
        },
        {
            "parentUuid": "a1",
            "isSidechain": False,
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool1",
                        "content": "def login(user):\n    return True",
                    }
                ],
            },
            "uuid": "u2",
            "timestamp": "2026-03-29T01:00:02Z",
            "toolUseResult": {
                "stdout": "def login(user):\n    return True",
                "stderr": "",
                "interrupted": False,
            },
            "sessionId": session_id,
        },
        {
            "parentUuid": "u2",
            "isSidechain": False,
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I found the issue. The login function always returns True."},
                ],
            },
            "uuid": "a2",
            "timestamp": "2026-03-29T01:00:03Z",
            "sessionId": session_id,
        },
        {
            "type": "system",
            "subtype": "turn_duration",
            "parentUuid": "a2",
            "isMeta": True,
            "isSidechain": False,
            "durationMs": 3000,
            "uuid": "s1",
            "timestamp": "2026-03-29T01:00:03Z",
            "sessionId": session_id,
        },
        {
            "parentUuid": "a2",
            "isSidechain": False,
            "type": "user",
            "message": {"role": "user", "content": "Great, fix it please"},
            "isMeta": False,
            "uuid": "u3",
            "timestamp": "2026-03-29T01:01:00Z",
            "sessionId": session_id,
            "cwd": "/home/user/project",
        },
        {
            "parentUuid": "u3",
            "isSidechain": False,
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "I need to add proper password checking"},
                    {
                        "type": "tool_use",
                        "id": "tool2",
                        "name": "Edit",
                        "input": {
                            "file_path": "/home/user/project/auth.py",
                            "old_string": "return True",
                            "new_string": "return check_password(user)",
                        },
                    },
                ],
            },
            "uuid": "a3",
            "timestamp": "2026-03-29T01:01:01Z",
            "sessionId": session_id,
        },
        {
            "parentUuid": "a3",
            "isSidechain": False,
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool2",
                        "content": "File updated successfully",
                    }
                ],
            },
            "uuid": "u4",
            "timestamp": "2026-03-29T01:01:02Z",
            "toolUseResult": {
                "stdout": "File updated successfully",
                "stderr": "",
                "interrupted": False,
            },
            "sessionId": session_id,
        },
        {
            "parentUuid": "u4",
            "isSidechain": False,
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Fixed. The login function now checks the password."},
                ],
            },
            "uuid": "a4",
            "timestamp": "2026-03-29T01:01:03Z",
            "sessionId": session_id,
        },
        {
            "type": "system",
            "subtype": "stop_hook_summary",
            "parentUuid": "a4",
            "isSidechain": False,
            "hasOutput": False,
            "hookCount": 0,
            "hookErrors": [],
            "hookInfos": [],
            "stopReason": "end_turn",
            "uuid": "s2",
            "timestamp": "2026-03-29T01:01:03Z",
            "sessionId": session_id,
        },
    ]

    jsonl_path = tmp_path / "test-session-001.jsonl"
    with open(jsonl_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    return jsonl_path


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    """Path for a temporary SQLite database."""
    return tmp_path / "test_tips.db"
```

- [ ] **Step 6: Set up virtual environment and verify tests run**

Run:
```bash
cd /home/brummerv/future_memory
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
pytest tests/test_models.py -v
```

Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
git init
git add pyproject.toml src/ tests/
git commit -m "feat: project scaffolding with models and test fixtures"
```

---

### Task 2: Trajectory Parser

**Files:**
- Create: `src/fm/parser.py`
- Create: `tests/test_parser.py`

- [ ] **Step 1: Write failing tests for the parser**

```python
# tests/test_parser.py
from pathlib import Path

from fm.models import Action, Turn
from fm.parser import parse_session


class TestParseSession:
    def test_extracts_turns(self, sample_jsonl: Path) -> None:
        turns = parse_session(sample_jsonl)
        assert len(turns) == 2  # Two user prompts = two turns

    def test_first_turn_user_prompt(self, sample_jsonl: Path) -> None:
        turns = parse_session(sample_jsonl)
        assert turns[0].user_prompt == "Fix the login bug"

    def test_first_turn_thinking(self, sample_jsonl: Path) -> None:
        turns = parse_session(sample_jsonl)
        assert "I should check the auth module" in turns[0].thinking

    def test_first_turn_actions(self, sample_jsonl: Path) -> None:
        turns = parse_session(sample_jsonl)
        assert len(turns[0].actions) == 1
        assert turns[0].actions[0].tool_name == "Read"
        assert turns[0].actions[0].result_stdout == "def login(user):\n    return True"

    def test_first_turn_response_text(self, sample_jsonl: Path) -> None:
        turns = parse_session(sample_jsonl)
        assert "always returns True" in turns[0].response_text

    def test_second_turn_user_prompt(self, sample_jsonl: Path) -> None:
        turns = parse_session(sample_jsonl)
        assert turns[1].user_prompt == "Great, fix it please"

    def test_second_turn_actions(self, sample_jsonl: Path) -> None:
        turns = parse_session(sample_jsonl)
        assert len(turns[1].actions) == 1
        assert turns[1].actions[0].tool_name == "Edit"

    def test_drops_system_entries(self, sample_jsonl: Path) -> None:
        turns = parse_session(sample_jsonl)
        # System entries (turn_duration, stop_hook_summary) should not appear
        for turn in turns:
            for action in turn.actions:
                assert action.tool_name not in ("turn_duration", "stop_hook_summary")

    def test_drops_file_history_snapshots(self, sample_jsonl: Path) -> None:
        turns = parse_session(sample_jsonl)
        # Should have parsed without error despite snapshot entries
        assert len(turns) == 2

    def test_extracts_metadata(self, sample_jsonl: Path) -> None:
        turns = parse_session(sample_jsonl)
        assert turns[0].cwd == "/home/user/project"
        assert turns[0].timestamp == "2026-03-29T01:00:00Z"

    def test_extracts_session_id(self, sample_jsonl: Path) -> None:
        session_id, turns = parse_session(sample_jsonl, return_session_id=True)
        assert session_id == "test-session-001"


class TestParseSessionEdgeCases:
    def test_empty_file(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        turns = parse_session(empty)
        assert turns == []

    def test_meta_only_messages_skipped(self, tmp_path: Path) -> None:
        """Messages with isMeta=True should be skipped."""
        entries = [
            {
                "type": "user",
                "message": {"role": "user", "content": "some meta thing"},
                "isMeta": True,
                "uuid": "m1",
                "parentUuid": None,
                "isSidechain": False,
                "sessionId": "s1",
                "timestamp": "2026-03-29T01:00:00Z",
            },
        ]
        jsonl_path = tmp_path / "meta.jsonl"
        with open(jsonl_path, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        turns = parse_session(jsonl_path)
        assert turns == []
```

Add import at top of test file:
```python
import json
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_parser.py -v`

Expected: FAIL — `ImportError: cannot import name 'parse_session' from 'fm.parser'`

- [ ] **Step 3: Implement the parser**

```python
# src/fm/parser.py
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
        # Check if this is a tool_result (not a user prompt)
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                return None
        # Otherwise concatenate text blocks
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block["text"])
        return _strip_noise(" ".join(texts)) if texts else None
    return None


def _build_tree(entries: list[dict]) -> dict[str | None, list[dict]]:
    """Build a parent → children mapping from JSONL entries."""
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
    """Parse a Claude Code JSONL session file into a list of Turns.

    Args:
        jsonl_path: Path to the .jsonl session file.
        return_session_id: If True, return (session_id, turns) tuple.

    Returns:
        List of Turn objects, or (session_id, turns) if return_session_id is True.
    """
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

    # Filter out noise entry types
    skip_types = {"file-history-snapshot"}
    skip_subtypes = {"stop_hook_summary", "turn_duration", "compact_boundary", "local_command"}

    filtered = []
    for entry in entries:
        if entry.get("type") in skip_types:
            continue
        if entry.get("type") == "system" and entry.get("subtype") in skip_subtypes:
            continue
        filtered.append(entry)

    # Build tree and walk in order
    tree = _build_tree(filtered)
    ordered = _walk_tree(tree, None)

    # Group into turns: each turn starts with a user prompt
    turns: list[Turn] = []
    current_turn: Turn | None = None

    # Also maintain a lookup from tool_use_id to Action for attaching results
    pending_actions: dict[str, Action] = {}

    for entry in ordered:
        entry_type = entry.get("type")

        if entry_type == "user" and _is_user_prompt(entry):
            # Start a new turn
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
            # Attach tool result to the pending action
            result = entry["toolUseResult"]
            # Find the matching action via tool_use_id in the content
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_parser.py -v`

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/fm/parser.py tests/test_parser.py
git commit -m "feat: trajectory parser — JSONL session logs to structured turns"
```

---

### Task 3: Tip Store

**Files:**
- Create: `src/fm/store.py`
- Create: `tests/test_store.py`

- [ ] **Step 1: Write failing tests for the store**

```python
# tests/test_store.py
import json
from pathlib import Path

import pytest

from fm.models import Tip
from fm.store import TipStore


def _make_tip(**overrides) -> Tip:
    defaults = dict(
        category="strategy",
        content="Always verify prerequisites",
        purpose="Prevents failures",
        steps=["Step 1", "Step 2"],
        trigger="When task involves multi-step operations",
        priority="high",
        source_session_id="session-001",
        source_project="test-project",
    )
    defaults.update(overrides)
    return Tip(**defaults)


class TestTipStore:
    def test_init_creates_tables(self, tmp_db: Path) -> None:
        store = TipStore(tmp_db)
        # Should not raise
        assert tmp_db.exists()

    def test_add_and_get_tip(self, tmp_db: Path) -> None:
        store = TipStore(tmp_db)
        tip = _make_tip()
        store.add_tip(tip)
        retrieved = store.get_tip(tip.id)
        assert retrieved is not None
        assert retrieved.content == tip.content
        assert retrieved.category == tip.category

    def test_get_nonexistent_returns_none(self, tmp_db: Path) -> None:
        store = TipStore(tmp_db)
        assert store.get_tip("nonexistent") is None

    def test_list_tips(self, tmp_db: Path) -> None:
        store = TipStore(tmp_db)
        store.add_tip(_make_tip(content="Tip 1"))
        store.add_tip(_make_tip(content="Tip 2"))
        tips = store.list_tips()
        assert len(tips) == 2

    def test_list_tips_filter_by_category(self, tmp_db: Path) -> None:
        store = TipStore(tmp_db)
        store.add_tip(_make_tip(category="strategy"))
        store.add_tip(_make_tip(category="recovery"))
        tips = store.list_tips(category="strategy")
        assert len(tips) == 1
        assert tips[0].category == "strategy"

    def test_add_tip_with_embedding(self, tmp_db: Path) -> None:
        store = TipStore(tmp_db)
        tip = _make_tip()
        embedding = [0.1, 0.2, 0.3]
        store.add_tip(tip, embedding=embedding, embedding_provider="voyage")
        raw = store.get_tip_with_embedding(tip.id)
        assert raw is not None
        assert raw["embedding_provider"] == "voyage"
        assert len(raw["embedding"]) == 3

    def test_get_all_embeddings(self, tmp_db: Path) -> None:
        store = TipStore(tmp_db)
        store.add_tip(_make_tip(content="Tip 1"), embedding=[0.1, 0.2], embedding_provider="voyage")
        store.add_tip(_make_tip(content="Tip 2"), embedding=[0.3, 0.4], embedding_provider="voyage")
        store.add_tip(_make_tip(content="Tip 3"))  # No embedding
        results = store.get_tips_with_embeddings(provider="voyage")
        assert len(results) == 2

    def test_mark_session_processed(self, tmp_db: Path) -> None:
        store = TipStore(tmp_db)
        store.mark_session_processed("session-001", "/path/to/file.jsonl", tip_count=3)
        assert store.is_session_processed("session-001")

    def test_unprocessed_session(self, tmp_db: Path) -> None:
        store = TipStore(tmp_db)
        assert not store.is_session_processed("session-999")

    def test_steps_stored_as_json(self, tmp_db: Path) -> None:
        store = TipStore(tmp_db)
        tip = _make_tip(steps=["Do A", "Do B", "Do C"])
        store.add_tip(tip)
        retrieved = store.get_tip(tip.id)
        assert retrieved is not None
        assert retrieved.steps == ["Do A", "Do B", "Do C"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_store.py -v`

Expected: FAIL — `ImportError: cannot import name 'TipStore' from 'fm.store'`

- [ ] **Step 3: Implement the store**

```python
# src/fm/store.py
from __future__ import annotations

import json
import sqlite3
import struct
from datetime import datetime, timezone
from pathlib import Path

from fm.models import Tip

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tips (
    id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    content TEXT NOT NULL,
    purpose TEXT,
    steps TEXT,
    trigger TEXT,
    negative_example TEXT,
    priority TEXT NOT NULL,
    source_session_id TEXT NOT NULL,
    source_project TEXT,
    task_context TEXT,
    embedding BLOB,
    embedding_provider TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS processed_sessions (
    session_id TEXT PRIMARY KEY,
    jsonl_path TEXT NOT NULL,
    processed_at TEXT NOT NULL,
    tip_count INTEGER
);
"""


def _pack_embedding(embedding: list[float]) -> bytes:
    """Pack a list of floats into a compact binary blob."""
    return struct.pack(f"{len(embedding)}f", *embedding)


def _unpack_embedding(blob: bytes) -> list[float]:
    """Unpack a binary blob back into a list of floats."""
    count = len(blob) // 4  # 4 bytes per float
    return list(struct.unpack(f"{count}f", blob))


class TipStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    def add_tip(
        self,
        tip: Tip,
        *,
        embedding: list[float] | None = None,
        embedding_provider: str | None = None,
    ) -> None:
        embedding_blob = _pack_embedding(embedding) if embedding else None
        self._conn.execute(
            """INSERT INTO tips (
                id, category, content, purpose, steps, trigger,
                negative_example, priority, source_session_id,
                source_project, task_context, embedding,
                embedding_provider, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                tip.id,
                tip.category,
                tip.content,
                tip.purpose,
                json.dumps(tip.steps),
                tip.trigger,
                tip.negative_example,
                tip.priority,
                tip.source_session_id,
                tip.source_project,
                tip.task_context,
                embedding_blob,
                embedding_provider,
                tip.created_at,
            ),
        )
        self._conn.commit()

    def get_tip(self, tip_id: str) -> Tip | None:
        row = self._conn.execute(
            "SELECT * FROM tips WHERE id = ?", (tip_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_tip(row)

    def get_tip_with_embedding(self, tip_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM tips WHERE id = ?", (tip_id,)
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        if result["embedding"]:
            result["embedding"] = _unpack_embedding(result["embedding"])
        return result

    def get_tips_with_embeddings(self, provider: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM tips WHERE embedding IS NOT NULL AND embedding_provider = ?",
            (provider,),
        ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["embedding"] = _unpack_embedding(d["embedding"])
            results.append(d)
        return results

    def list_tips(self, *, category: str | None = None) -> list[Tip]:
        if category:
            rows = self._conn.execute(
                "SELECT * FROM tips WHERE category = ? ORDER BY created_at DESC",
                (category,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM tips ORDER BY created_at DESC"
            ).fetchall()
        return [self._row_to_tip(row) for row in rows]

    def mark_session_processed(
        self, session_id: str, jsonl_path: str, tip_count: int
    ) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO processed_sessions
            (session_id, jsonl_path, processed_at, tip_count)
            VALUES (?, ?, ?, ?)""",
            (
                session_id,
                jsonl_path,
                datetime.now(timezone.utc).isoformat(),
                tip_count,
            ),
        )
        self._conn.commit()

    def is_session_processed(self, session_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM processed_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return row is not None

    def _row_to_tip(self, row: sqlite3.Row) -> Tip:
        return Tip(
            id=row["id"],
            category=row["category"],
            content=row["content"],
            purpose=row["purpose"] or "",
            steps=json.loads(row["steps"]) if row["steps"] else [],
            trigger=row["trigger"] or "",
            negative_example=row["negative_example"],
            priority=row["priority"],
            source_session_id=row["source_session_id"],
            source_project=row["source_project"] or "",
            task_context=row["task_context"],
            created_at=row["created_at"],
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_store.py -v`

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/fm/store.py tests/test_store.py
git commit -m "feat: SQLite tip store with embedding support"
```

---

### Task 4: Embedding Provider Chain

**Files:**
- Create: `src/fm/embeddings.py`
- Create: `tests/test_embeddings.py`

- [ ] **Step 1: Write failing tests for the embedding layer**

```python
# tests/test_embeddings.py
from unittest.mock import MagicMock, patch

import pytest

from fm.embeddings import EmbeddingResult, embed_text, get_available_provider


class TestEmbeddingResult:
    def test_create(self) -> None:
        result = EmbeddingResult(vector=[0.1, 0.2, 0.3], provider="voyage")
        assert result.provider == "voyage"
        assert len(result.vector) == 3


class TestGetAvailableProvider:
    @patch.dict("os.environ", {"VOYAGE_API_KEY": "test-key"})
    def test_voyage_available(self) -> None:
        provider = get_available_provider()
        assert provider in ("voyage", "huggingface")

    @patch.dict("os.environ", {}, clear=True)
    def test_falls_back_when_no_keys(self) -> None:
        provider = get_available_provider()
        # Should return huggingface (free, no key) or None
        assert provider in ("huggingface", None)


class TestEmbedText:
    def test_returns_embedding_result(self) -> None:
        # Mock the actual API call
        with patch("fm.embeddings._embed_voyage") as mock_voyage:
            mock_voyage.return_value = [0.1, 0.2, 0.3]
            with patch.dict("os.environ", {"VOYAGE_API_KEY": "test-key"}):
                result = embed_text("test query", provider="voyage")
                assert result is not None
                assert result.provider == "voyage"
                assert len(result.vector) == 3

    def test_returns_none_when_provider_unavailable(self) -> None:
        result = embed_text("test query", provider=None)
        assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_embeddings.py -v`

Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement the embedding provider chain**

```python
# src/fm/embeddings.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import requests


@dataclass
class EmbeddingResult:
    vector: list[float]
    provider: str  # "voyage" | "huggingface"


def get_available_provider() -> str | None:
    """Return the best available embedding provider, or None."""
    if os.environ.get("VOYAGE_API_KEY"):
        return "voyage"
    # HuggingFace Inference API is free, no key required
    # but may be rate-limited or unavailable
    try:
        resp = requests.head(
            "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2",
            timeout=2,
        )
        if resp.status_code < 500:
            return "huggingface"
    except requests.RequestException:
        pass
    return None


def embed_text(text: str, *, provider: str | None = None) -> EmbeddingResult | None:
    """Embed a text string using the specified provider.

    Args:
        text: The text to embed.
        provider: Which provider to use. If None, returns None.

    Returns:
        EmbeddingResult with the vector and provider name, or None if unavailable.
    """
    if provider is None:
        return None

    if provider == "voyage":
        vector = _embed_voyage(text)
    elif provider == "huggingface":
        vector = _embed_huggingface(text)
    else:
        return None

    if vector is None:
        return None
    return EmbeddingResult(vector=vector, provider=provider)


def _embed_voyage(text: str) -> list[float] | None:
    """Embed text using Voyage AI API."""
    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        return None
    try:
        import voyageai

        client = voyageai.Client(api_key=api_key)
        result = client.embed([text], model="voyage-3-lite")
        return result.embeddings[0]
    except Exception:
        return None


def _embed_huggingface(text: str) -> list[float] | None:
    """Embed text using HuggingFace Inference API (free tier)."""
    api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
    try:
        resp = requests.post(
            api_url,
            json={"inputs": text, "options": {"wait_for_model": True}},
            timeout=10,
        )
        resp.raise_for_status()
        vector = resp.json()
        # The API returns a list of token embeddings; we want the sentence embedding
        # For sentence-transformers, it returns a single vector
        if isinstance(vector, list) and isinstance(vector[0], float):
            return vector
        if isinstance(vector, list) and isinstance(vector[0], list):
            # Mean pooling over token embeddings
            import numpy as np

            return list(np.mean(vector, axis=0).astype(float))
        return None
    except Exception:
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_embeddings.py -v`

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/fm/embeddings.py tests/test_embeddings.py
git commit -m "feat: embedding provider chain — Voyage AI → HuggingFace → None"
```

---

### Task 5: Tip Retriever

**Files:**
- Create: `src/fm/retriever.py`
- Create: `tests/test_retriever.py`

- [ ] **Step 1: Write failing tests for the retriever**

```python
# tests/test_retriever.py
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from fm.embeddings import EmbeddingResult
from fm.models import Tip
from fm.retriever import format_tips, retrieve_tips
from fm.store import TipStore


def _make_tip(content: str, trigger: str = "general", **kw) -> Tip:
    defaults = dict(
        category="strategy",
        content=content,
        purpose="Test purpose",
        steps=["Step 1"],
        trigger=trigger,
        priority="high",
        source_session_id="s1",
        source_project="proj",
    )
    defaults.update(kw)
    return Tip(**defaults)


class TestRetrieveTips:
    def test_returns_matching_tips(self, tmp_db: Path) -> None:
        store = TipStore(tmp_db)
        # Store tips with embeddings — embed them as known vectors
        tip1 = _make_tip("Check prerequisites before deployment")
        tip2 = _make_tip("Use bulk operations for batch deletes")
        # tip1 vector close to query, tip2 vector far from query
        store.add_tip(tip1, embedding=[1.0, 0.0, 0.0], embedding_provider="voyage")
        store.add_tip(tip2, embedding=[0.0, 1.0, 0.0], embedding_provider="voyage")

        # Query vector is close to tip1
        with patch("fm.retriever.embed_text") as mock_embed:
            mock_embed.return_value = EmbeddingResult(
                vector=[0.95, 0.05, 0.0], provider="voyage"
            )
            results = retrieve_tips("deploy the app", store)

        assert len(results) >= 1
        assert results[0].content == "Check prerequisites before deployment"

    def test_filters_below_threshold(self, tmp_db: Path) -> None:
        store = TipStore(tmp_db)
        tip = _make_tip("Unrelated tip")
        # Orthogonal vector
        store.add_tip(tip, embedding=[0.0, 0.0, 1.0], embedding_provider="voyage")

        with patch("fm.retriever.embed_text") as mock_embed:
            mock_embed.return_value = EmbeddingResult(
                vector=[1.0, 0.0, 0.0], provider="voyage"
            )
            results = retrieve_tips("deploy the app", store, threshold=0.6)

        assert len(results) == 0

    def test_returns_top_k(self, tmp_db: Path) -> None:
        store = TipStore(tmp_db)
        for i in range(10):
            tip = _make_tip(f"Tip number {i}")
            # All similar vectors
            vec = [0.9, 0.1 * (i / 10), 0.0]
            store.add_tip(tip, embedding=vec, embedding_provider="voyage")

        with patch("fm.retriever.embed_text") as mock_embed:
            mock_embed.return_value = EmbeddingResult(
                vector=[1.0, 0.0, 0.0], provider="voyage"
            )
            results = retrieve_tips("query", store, top_k=5)

        assert len(results) <= 5

    def test_returns_empty_when_no_embeddings(self, tmp_db: Path) -> None:
        store = TipStore(tmp_db)
        with patch("fm.retriever.embed_text") as mock_embed:
            mock_embed.return_value = None
            results = retrieve_tips("query", store)
        assert results == []


class TestFormatTips:
    def test_formats_single_tip(self) -> None:
        tip = _make_tip(
            content="Always check prerequisites",
            trigger="When deploying",
            priority="high",
            category="strategy",
        )
        tip.source_session_id = "abc123"
        tip.created_at = "2026-03-29T01:00:00+00:00"
        output = format_tips([tip])
        assert "[PRIORITY: HIGH]" in output
        assert "Strategy Tip" in output
        assert "Always check prerequisites" in output
        assert "When deploying" in output
        assert "abc123" in output

    def test_formats_empty_list(self) -> None:
        output = format_tips([])
        assert output == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_retriever.py -v`

Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement the retriever**

```python
# src/fm/retriever.py
from __future__ import annotations

import numpy as np

from fm.embeddings import embed_text
from fm.models import Tip
from fm.store import TipStore


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a, dtype=np.float64)
    b_arr = np.array(b, dtype=np.float64)
    dot = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def retrieve_tips(
    query: str,
    store: TipStore,
    *,
    threshold: float = 0.6,
    top_k: int = 5,
    provider: str | None = None,
) -> list[Tip]:
    """Retrieve relevant tips for a query using cosine similarity.

    Args:
        query: The user's prompt text.
        store: The tip store to search.
        threshold: Minimum cosine similarity (default 0.6).
        top_k: Maximum number of tips to return (default 5).
        provider: Embedding provider override. If None, auto-detect.

    Returns:
        List of matching Tip objects, sorted by relevance.
    """
    if provider is None:
        from fm.embeddings import get_available_provider

        provider = get_available_provider()

    query_result = embed_text(query, provider=provider)
    if query_result is None:
        return []

    # Get all tips with embeddings from the same provider
    stored = store.get_tips_with_embeddings(provider=query_result.provider)
    if not stored:
        return []

    # Score each tip
    scored: list[tuple[float, dict]] = []
    for tip_row in stored:
        score = _cosine_similarity(query_result.vector, tip_row["embedding"])
        if score >= threshold:
            scored.append((score, tip_row))

    # Sort by score descending, take top_k
    scored.sort(key=lambda x: x[0], reverse=True)
    scored = scored[:top_k]

    # Convert back to Tip objects
    return [store._row_to_tip(row) for _, row in scored]


def format_tips(tips: list[Tip]) -> str:
    """Format tips for prompt injection."""
    if not tips:
        return ""

    parts = []
    for tip in tips:
        category_label = f"{tip.category.title()} Tip"
        priority_label = tip.priority.upper()
        date = tip.created_at[:10] if tip.created_at else "unknown"

        section = f"[PRIORITY: {priority_label}] {category_label}:\n"
        section += f"{tip.content}\n\n"
        section += f"Apply when: {tip.trigger}\n"

        if tip.steps:
            section += "Steps:\n"
            for i, step in enumerate(tip.steps, 1):
                section += f"{i}. {step}\n"

        if tip.negative_example:
            section += f"\nAvoid: {tip.negative_example}\n"

        section += f"\nSource: session {tip.source_session_id} ({date})"
        parts.append(section)

    return "\n---\n".join(parts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_retriever.py -v`

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/fm/retriever.py tests/test_retriever.py
git commit -m "feat: tip retriever with cosine similarity and formatting"
```

---

### Task 6: Extraction Prompt Templates

**Files:**
- Create: `src/fm/prompts/__init__.py`
- Create: `src/fm/prompts/extract.py`
- Create: `tests/test_extractor.py` (prompt template tests only — extraction integration tested in Task 7)

- [ ] **Step 1: Write tests for prompt building**

```python
# tests/test_extractor.py
import json

from fm.models import Action, Turn
from fm.prompts.extract import build_extraction_prompt


class TestBuildExtractionPrompt:
    def test_includes_turns(self) -> None:
        turns = [
            Turn(
                user_prompt="Fix the bug",
                thinking=["Need to check auth"],
                actions=[
                    Action(
                        tool_name="Read",
                        tool_input={"file_path": "auth.py"},
                        result_stdout="def login(): return True",
                    )
                ],
                response_text="Found the issue.",
                timestamp="2026-03-29T01:00:00Z",
                cwd="/project",
            )
        ]
        prompt = build_extraction_prompt(turns, session_id="s1", project="my-proj")
        assert "Fix the bug" in prompt
        assert "Need to check auth" in prompt
        assert "Read" in prompt
        assert "auth.py" in prompt

    def test_includes_json_schema(self) -> None:
        turns = [Turn(user_prompt="Do something")]
        prompt = build_extraction_prompt(turns, session_id="s1", project="proj")
        assert '"category"' in prompt
        assert '"strategy"' in prompt
        assert '"recovery"' in prompt
        assert '"optimization"' in prompt

    def test_includes_few_shot_examples(self) -> None:
        turns = [Turn(user_prompt="Do something")]
        prompt = build_extraction_prompt(turns, session_id="s1", project="proj")
        assert "Example" in prompt or "example" in prompt

    def test_truncates_large_tool_output(self) -> None:
        turns = [
            Turn(
                user_prompt="Run it",
                actions=[
                    Action(
                        tool_name="Bash",
                        tool_input={"command": "cat bigfile"},
                        result_stdout="x" * 20000,
                    )
                ],
            )
        ]
        prompt = build_extraction_prompt(turns, session_id="s1", project="proj")
        # Should be truncated to something reasonable
        assert len(prompt) < 100000
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_extractor.py -v`

Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement the prompt template**

```python
# src/fm/prompts/__init__.py
```

```python
# src/fm/prompts/extract.py
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
    """Build the tip extraction prompt from parsed turns.

    Args:
        turns: Parsed turns from the session.
        session_id: The session identifier for provenance.
        project: The project name for provenance.

    Returns:
        The complete prompt string to send to the LLM.
    """
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_extractor.py -v`

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/fm/prompts/ tests/test_extractor.py
git commit -m "feat: extraction prompt template with few-shot examples"
```

---

### Task 7: Tip Extractor (claude CLI Integration)

**Files:**
- Create: `src/fm/extractor.py`
- Modify: `tests/test_extractor.py` (add integration tests)

- [ ] **Step 1: Write failing tests for the extractor**

Add to `tests/test_extractor.py`:

```python
from unittest.mock import patch, MagicMock
from fm.extractor import extract_tips_from_session
from fm.models import Turn, Action
from pathlib import Path
import json


class TestExtractTipsFromSession:
    def test_calls_claude_cli_and_parses_response(self, tmp_path: Path) -> None:
        """Test that the extractor calls claude and parses JSON tips."""
        mock_tips_json = json.dumps({
            "tips": [
                {
                    "category": "strategy",
                    "content": "Always read files before editing",
                    "purpose": "Prevents edit failures from stale content",
                    "steps": ["Read the file", "Then edit"],
                    "trigger": "When editing files",
                    "negative_example": None,
                    "priority": "high",
                    "task_context": None,
                }
            ]
        })

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = mock_tips_json

        turns = [
            Turn(
                user_prompt="Fix the bug",
                thinking=["Let me check"],
                actions=[],
                response_text="Done.",
            )
        ]

        with patch("fm.extractor.subprocess.run", return_value=mock_result):
            tips = extract_tips_from_session(
                turns, session_id="s1", project="proj"
            )

        assert len(tips) == 1
        assert tips[0].category == "strategy"
        assert tips[0].source_session_id == "s1"
        assert tips[0].source_project == "proj"

    def test_returns_empty_on_cli_failure(self) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "error"

        turns = [Turn(user_prompt="Do something")]

        with patch("fm.extractor.subprocess.run", return_value=mock_result):
            tips = extract_tips_from_session(
                turns, session_id="s1", project="proj"
            )

        assert tips == []

    def test_returns_empty_on_invalid_json(self) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "This is not JSON"

        turns = [Turn(user_prompt="Do something")]

        with patch("fm.extractor.subprocess.run", return_value=mock_result):
            tips = extract_tips_from_session(
                turns, session_id="s1", project="proj"
            )

        assert tips == []

    def test_skips_empty_turns(self) -> None:
        """Don't bother calling claude for sessions with no real content."""
        tips = extract_tips_from_session([], session_id="s1", project="proj")
        assert tips == []
```

- [ ] **Step 2: Run new tests to verify they fail**

Run: `pytest tests/test_extractor.py::TestExtractTipsFromSession -v`

Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement the extractor**

```python
# src/fm/extractor.py
from __future__ import annotations

import json
import re
import subprocess
import sys
from fm.models import Tip, Turn
from fm.prompts.extract import build_extraction_prompt


def _parse_tips_json(raw: str, session_id: str, project: str) -> list[Tip]:
    """Parse the LLM's JSON response into Tip objects."""
    # Try to extract JSON from the response (in case there's surrounding text)
    json_match = re.search(r"\{[\s\S]*\}", raw)
    if not json_match:
        return []

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return []

    tips_data = data.get("tips", [])
    if not isinstance(tips_data, list):
        return []

    tips = []
    for item in tips_data:
        try:
            tip = Tip(
                category=item["category"],
                content=item["content"],
                purpose=item.get("purpose", ""),
                steps=item.get("steps", []),
                trigger=item.get("trigger", ""),
                negative_example=item.get("negative_example"),
                priority=item.get("priority", "medium"),
                source_session_id=session_id,
                source_project=project,
                task_context=item.get("task_context"),
            )
            tips.append(tip)
        except (KeyError, ValueError):
            # Skip malformed tips
            continue

    return tips


def extract_tips_from_session(
    turns: list[Turn],
    *,
    session_id: str,
    project: str,
    model: str = "sonnet",
) -> list[Tip]:
    """Extract tips from parsed session turns using the claude CLI.

    Args:
        turns: Parsed turns from the session.
        session_id: Session identifier for provenance.
        project: Project name for provenance.
        model: Claude model to use (default: sonnet).

    Returns:
        List of extracted Tip objects.
    """
    if not turns:
        return []

    prompt = build_extraction_prompt(turns, session_id=session_id, project=project)

    try:
        result = subprocess.run(
            [
                "claude",
                "--print",
                "--model", model,
                "--prompt", prompt,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []

    if result.returncode != 0:
        print(
            f"Warning: claude CLI returned {result.returncode}: {result.stderr}",
            file=sys.stderr,
        )
        return []

    return _parse_tips_json(result.stdout, session_id, project)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_extractor.py -v`

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/fm/extractor.py tests/test_extractor.py
git commit -m "feat: tip extractor — calls claude CLI and parses structured tips"
```

---

### Task 8: CLI Interface

**Files:**
- Create: `src/fm/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests for the CLI**

```python
# tests/test_cli.py
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from click.testing import CliRunner

from fm.cli import main


class TestExtractCommand:
    def test_extract_single_session(self, sample_jsonl: Path, tmp_path: Path) -> None:
        db_path = tmp_path / "tips.db"
        mock_tips_json = json.dumps({
            "tips": [
                {
                    "category": "strategy",
                    "content": "Read before editing",
                    "purpose": "Prevents failures",
                    "steps": ["Read", "Edit"],
                    "trigger": "When editing files",
                    "priority": "high",
                }
            ]
        })
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = mock_tips_json

        runner = CliRunner()
        with patch("fm.extractor.subprocess.run", return_value=mock_result):
            with patch("fm.embeddings._embed_voyage", return_value=[0.1, 0.2]):
                with patch.dict("os.environ", {"VOYAGE_API_KEY": "test"}):
                    result = runner.invoke(
                        main,
                        ["extract", str(sample_jsonl), "--db", str(db_path)],
                    )

        assert result.exit_code == 0
        assert "Extracted" in result.output or "tip" in result.output.lower()

    def test_extract_skips_processed(self, sample_jsonl: Path, tmp_path: Path) -> None:
        db_path = tmp_path / "tips.db"
        from fm.store import TipStore

        store = TipStore(db_path)
        store.mark_session_processed("test-session-001", str(sample_jsonl), tip_count=1)

        runner = CliRunner()
        result = runner.invoke(
            main, ["extract", str(sample_jsonl), "--db", str(db_path)]
        )
        assert result.exit_code == 0
        assert "already processed" in result.output.lower() or "skip" in result.output.lower()


class TestRetrieveCommand:
    def test_retrieve_outputs_tips(self, tmp_path: Path) -> None:
        db_path = tmp_path / "tips.db"
        from fm.models import Tip
        from fm.store import TipStore

        store = TipStore(db_path)
        tip = Tip(
            category="strategy",
            content="Always check prerequisites",
            purpose="Prevents failures",
            steps=["Check step 1"],
            trigger="When deploying",
            priority="high",
            source_session_id="s1",
            source_project="proj",
        )
        store.add_tip(tip, embedding=[1.0, 0.0, 0.0], embedding_provider="voyage")

        runner = CliRunner()
        with patch("fm.retriever.embed_text") as mock_embed:
            from fm.embeddings import EmbeddingResult

            mock_embed.return_value = EmbeddingResult(
                vector=[0.95, 0.05, 0.0], provider="voyage"
            )
            result = runner.invoke(
                main,
                ["retrieve", "deploy the app", "--db", str(db_path)],
            )

        assert result.exit_code == 0
        assert "prerequisites" in result.output


class TestHookRetrieveCommand:
    def test_reads_json_from_stdin(self, tmp_path: Path) -> None:
        db_path = tmp_path / "tips.db"
        from fm.models import Tip
        from fm.store import TipStore

        store = TipStore(db_path)
        tip = Tip(
            category="strategy",
            content="Always check prerequisites",
            purpose="Prevents failures",
            steps=["Check step 1"],
            trigger="When deploying",
            priority="high",
            source_session_id="s1",
            source_project="proj",
        )
        store.add_tip(tip, embedding=[1.0, 0.0, 0.0], embedding_provider="voyage")

        hook_input = json.dumps({
            "session_id": "s1",
            "prompt": "deploy the app",
            "hook_event_name": "UserPromptSubmit",
            "cwd": "/project",
        })

        runner = CliRunner()
        with patch("fm.retriever.embed_text") as mock_embed:
            from fm.embeddings import EmbeddingResult

            mock_embed.return_value = EmbeddingResult(
                vector=[0.95, 0.05, 0.0], provider="voyage"
            )
            result = runner.invoke(
                main,
                ["hook-retrieve", "--db", str(db_path)],
                input=hook_input,
            )

        assert result.exit_code == 0
        assert "prerequisites" in result.output


class TestTipsCommands:
    def test_tips_list(self, tmp_path: Path) -> None:
        db_path = tmp_path / "tips.db"
        from fm.models import Tip
        from fm.store import TipStore

        store = TipStore(db_path)
        tip = Tip(
            category="strategy",
            content="Test tip content",
            purpose="Testing",
            steps=[],
            trigger="Always",
            priority="low",
            source_session_id="s1",
            source_project="proj",
        )
        store.add_tip(tip)

        runner = CliRunner()
        result = runner.invoke(main, ["tips", "list", "--db", str(db_path)])
        assert result.exit_code == 0
        assert "Test tip content" in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py -v`

Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement the CLI**

```python
# src/fm/cli.py
from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from fm.embeddings import embed_text, get_available_provider
from fm.extractor import extract_tips_from_session
from fm.models import Tip
from fm.parser import parse_session
from fm.retriever import format_tips, retrieve_tips
from fm.store import TipStore

_DEFAULT_DB = Path.home() / ".future_memory" / "tips.db"


@click.group()
def main() -> None:
    """future_memory — trajectory-informed tips for self-improving agents."""
    pass


@main.command()
@click.argument("jsonl_path", type=click.Path(exists=True, path_type=Path))
@click.option("--db", type=click.Path(path_type=Path), default=_DEFAULT_DB)
@click.option("--model", default="sonnet", help="Claude model for extraction.")
def extract(jsonl_path: Path, db: Path, model: str) -> None:
    """Extract tips from a Claude Code session log."""
    store = TipStore(db)

    # Parse session to get the session ID
    session_id, turns = parse_session(jsonl_path, return_session_id=True)
    if not session_id:
        session_id = jsonl_path.stem

    if store.is_session_processed(session_id):
        click.echo(f"Session {session_id} already processed, skipping.")
        return

    if not turns:
        click.echo("No turns found in session.")
        store.mark_session_processed(session_id, str(jsonl_path), tip_count=0)
        return

    # Derive project name from jsonl path
    project = jsonl_path.parent.name

    click.echo(f"Extracting tips from {len(turns)} turns (model: {model})...")
    tips = extract_tips_from_session(
        turns, session_id=session_id, project=project, model=model
    )

    # Embed and store tips
    provider = get_available_provider()
    for tip in tips:
        embedding_result = embed_text(
            f"{tip.content} {tip.trigger}", provider=provider
        )
        if embedding_result:
            store.add_tip(
                tip,
                embedding=embedding_result.vector,
                embedding_provider=embedding_result.provider,
            )
        else:
            store.add_tip(tip)

    store.mark_session_processed(session_id, str(jsonl_path), tip_count=len(tips))
    click.echo(f"Extracted {len(tips)} tips from session {session_id}.")


@main.command("extract-all")
@click.option("--db", type=click.Path(path_type=Path), default=_DEFAULT_DB)
@click.option("--model", default="sonnet", help="Claude model for extraction.")
def extract_all(db: Path, model: str) -> None:
    """Process all unprocessed Claude Code sessions."""
    projects_dir = Path.home() / ".claude" / "projects"
    if not projects_dir.exists():
        click.echo("No Claude Code projects directory found.")
        return

    store = TipStore(db)
    total_tips = 0
    processed = 0

    for project_dir in projects_dir.iterdir():
        if not project_dir.is_dir():
            continue
        for jsonl_file in project_dir.glob("*.jsonl"):
            session_id = jsonl_file.stem
            if store.is_session_processed(session_id):
                continue

            click.echo(f"Processing {project_dir.name}/{jsonl_file.name}...")
            session_id, turns = parse_session(jsonl_file, return_session_id=True)
            if not session_id:
                session_id = jsonl_file.stem

            if not turns:
                store.mark_session_processed(session_id, str(jsonl_file), tip_count=0)
                continue

            project = project_dir.name
            tips = extract_tips_from_session(
                turns, session_id=session_id, project=project, model=model
            )

            provider = get_available_provider()
            for tip in tips:
                embedding_result = embed_text(
                    f"{tip.content} {tip.trigger}", provider=provider
                )
                if embedding_result:
                    store.add_tip(
                        tip,
                        embedding=embedding_result.vector,
                        embedding_provider=embedding_result.provider,
                    )
                else:
                    store.add_tip(tip)

            store.mark_session_processed(
                session_id, str(jsonl_file), tip_count=len(tips)
            )
            total_tips += len(tips)
            processed += 1

    click.echo(f"Processed {processed} sessions, extracted {total_tips} tips total.")


@main.command()
@click.argument("query")
@click.option("--db", type=click.Path(path_type=Path), default=_DEFAULT_DB)
@click.option("--threshold", default=0.6, help="Cosine similarity threshold.")
@click.option("--top-k", default=5, help="Max tips to return.")
def retrieve(query: str, db: Path, threshold: float, top_k: int) -> None:
    """Retrieve relevant tips for a task description."""
    store = TipStore(db)
    tips = retrieve_tips(query, store, threshold=threshold, top_k=top_k)
    output = format_tips(tips)
    if output:
        click.echo(output)
    else:
        click.echo("No matching tips found.")


@main.command("hook-retrieve")
@click.option("--db", type=click.Path(path_type=Path), default=_DEFAULT_DB)
@click.option("--threshold", default=0.6)
@click.option("--top-k", default=5)
def hook_retrieve(db: Path, threshold: float, top_k: int) -> None:
    """Hook entrypoint: reads JSON from stdin, outputs tips to stdout."""
    try:
        raw = sys.stdin.read()
        data = json.loads(raw)
        prompt = data.get("prompt", "")
    except (json.JSONDecodeError, KeyError):
        return  # Silently fail — don't block the session

    if not prompt:
        return

    store = TipStore(db)
    tips = retrieve_tips(prompt, store, threshold=threshold, top_k=top_k)
    output = format_tips(tips)
    if output:
        click.echo(output)


@main.group()
def tips() -> None:
    """Manage stored tips."""
    pass


@tips.command("list")
@click.option("--db", type=click.Path(path_type=Path), default=_DEFAULT_DB)
@click.option("--category", type=click.Choice(["strategy", "recovery", "optimization"]))
def tips_list(db: Path, category: str | None) -> None:
    """List all stored tips."""
    store = TipStore(db)
    all_tips = store.list_tips(category=category)
    if not all_tips:
        click.echo("No tips stored.")
        return
    for tip in all_tips:
        click.echo(f"[{tip.id[:8]}] [{tip.priority}] {tip.category}: {tip.content[:80]}")


@tips.command("show")
@click.argument("tip_id")
@click.option("--db", type=click.Path(path_type=Path), default=_DEFAULT_DB)
def tips_show(tip_id: str, db: Path) -> None:
    """Show details of a specific tip."""
    store = TipStore(db)
    # Support prefix matching
    all_tips = store.list_tips()
    matches = [t for t in all_tips if t.id.startswith(tip_id)]
    if not matches:
        click.echo(f"No tip found matching '{tip_id}'.")
        return
    if len(matches) > 1:
        click.echo(f"Multiple tips match '{tip_id}'. Be more specific.")
        return
    tip = matches[0]
    click.echo(format_tips([tip]))
    click.echo(f"\nID: {tip.id}")
    click.echo(f"Project: {tip.source_project}")
    click.echo(f"Session: {tip.source_session_id}")
    click.echo(f"Created: {tip.created_at}")


main.add_command(tips)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cli.py -v`

Expected: All tests pass.

- [ ] **Step 5: Run the full test suite**

Run: `pytest tests/ -v`

Expected: All tests pass.

- [ ] **Step 6: Verify the CLI is installable and runs**

Run:
```bash
uv pip install -e ".[dev]"
fm --help
fm tips list --help
fm extract --help
fm retrieve --help
```

Expected: Help text displayed for each command.

- [ ] **Step 7: Commit**

```bash
git add src/fm/cli.py tests/test_cli.py
git commit -m "feat: CLI interface — extract, retrieve, hook-retrieve, tips list/show"
```

---

### Task 9: Integration Test with Real Session Data

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write an integration test that exercises the full pipeline**

```python
# tests/test_integration.py
"""Integration test exercising the full pipeline with mocked LLM calls."""
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from click.testing import CliRunner

from fm.cli import main
from fm.embeddings import EmbeddingResult
from fm.store import TipStore


class TestFullPipeline:
    def test_extract_then_retrieve(self, sample_jsonl: Path, tmp_path: Path) -> None:
        """Full loop: parse → extract → store → retrieve."""
        db_path = tmp_path / "tips.db"

        # Mock the claude CLI to return tips
        mock_tips = json.dumps({
            "tips": [
                {
                    "category": "recovery",
                    "content": "When file read fails, check if the path exists before retrying",
                    "purpose": "Prevents repeated failures on missing files",
                    "steps": ["Check file exists", "Read file", "Handle missing file"],
                    "trigger": "When reading files that may not exist",
                    "priority": "high",
                },
                {
                    "category": "strategy",
                    "content": "Always verify login functionality after modifying auth code",
                    "purpose": "Auth changes are high-risk and need immediate verification",
                    "steps": ["Modify auth code", "Run auth tests", "Verify manually"],
                    "trigger": "When modifying authentication code",
                    "priority": "critical",
                },
            ]
        })
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = mock_tips

        # Step 1: Extract
        runner = CliRunner()
        with patch("fm.extractor.subprocess.run", return_value=mock_result):
            with patch("fm.embeddings._embed_voyage") as mock_voyage:
                mock_voyage.side_effect = lambda text: [0.8, 0.1, 0.1] if "auth" in text.lower() else [0.1, 0.8, 0.1]
                with patch.dict("os.environ", {"VOYAGE_API_KEY": "test"}):
                    result = runner.invoke(
                        main,
                        ["extract", str(sample_jsonl), "--db", str(db_path)],
                    )

        assert result.exit_code == 0
        assert "2 tips" in result.output.lower() or "extracted 2" in result.output.lower()

        # Step 2: Verify tips are stored
        store = TipStore(db_path)
        all_tips = store.list_tips()
        assert len(all_tips) == 2

        # Step 3: Retrieve with a related query
        with patch("fm.retriever.embed_text") as mock_embed:
            mock_embed.return_value = EmbeddingResult(
                vector=[0.75, 0.15, 0.1], provider="voyage"
            )
            result = runner.invoke(
                main,
                ["retrieve", "fix the authentication module", "--db", str(db_path)],
            )

        assert result.exit_code == 0
        # Should match the auth-related tip
        assert "auth" in result.output.lower()

    def test_hook_retrieve_integration(self, sample_jsonl: Path, tmp_path: Path) -> None:
        """Test the hook entrypoint reads stdin and returns tips."""
        db_path = tmp_path / "tips.db"

        # Pre-populate the store
        from fm.models import Tip

        store = TipStore(db_path)
        tip = Tip(
            category="strategy",
            content="Run tests after every code change",
            purpose="Catch regressions early",
            steps=["Make change", "Run tests"],
            trigger="After modifying code",
            priority="high",
            source_session_id="s1",
            source_project="proj",
        )
        store.add_tip(tip, embedding=[0.9, 0.1, 0.0], embedding_provider="voyage")

        hook_input = json.dumps({
            "prompt": "I just changed the validation logic",
            "session_id": "s2",
            "hook_event_name": "UserPromptSubmit",
            "cwd": "/project",
        })

        runner = CliRunner()
        with patch("fm.retriever.embed_text") as mock_embed:
            mock_embed.return_value = EmbeddingResult(
                vector=[0.85, 0.15, 0.0], provider="voyage"
            )
            result = runner.invoke(
                main,
                ["hook-retrieve", "--db", str(db_path)],
                input=hook_input,
            )

        assert result.exit_code == 0
        assert "tests" in result.output.lower()

    def test_extract_all_skips_processed(self, sample_jsonl: Path, tmp_path: Path) -> None:
        """extract-all should skip already-processed sessions."""
        db_path = tmp_path / "tips.db"
        store = TipStore(db_path)

        # Pre-mark the session as processed
        store.mark_session_processed("test-session-001", str(sample_jsonl), tip_count=1)

        # Create a fake projects dir structure
        projects_dir = tmp_path / ".claude" / "projects" / "test-project"
        projects_dir.mkdir(parents=True)
        import shutil
        shutil.copy(sample_jsonl, projects_dir / "test-session-001.jsonl")

        runner = CliRunner()
        with patch("fm.cli.Path.home", return_value=tmp_path):
            result = runner.invoke(
                main, ["extract-all", "--db", str(db_path)]
            )

        assert result.exit_code == 0
        assert "0 tips" in result.output.lower() or "processed 0" in result.output.lower()
```

- [ ] **Step 2: Run integration tests**

Run: `pytest tests/test_integration.py -v`

Expected: All tests pass.

- [ ] **Step 3: Run the complete test suite**

Run: `pytest tests/ -v --tb=short`

Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: integration tests for full extract → retrieve pipeline"
```

---

### Task 10: Hook Configuration & Final Wiring

**Files:**
- Modify: `pyproject.toml` (verify entry point)
- Create: `.env.example`

- [ ] **Step 1: Create .env.example**

```bash
# .env.example
# Required: Voyage AI API key for embeddings
VOYAGE_API_KEY=your-voyage-api-key-here

# Optional: HuggingFace API token (free tier works without it, but rate limits are higher with one)
# HF_API_TOKEN=your-huggingface-token-here
```

- [ ] **Step 2: Add .env to .gitignore**

Add `.env` to `.gitignore` (should already be there from scaffolding — verify).

Run: `grep -q "^\.env$" .gitignore && echo "Already present" || echo ".env" >> .gitignore`

- [ ] **Step 3: Verify the CLI installs and runs end-to-end**

Run:
```bash
cd /home/brummerv/future_memory
source .venv/bin/activate
uv pip install -e ".[dev]"
fm --help
fm tips list
```

Expected: Help text displayed, empty tips list (no errors).

- [ ] **Step 4: Document the hook setup**

Add to the project README:

```bash
# Add to the bottom of README.md
cat >> README.md << 'EOF'

## Setup

```bash
# Install
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Set your Voyage AI key
export VOYAGE_API_KEY=your-key-here

# Extract tips from a session
fm extract ~/.claude/projects/<project>/<session>.jsonl

# Extract all unprocessed sessions
fm extract-all

# Retrieve tips manually
fm retrieve "your task description"

# List stored tips
fm tips list
```

### Hook Integration

Add to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "command": "fm hook-retrieve",
        "timeout": 5000
      }
    ]
  }
}
```
EOF
```

- [ ] **Step 5: Run the full test suite one final time**

Run: `pytest tests/ -v --tb=short`

Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add .env.example .gitignore README.md
git commit -m "docs: setup instructions and hook configuration"
```

- [ ] **Step 7: Final commit — tag v0.1.0**

```bash
git tag v0.1.0
```
