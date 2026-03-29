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
