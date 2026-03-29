import json
from pathlib import Path

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
        for turn in turns:
            for action in turn.actions:
                assert action.tool_name not in ("turn_duration", "stop_hook_summary")

    def test_drops_file_history_snapshots(self, sample_jsonl: Path) -> None:
        turns = parse_session(sample_jsonl)
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
