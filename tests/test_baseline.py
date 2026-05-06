"""Tests for baseline metric computation."""
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from fm.baseline import compute_baseline
from fm.models import Action, Turn


def _make_jsonl(path: Path, turns_data: list[dict]) -> None:
    """Write a minimal JSONL session file with given turn data."""
    import uuid
    lines = []
    prev_uuid = None
    for td in turns_data:
        entry_uuid = str(uuid.uuid4())
        lines.append(json.dumps({
            "uuid": entry_uuid,
            "parentUuid": prev_uuid,
            "type": "user",
            "message": {"role": "user", "content": td["prompt"]},
            "sessionId": "test-session",
            "timestamp": "2026-01-01T00:00:00Z",
        }))
        prev_uuid = entry_uuid
    path.write_text("\n".join(lines))


class TestComputeBaseline:
    def test_basic_metrics(self, tmp_path: Path) -> None:
        sessions_dir = tmp_path / "projects" / "proj1"
        sessions_dir.mkdir(parents=True)

        # Pre-cutoff timestamps: turn-level filter requires Turn.timestamp to
        # be set; cutoff in this test is 2026-03-29.
        turns = [
            Turn(
                user_prompt="do a thing",
                timestamp="2026-01-01T00:00:00Z",
                actions=[
                    Action(tool_name="Bash", tool_input={}, success=True),
                    Action(tool_name="Read", tool_input={}, success=False),
                    Action(tool_name="Read", tool_input={}, success=True),
                ],
            ),
            Turn(
                user_prompt="do another thing",
                timestamp="2026-01-01T00:01:00Z",
                actions=[
                    Action(tool_name="Bash", tool_input={}, success=True),
                ],
            ),
        ]

        # Create a fake JSONL file with old mtime
        session_file = sessions_dir / "session1.jsonl"
        session_file.write_text("{}")

        import os
        old_mtime = datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
        os.utime(session_file, (old_mtime, old_mtime))

        cutoff = datetime(2026, 3, 29, tzinfo=timezone.utc)

        with patch("fm.baseline.parse_session", return_value=turns):
            result = compute_baseline(before=cutoff, sample=0, projects_dir=sessions_dir.parent)

        assert result["sessions_analyzed"] == 1
        m = result["metrics"]
        # 1 failed out of 4 total actions
        assert m["error_rate"] == 0.25
        # 1 error sequence (Read failed then succeeded), 1 recovery
        assert m["recovery_rate"] == 1.0
        assert m["avg_retries_after_error"] == 1.0
        assert m["avg_actions_per_turn"] == 2.0
        assert m["avg_turns_per_session"] == 2.0

    def test_no_sessions_raises(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "projects"
        empty_dir.mkdir()
        cutoff = datetime(2020, 1, 1, tzinfo=timezone.utc)  # nothing before this

        import pytest
        with pytest.raises(ValueError, match="No valid sessions"):
            compute_baseline(before=cutoff, sample=0, projects_dir=empty_dir)

    def test_output_schema(self, tmp_path: Path) -> None:
        sessions_dir = tmp_path / "projects" / "proj1"
        sessions_dir.mkdir(parents=True)

        session_file = sessions_dir / "s.jsonl"
        session_file.write_text("{}")

        import os
        old_mtime = datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
        os.utime(session_file, (old_mtime, old_mtime))

        turns = [Turn(
            user_prompt="x",
            timestamp="2026-01-01T00:00:00Z",
            actions=[Action(tool_name="Bash", tool_input={}, success=True)],
        )]
        cutoff = datetime(2026, 3, 29, tzinfo=timezone.utc)

        with patch("fm.baseline.parse_session", return_value=turns):
            result = compute_baseline(before=cutoff, sample=0, projects_dir=sessions_dir.parent)

        assert "computed_at" in result
        assert "cutoff_date" in result
        assert "sessions_analyzed" in result
        assert "sessions_available" in result
        required_metrics = {
            "error_rate", "recovery_rate", "avg_retries_after_error",
            "avg_actions_per_turn", "avg_turns_per_session",
            "session_length_p50", "session_length_p90", "repeated_op_rate",
        }
        assert required_metrics == set(result["metrics"].keys())
