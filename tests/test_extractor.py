import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from fm.extractor import extract_tips_from_session
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
        assert len(prompt) < 100000


class TestExtractTipsFromSession:
    def test_calls_claude_cli_and_parses_response(self, tmp_path: Path) -> None:
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
        tips = extract_tips_from_session([], session_id="s1", project="proj")
        assert tips == []
