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
        assert len(prompt) < 100000
