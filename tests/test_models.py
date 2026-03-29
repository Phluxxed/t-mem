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
