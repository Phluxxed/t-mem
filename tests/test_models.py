import pytest

from fm.models import Action, Tip, Turn, Subtask, SubtaskIntelligence, SubtaskAttribution


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


def test_subtask_fields() -> None:
    turn = Turn(user_prompt="hello")
    s = Subtask(
        id="s1",
        session_id="sess-abc",
        raw_description="User fixed SSL cert issue in future_memory",
        generalized_description="Agent resolves SSL certificate verification failure in HTTP client",
        turns=[turn],
    )
    assert s.generalized_description == "Agent resolves SSL certificate verification failure in HTTP client"
    assert len(s.turns) == 1


def test_subtask_intelligence_fields() -> None:
    si = SubtaskIntelligence(
        reasoning_categories={"analytical": ["assessed situation"], "planning": [], "validation": [], "reflection": []},
        cognitive_patterns=["error_recognition", "self_correction"],
        outcome="recovery",
    )
    assert si.outcome == "recovery"
    assert "error_recognition" in si.cognitive_patterns


def test_subtask_attribution_fields() -> None:
    sa = SubtaskAttribution(
        root_causes=["SSL cert not in trust store"],
        contributing_factors=["corporate proxy intercepts traffic"],
        causal_chain=["step 1: request made", "step 2: SSL handshake fails"],
    )
    assert len(sa.root_causes) == 1


def test_tip_subtask_fields() -> None:
    tip = Tip(
        category="recovery",
        content="When SSL verification fails, check corporate proxy cert",
        purpose="Prevents repeated SSL failures",
        steps=["step 1"],
        trigger="SSL error on HTTPS requests",
        priority="high",
        source_session_id="sess-abc",
        source_project="future_memory",
        subtask_id="s1",
        subtask_description="Agent resolves SSL certificate verification failure in HTTP client",
    )
    assert tip.subtask_id == "s1"
    assert tip.subtask_description is not None
