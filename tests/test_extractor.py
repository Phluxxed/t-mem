import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from fm.extractor import extract_tips_from_session
from fm.models import Action, Turn

SEGMENTATION_RESPONSE = json.dumps([{
    "subtask_id": "s1",
    "raw_description": "Fix the bug",
    "generalized_description": "Agent fixes a bug in authentication",
    "turn_indices": [0],
}])

INTELLIGENCE_RESPONSE = json.dumps({
    "reasoning_categories": {"analytical": [], "planning": [], "validation": [], "reflection": []},
    "cognitive_patterns": ["error_recognition"],
    "outcome": "recovery",
})

ATTRIBUTION_RESPONSE = json.dumps({
    "root_causes": ["missing null check"],
    "contributing_factors": [],
    "causal_chain": ["1: request fails", "2: null pointer"],
})


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

        turns = [
            Turn(
                user_prompt="Fix the bug",
                thinking=["Let me check"],
                actions=[],
                response_text="Done.",
            )
        ]

        responses = [SEGMENTATION_RESPONSE, INTELLIGENCE_RESPONSE, ATTRIBUTION_RESPONSE, mock_tips_json]
        with patch("fm.llm.subprocess.run") as mock_run:
            mock_run.side_effect = [
                type("R", (), {"returncode": 0, "stdout": r, "stderr": ""})()
                for r in responses
            ]
            tips = extract_tips_from_session(
                turns, session_id="s1", project="proj"
            )

        assert len(tips) == 1
        assert tips[0].category == "strategy"
        assert tips[0].source_session_id == "s1"
        assert tips[0].source_project == "proj"

    def test_returns_empty_on_cli_failure(self) -> None:
        turns = [Turn(user_prompt="Do something")]

        with patch("fm.segmenter.call_claude", return_value=None), \
             patch("fm.intelligence.call_claude", return_value=None):
            tips = extract_tips_from_session(
                turns, session_id="s1", project="proj"
            )

        assert tips == []

    def test_returns_empty_on_invalid_json(self) -> None:
        turns = [Turn(user_prompt="Do something")]

        with patch("fm.segmenter.call_claude", return_value=SEGMENTATION_RESPONSE), \
             patch("fm.intelligence.call_claude", return_value=INTELLIGENCE_RESPONSE), \
             patch("fm.attribution.call_claude", return_value=ATTRIBUTION_RESPONSE), \
             patch("fm.extractor.call_claude", return_value="This is not JSON"):
            tips = extract_tips_from_session(
                turns, session_id="s1", project="proj"
            )

        assert tips == []

    def test_skips_empty_turns(self) -> None:
        tips = extract_tips_from_session([], session_id="s1", project="proj")
        assert tips == []
