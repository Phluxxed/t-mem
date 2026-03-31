import json
from unittest.mock import patch
from fm.models import Turn
from fm.segmenter import segment_session, _parse_segmentation

VALID_SEGMENTATION_JSON = json.dumps([
    {
        "subtask_id": "s1",
        "raw_description": "User set up Python project",
        "generalized_description": "Agent configures Python project environment",
        "turn_indices": [0, 1],
    },
    {
        "subtask_id": "s2",
        "raw_description": "User fixed SSL error",
        "generalized_description": "Agent debugs SSL certificate verification failure",
        "turn_indices": [2],
    },
])

def _make_turns(n: int) -> list[Turn]:
    return [Turn(user_prompt=f"turn {i}") for i in range(n)]

def test_parse_segmentation_valid():
    turns = _make_turns(3)
    subtasks = _parse_segmentation(VALID_SEGMENTATION_JSON, turns, session_id="sess-1")
    assert len(subtasks) == 2
    assert subtasks[0].id == "s1"
    assert subtasks[0].session_id == "sess-1"
    assert subtasks[0].generalized_description == "Agent configures Python project environment"
    assert len(subtasks[0].turns) == 2
    assert len(subtasks[1].turns) == 1

def test_parse_segmentation_with_markdown_fence():
    turns = _make_turns(3)
    wrapped = f"```json\n{VALID_SEGMENTATION_JSON}\n```"
    subtasks = _parse_segmentation(wrapped, turns, session_id="sess-1")
    assert len(subtasks) == 2

def test_parse_segmentation_invalid_json_returns_single_subtask():
    turns = _make_turns(3)
    subtasks = _parse_segmentation("not json at all", turns, session_id="sess-1")
    assert len(subtasks) == 1
    assert subtasks[0].id == "s1"
    assert len(subtasks[0].turns) == 3

def test_segment_session_calls_claude():
    turns = _make_turns(3)
    with patch("fm.segmenter.call_claude", return_value=VALID_SEGMENTATION_JSON):
        subtasks = segment_session(turns, session_id="sess-1", model="sonnet")
    assert len(subtasks) == 2

def test_segment_session_falls_back_on_claude_failure():
    turns = _make_turns(3)
    with patch("fm.segmenter.call_claude", return_value=None):
        subtasks = segment_session(turns, session_id="sess-1", model="sonnet")
    assert len(subtasks) == 1
    assert len(subtasks[0].turns) == 3
