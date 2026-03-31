import json
from unittest.mock import patch
from fm.models import Turn, Subtask
from fm.intelligence import extract_intelligence, _parse_intelligence

VALID_INTELLIGENCE_JSON = json.dumps({
    "reasoning_categories": {
        "analytical": ["assessed the SSL error context"],
        "planning": ["decided to bypass SSL verification"],
        "validation": ["confirmed request succeeded after bypass"],
        "reflection": ["reconsidered using verify=False permanently"],
    },
    "cognitive_patterns": ["error_recognition", "self_correction"],
    "outcome": "recovery",
})

def _make_subtask(outcome_hint: str = "recovery") -> Subtask:
    return Subtask(
        id="s1",
        session_id="sess-1",
        raw_description="Fixed SSL error",
        generalized_description="Agent debugs SSL certificate verification failure",
        turns=[Turn(user_prompt="fix the SSL error", response_text=outcome_hint)],
    )

def test_parse_intelligence_valid():
    si = _parse_intelligence(VALID_INTELLIGENCE_JSON)
    assert si is not None
    assert si.outcome == "recovery"
    assert "error_recognition" in si.cognitive_patterns
    assert "analytical" in si.reasoning_categories

def test_parse_intelligence_with_fence():
    wrapped = f"```json\n{VALID_INTELLIGENCE_JSON}\n```"
    si = _parse_intelligence(wrapped)
    assert si is not None
    assert si.outcome == "recovery"

def test_parse_intelligence_invalid_returns_none():
    si = _parse_intelligence("not json")
    assert si is None

def test_extract_intelligence_calls_claude():
    subtask = _make_subtask()
    with patch("fm.intelligence.call_claude", return_value=VALID_INTELLIGENCE_JSON):
        si = extract_intelligence(subtask, model="sonnet")
    assert si is not None
    assert si.outcome == "recovery"

def test_extract_intelligence_returns_none_on_failure():
    subtask = _make_subtask()
    with patch("fm.intelligence.call_claude", return_value=None):
        si = extract_intelligence(subtask, model="sonnet")
    assert si is None
