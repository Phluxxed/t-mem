import json
from unittest.mock import patch
from fm.models import Turn, Tip
from fm.extractor import extract_tips_from_session

SEGMENTATION_RESPONSE = json.dumps([{
    "subtask_id": "s1",
    "raw_description": "User fixed SSL error",
    "generalized_description": "Agent debugs SSL certificate verification failure",
    "turn_indices": [0],
}])

INTELLIGENCE_RESPONSE = json.dumps({
    "reasoning_categories": {"analytical": [], "planning": [], "validation": [], "reflection": []},
    "cognitive_patterns": ["error_recognition"],
    "outcome": "recovery",
})

ATTRIBUTION_RESPONSE = json.dumps({
    "root_causes": ["SSL cert not trusted"],
    "contributing_factors": [],
    "causal_chain": ["1: request fails", "2: SSL error"],
})

TIPS_RESPONSE = json.dumps({
    "tips": [{
        "category": "recovery",
        "content": "When SSL verification fails on HTTPS requests, check if corporate proxy cert is in trust store",
        "purpose": "Corporate proxies re-sign TLS traffic causing verification failures",
        "steps": ["Check REQUESTS_CA_BUNDLE env var", "Add corp cert to bundle"],
        "trigger": "SSLError on HTTPS requests in corporate network",
        "negative_example": None,
        "priority": "high",
        "task_context": None,
    }]
})

def test_full_pipeline_produces_tips():
    turns = [Turn(user_prompt="fix SSL error", response_text="fixed by adding cert")]
    responses = [SEGMENTATION_RESPONSE, INTELLIGENCE_RESPONSE, ATTRIBUTION_RESPONSE, TIPS_RESPONSE]
    with patch("fm.llm.subprocess.run") as mock_run:
        mock_run.side_effect = [
            type("R", (), {"returncode": 0, "stdout": r, "stderr": ""})()
            for r in responses
        ]
        tips = extract_tips_from_session(
            turns, session_id="sess-1", project="test-project", model="sonnet"
        )
    assert len(tips) == 1
    assert tips[0].category == "recovery"
    assert tips[0].subtask_id == "s1"
    assert tips[0].subtask_description == "Agent debugs SSL certificate verification failure"
    assert tips[0].source_session_id == "sess-1"

def test_pipeline_handles_segmentation_failure():
    turns = [Turn(user_prompt="do something")]
    with patch("fm.segmenter.call_claude", return_value=None), \
         patch("fm.intelligence.call_claude", return_value=INTELLIGENCE_RESPONSE), \
         patch("fm.attribution.call_claude", return_value=ATTRIBUTION_RESPONSE), \
         patch("fm.extractor.call_claude", return_value=TIPS_RESPONSE):
        tips = extract_tips_from_session(
            turns, session_id="sess-1", project="test-project", model="sonnet"
        )
    assert isinstance(tips, list)

def test_pipeline_skips_subtask_when_intelligence_fails():
    turns = [Turn(user_prompt="do something")]
    with patch("fm.segmenter.call_claude", return_value=SEGMENTATION_RESPONSE), \
         patch("fm.intelligence.call_claude", return_value=None):
        tips = extract_tips_from_session(
            turns, session_id="sess-1", project="test-project", model="sonnet"
        )
    assert tips == []
