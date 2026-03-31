import json
from unittest.mock import patch
from fm.models import Turn, Subtask, SubtaskIntelligence
from fm.attribution import extract_attribution, _parse_attribution

VALID_ATTRIBUTION_JSON = json.dumps({
    "root_causes": ["SSL cert from corporate proxy not in Python trust store"],
    "contributing_factors": ["Zscaler intercepts HTTPS traffic and re-signs with its own cert"],
    "causal_chain": [
        "1: requests.post() initiates TLS handshake",
        "2: Zscaler presents its intermediate cert",
        "3: Python ssl module rejects cert not in certifi bundle",
        "4: SSLError raised, request fails",
    ],
})

def _make_subtask_with_intelligence() -> tuple[Subtask, SubtaskIntelligence]:
    subtask = Subtask(
        id="s1",
        session_id="sess-1",
        raw_description="Fixed SSL cert error",
        generalized_description="Agent debugs SSL certificate verification failure",
        turns=[Turn(user_prompt="fix SSL error")],
    )
    intelligence = SubtaskIntelligence(
        reasoning_categories={"analytical": [], "planning": [], "validation": [], "reflection": []},
        cognitive_patterns=["error_recognition"],
        outcome="recovery",
    )
    return subtask, intelligence

def test_parse_attribution_valid():
    sa = _parse_attribution(VALID_ATTRIBUTION_JSON)
    assert sa is not None
    assert len(sa.root_causes) == 1
    assert len(sa.causal_chain) == 4

def test_parse_attribution_with_fence():
    wrapped = f"```json\n{VALID_ATTRIBUTION_JSON}\n```"
    sa = _parse_attribution(wrapped)
    assert sa is not None

def test_parse_attribution_invalid_returns_none():
    sa = _parse_attribution("garbage")
    assert sa is None

def test_extract_attribution_calls_claude():
    subtask, intelligence = _make_subtask_with_intelligence()
    with patch("fm.attribution.call_claude", return_value=VALID_ATTRIBUTION_JSON):
        sa = extract_attribution(subtask, intelligence, model="sonnet")
    assert sa is not None
    assert "SSL cert" in sa.root_causes[0]

def test_extract_attribution_returns_none_on_failure():
    subtask, intelligence = _make_subtask_with_intelligence()
    with patch("fm.attribution.call_claude", return_value=None):
        sa = extract_attribution(subtask, intelligence, model="sonnet")
    assert sa is None
