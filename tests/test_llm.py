from unittest.mock import patch, MagicMock
from fm.llm import call_claude

def test_call_claude_returns_stdout():
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "extracted output"
    mock_result.stderr = ""
    with patch("fm.llm.subprocess.run", return_value=mock_result):
        result = call_claude("some prompt", model="sonnet")
    assert result == "extracted output"

def test_call_claude_returns_none_on_nonzero():
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_result.stderr = "error"
    with patch("fm.llm.subprocess.run", return_value=mock_result):
        result = call_claude("prompt", model="sonnet")
    assert result is None

def test_call_claude_returns_none_on_timeout():
    import subprocess
    with patch("fm.llm.subprocess.run", side_effect=subprocess.TimeoutExpired("claude", 120)):
        result = call_claude("prompt", model="sonnet")
    assert result is None
