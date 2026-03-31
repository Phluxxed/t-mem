from fm.models import Turn, Action
from fm.prompts.segment import build_segmentation_prompt


def test_segmentation_prompt_contains_turns():
    turns = [
        Turn(user_prompt="Set up a new Python project", response_text="Done, created venv"),
        Turn(user_prompt="Fix the SSL error", response_text="Added cert to bundle"),
    ]
    prompt = build_segmentation_prompt(turns)
    assert "Set up a new Python project" in prompt
    assert "Fix the SSL error" in prompt
    assert "generalized_description" in prompt
    assert "turn_indices" in prompt
