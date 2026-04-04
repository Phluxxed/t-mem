# Extraction Pipeline Rebuild — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the extraction pipeline to match the IBM Research paper (arXiv:2603.10600) — subtask-level segmentation, three-stage extraction per subtask, and description generalisation — replacing the current single-pass session-level approach.

**Architecture:** Sessions are first segmented into logical subtasks with generalised descriptions (Phase A, 1 LLM call per session). Each subtask then passes through three sequential LLM stages: Trajectory Intelligence Extractor → Decision Attribution Analyzer → Contextual Learning Generator (3 LLM calls per subtask). The generalised subtask description (not the tip content) becomes the embedding key for retrieval, enabling cross-project transfer.

**Tech Stack:** Python 3.12+, `claude` CLI via subprocess, SQLite, Voyage AI embeddings, Click CLI.

---

## File Map

| File | Status | Responsibility |
|------|--------|----------------|
| `src/fm/models.py` | Modify | Add `Subtask`, `SubtaskIntelligence`, `SubtaskAttribution` dataclasses; add `subtask_id`/`subtask_description` to `Tip` |
| `src/fm/llm.py` | **New** | Shared `call_claude(prompt, model) -> str` utility (extracted from extractor.py) |
| `src/fm/segmenter.py` | **New** | `segment_session(turns, session_id) -> list[Subtask]` — Phase A |
| `src/fm/prompts/segment.py` | **New** | Segmentation + generalisation prompt |
| `src/fm/intelligence.py` | **New** | `extract_intelligence(subtask) -> SubtaskIntelligence` |
| `src/fm/prompts/intelligence.py` | **New** | Trajectory Intelligence Extractor prompt |
| `src/fm/attribution.py` | **New** | `extract_attribution(subtask, intelligence) -> SubtaskAttribution` |
| `src/fm/prompts/attribution.py` | **New** | Decision Attribution Analyzer prompt |
| `src/fm/prompts/tips_gen.py` | **New** | Contextual Learning Generator prompt (replaces `prompts/extract.py`) |
| `src/fm/extractor.py` | Rewrite | Orchestrate full pipeline: segment → intelligence → attribution → tips |
| `src/fm/store.py` | Modify | Add `subtask_id`/`subtask_description` columns; `migrate_add_subtask_columns()`; update `add_tip` |
| `src/fm/cli.py` | Modify | Update embed key to use `subtask_description`; add `fm db migrate` command |
| `tests/test_segmenter.py` | **New** | Unit tests for segmentation parsing |
| `tests/test_intelligence.py` | **New** | Unit tests for intelligence extraction parsing |
| `tests/test_attribution.py` | **New** | Unit tests for attribution parsing |
| `tests/test_extractor_pipeline.py` | **New** | Integration test of full pipeline with mocked LLM calls |

> `src/fm/prompts/extract.py` is replaced by `prompts/tips_gen.py`. Delete it after Task 7 passes tests.

---

## Task 1: Add new dataclasses to models.py

**Files:**
- Modify: `src/fm/models.py`
- Test: `tests/test_models.py` (create if absent)

- [ ] **Step 1: Write failing test**

```python
# tests/test_models.py
from fm.models import Subtask, SubtaskIntelligence, SubtaskAttribution, Tip, Turn

def test_subtask_fields():
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

def test_subtask_intelligence_fields():
    si = SubtaskIntelligence(
        reasoning_categories={"analytical": ["assessed situation"], "planning": [], "validation": [], "reflection": []},
        cognitive_patterns=["error_recognition", "self_correction"],
        outcome="recovery",
    )
    assert si.outcome == "recovery"
    assert "error_recognition" in si.cognitive_patterns

def test_subtask_attribution_fields():
    sa = SubtaskAttribution(
        root_causes=["SSL cert not in trust store"],
        contributing_factors=["corporate proxy intercepts traffic"],
        causal_chain=["step 1: request made", "step 2: SSL handshake fails"],
    )
    assert len(sa.root_causes) == 1

def test_tip_subtask_fields():
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/brummerv/future_memory && .venv/bin/pytest tests/test_models.py -v
```

Expected: `ImportError` or `TypeError` — `Subtask`, `SubtaskIntelligence`, `SubtaskAttribution` not defined; `Tip` missing `subtask_id`/`subtask_description`.

- [ ] **Step 3: Add dataclasses and update Tip in models.py**

Append after the `Turn` dataclass and before `Tip`:

```python
@dataclass
class Subtask:
    id: str
    session_id: str
    raw_description: str
    generalized_description: str
    turns: list[Turn]


@dataclass
class SubtaskIntelligence:
    reasoning_categories: dict[str, list[str]]  # analytical/planning/validation/reflection
    cognitive_patterns: list[str]
    outcome: str  # "clean_success" | "inefficient_success" | "recovery" | "failure"


@dataclass
class SubtaskAttribution:
    root_causes: list[str]
    contributing_factors: list[str]
    causal_chain: list[str]
```

Add to `Tip` dataclass (after `task_context`):

```python
    subtask_id: str | None = None
    subtask_description: str | None = None
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/brummerv/future_memory && .venv/bin/pytest tests/test_models.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/brummerv/future_memory && git add src/fm/models.py tests/test_models.py && git commit -m "feat: add Subtask, SubtaskIntelligence, SubtaskAttribution dataclasses; add subtask fields to Tip"
```

---

## Task 2: Extract shared LLM call utility

**Files:**
- Create: `src/fm/llm.py`
- Test: `tests/test_llm.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_llm.py
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/brummerv/future_memory && .venv/bin/pytest tests/test_llm.py -v
```

Expected: `ModuleNotFoundError: No module named 'fm.llm'`

- [ ] **Step 3: Create llm.py**

```python
from __future__ import annotations

import subprocess
import sys


def call_claude(prompt: str, *, model: str = "sonnet", timeout: int = 120) -> str | None:
    """Call the claude CLI with a prompt, return stdout or None on failure."""
    try:
        result = subprocess.run(
            ["claude", "--print", "--model", model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None

    if result.returncode != 0:
        print(f"Warning: claude CLI returned {result.returncode}: {result.stderr[:200]}", file=sys.stderr)
        return None

    if not result.stdout.strip():
        print("Warning: claude CLI returned empty output", file=sys.stderr)
        return None

    return result.stdout
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/brummerv/future_memory && .venv/bin/pytest tests/test_llm.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/brummerv/future_memory && git add src/fm/llm.py tests/test_llm.py && git commit -m "feat: extract shared call_claude utility into fm.llm"
```

---

## Task 3: Write segmentation prompt

**Files:**
- Create: `src/fm/prompts/segment.py`
- Test: `tests/test_prompts.py` (smoke test — prompt is a string, contains key instructions)

- [ ] **Step 1: Write failing test**

```python
# tests/test_prompts.py
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/brummerv/future_memory && .venv/bin/pytest tests/test_prompts.py::test_segmentation_prompt_contains_turns -v
```

Expected: `ModuleNotFoundError: No module named 'fm.prompts.segment'`

- [ ] **Step 3: Create prompts/segment.py**

```python
from __future__ import annotations

import json
from fm.models import Turn

_MAX_CHARS = 1500


def _format_turn(turn: Turn, index: int) -> str:
    parts = [f"[Turn {index}] User: {turn.user_prompt[:300]}"]
    if turn.actions:
        tools = [a.tool_name for a in turn.actions]
        successes = [a.success for a in turn.actions]
        parts.append(f"  Tools used: {', '.join(tools)}")
        parts.append(f"  Outcomes: {successes}")
    if turn.response_text:
        parts.append(f"  Response: {turn.response_text[:300]}")
    return "\n".join(parts)


def build_segmentation_prompt(turns: list[Turn]) -> str:
    formatted = "\n\n".join(_format_turn(t, i) for i, t in enumerate(turns))

    return f"""\
You are segmenting a Claude Code session into logical subtasks.

A subtask is a self-contained unit of work with a clear goal and outcome. Typical subtask types:
- Project/environment setup
- Implementing a specific feature
- Debugging or fixing a specific error
- Refactoring or reorganising code
- Running and fixing tests
- Configuring a tool or integration
- Answering a technical question

## Instructions

Given the session turns below, identify the logical subtask boundaries and produce a JSON array.

For each subtask, provide:
1. `subtask_id`: sequential string — "s1", "s2", etc.
2. `raw_description`: what actually happened in this session (may include specific file names, project names, errors)
3. `generalized_description`: abstract version that transfers to other contexts. Apply these transformations:
   - **Entity abstraction**: replace project names, file names, URLs, usernames, error IDs with generic placeholders
     - "future_memory" → "the target project"
     - "tips.db" → "the database file"
     - "voyage-4-lite" → "the embedding model"
   - **Action normalization**: use canonical verbs (configure, implement, debug, verify, refactor)
   - **Context removal**: strip task-specific qualifiers, keep the core operation
   - Examples:
     - "Fix SSL cert error for Voyage API calls in future_memory" → "Agent debugs SSL certificate verification failure in HTTP client requests"
     - "Add click CLI command fm extract-all" → "Agent implements a CLI command to batch-process items"
4. `turn_indices`: list of 0-based turn indices belonging to this subtask. Every turn must appear in exactly one subtask.

## Output Format

Return ONLY a JSON array — no other text:
[
  {{
    "subtask_id": "s1",
    "raw_description": "...",
    "generalized_description": "...",
    "turn_indices": [0, 1, 2]
  }}
]

## Session Turns

{formatted}
"""
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/brummerv/future_memory && .venv/bin/pytest tests/test_prompts.py::test_segmentation_prompt_contains_turns -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/brummerv/future_memory && git add src/fm/prompts/segment.py tests/test_prompts.py && git commit -m "feat: add subtask segmentation prompt"
```

---

## Task 4: Write segmenter.py

**Files:**
- Create: `src/fm/segmenter.py`
- Test: `tests/test_segmenter.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_segmenter.py
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/brummerv/future_memory && .venv/bin/pytest tests/test_segmenter.py -v
```

Expected: `ModuleNotFoundError: No module named 'fm.segmenter'`

- [ ] **Step 3: Create segmenter.py**

```python
from __future__ import annotations

import json
import re

from fm.llm import call_claude
from fm.models import Subtask, Turn
from fm.prompts.segment import build_segmentation_prompt


def _parse_segmentation(raw: str, turns: list[Turn], *, session_id: str) -> list[Subtask]:
    """Parse the LLM segmentation response into Subtask objects. Falls back to single subtask on failure."""
    # Strip markdown fences if present
    json_match = re.search(r"\[[\s\S]*\]", raw)
    if not json_match:
        return [_whole_session_subtask(turns, session_id)]

    try:
        items = json.loads(json_match.group())
    except json.JSONDecodeError:
        return [_whole_session_subtask(turns, session_id)]

    if not isinstance(items, list) or not items:
        return [_whole_session_subtask(turns, session_id)]

    subtasks = []
    for item in items:
        try:
            indices = item["turn_indices"]
            subtask_turns = [turns[i] for i in indices if 0 <= i < len(turns)]
            subtasks.append(
                Subtask(
                    id=item["subtask_id"],
                    session_id=session_id,
                    raw_description=item["raw_description"],
                    generalized_description=item["generalized_description"],
                    turns=subtask_turns,
                )
            )
        except (KeyError, TypeError, IndexError):
            continue

    if not subtasks:
        return [_whole_session_subtask(turns, session_id)]

    return subtasks


def _whole_session_subtask(turns: list[Turn], session_id: str) -> Subtask:
    """Fallback: treat the entire session as a single subtask."""
    return Subtask(
        id="s1",
        session_id=session_id,
        raw_description="Full session (segmentation failed or skipped)",
        generalized_description="Agent completes a multi-step task",
        turns=turns,
    )


def segment_session(
    turns: list[Turn],
    *,
    session_id: str,
    model: str = "sonnet",
) -> list[Subtask]:
    """Segment a session's turns into logical subtasks with generalised descriptions."""
    if not turns:
        return []

    prompt = build_segmentation_prompt(turns)
    raw = call_claude(prompt, model=model)

    if raw is None:
        return [_whole_session_subtask(turns, session_id)]

    return _parse_segmentation(raw, turns, session_id=session_id)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/brummerv/future_memory && .venv/bin/pytest tests/test_segmenter.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/brummerv/future_memory && git add src/fm/segmenter.py tests/test_segmenter.py && git commit -m "feat: add subtask segmenter with fallback to whole-session"
```

---

## Task 5: Trajectory Intelligence Extractor

**Files:**
- Create: `src/fm/prompts/intelligence.py`
- Create: `src/fm/intelligence.py`
- Test: `tests/test_intelligence.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_intelligence.py
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/brummerv/future_memory && .venv/bin/pytest tests/test_intelligence.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create prompts/intelligence.py**

```python
from __future__ import annotations

import json
from fm.models import Subtask, Turn

_MAX_CHARS = 1500


def _format_turn(turn: Turn, index: int) -> str:
    parts = [f"[Turn {index}] User: {turn.user_prompt[:400]}"]
    if turn.thinking:
        parts.append("  Thinking:")
        for t in turn.thinking[:3]:
            parts.append(f"    - {t[:300]}")
    if turn.actions:
        for a in turn.actions:
            parts.append(f"  Tool: {a.tool_name} (success={a.success})")
            if a.result_stderr:
                parts.append(f"    Error: {a.result_stderr[:200]}")
    if turn.response_text:
        parts.append(f"  Response: {turn.response_text[:400]}")
    return "\n".join(parts)


def build_intelligence_prompt(subtask: Subtask) -> str:
    formatted = "\n\n".join(_format_turn(t, i) for i, t in enumerate(subtask.turns))

    return f"""\
You are performing Trajectory Intelligence Extraction on a subtask from a Claude Code session.

## Subtask
{subtask.generalized_description}

## Task
Analyze the turns below and produce a structured intelligence report.

### 1. Reasoning Categories
Classify each piece of agent thinking/reasoning into one of four cognitive types:
- **analytical**: assessing the situation, reading context, understanding current state
- **planning**: deciding what actions to take, sequencing steps
- **validation**: checking assumptions, verifying results match expectations
- **reflection**: reconsidering approach, recognizing a mistake, changing strategy

### 2. Cognitive Patterns
Identify which of these behavioral patterns are present (list only those that apply):
- `validation_pattern`: agent checks results before proceeding to next step
- `reflection_pattern`: agent reconsiders and changes approach mid-task
- `self_correction`: agent recognizes and fixes its own mistake
- `error_recognition`: agent detects an error (tool failure, unexpected output, wrong assumption)
- `api_discovery`: agent discovers how an API or tool works through trial and exploration
- `efficiency_awareness`: agent recognizes a more efficient approach exists

### 3. Outcome
Classify the subtask outcome as exactly one of:
- `clean_success`: task completed directly without any failures or unnecessary steps
- `inefficient_success`: task completed but with unnecessary steps, repeated operations, or a slower path
- `recovery`: task initially failed but agent recognized the failure and successfully recovered
- `failure`: task did not complete successfully

## Output Format

Return ONLY a JSON object — no other text:
{{
  "reasoning_categories": {{
    "analytical": ["thought 1", ...],
    "planning": ["thought 1", ...],
    "validation": ["thought 1", ...],
    "reflection": ["thought 1", ...]
  }},
  "cognitive_patterns": ["pattern1", "pattern2"],
  "outcome": "recovery"
}}

## Turns

{formatted}
"""
```

- [ ] **Step 4: Create intelligence.py**

```python
from __future__ import annotations

import json
import re

from fm.llm import call_claude
from fm.models import Subtask, SubtaskIntelligence
from fm.prompts.intelligence import build_intelligence_prompt


def _parse_intelligence(raw: str) -> SubtaskIntelligence | None:
    json_match = re.search(r"\{[\s\S]*\}", raw)
    if not json_match:
        return None
    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return None

    try:
        return SubtaskIntelligence(
            reasoning_categories=data.get("reasoning_categories", {
                "analytical": [], "planning": [], "validation": [], "reflection": []
            }),
            cognitive_patterns=data.get("cognitive_patterns", []),
            outcome=data.get("outcome", "clean_success"),
        )
    except (KeyError, TypeError):
        return None


def extract_intelligence(
    subtask: Subtask,
    *,
    model: str = "sonnet",
) -> SubtaskIntelligence | None:
    """Run the Trajectory Intelligence Extractor on a subtask."""
    prompt = build_intelligence_prompt(subtask)
    raw = call_claude(prompt, model=model)
    if raw is None:
        return None
    return _parse_intelligence(raw)
```

- [ ] **Step 5: Run test to verify it passes**

```bash
cd /home/brummerv/future_memory && .venv/bin/pytest tests/test_intelligence.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /home/brummerv/future_memory && git add src/fm/prompts/intelligence.py src/fm/intelligence.py tests/test_intelligence.py && git commit -m "feat: add Trajectory Intelligence Extractor"
```

---

## Task 6: Decision Attribution Analyzer

**Files:**
- Create: `src/fm/prompts/attribution.py`
- Create: `src/fm/attribution.py`
- Test: `tests/test_attribution.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_attribution.py
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/brummerv/future_memory && .venv/bin/pytest tests/test_attribution.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create prompts/attribution.py**

```python
from __future__ import annotations

from fm.models import Subtask, SubtaskIntelligence


def build_attribution_prompt(subtask: Subtask, intelligence: SubtaskIntelligence) -> str:
    turns_summary = "\n".join(
        f"[Turn {i}] {t.user_prompt[:200]}"
        + (f" → tools: {[a.tool_name for a in t.actions]}" if t.actions else "")
        + (f" → errors: {[a.result_stderr[:100] for a in t.actions if a.result_stderr]}" if any(a.result_stderr for a in t.actions) else "")
        for i, t in enumerate(subtask.turns)
    )

    patterns_str = ", ".join(intelligence.cognitive_patterns) if intelligence.cognitive_patterns else "none"
    reflection_thoughts = intelligence.reasoning_categories.get("reflection", [])
    reflection_str = "\n".join(f"  - {t}" for t in reflection_thoughts[:5]) if reflection_thoughts else "  (none)"

    return f"""\
You are performing Decision Attribution Analysis on a subtask from a Claude Code session.

## Subtask
{subtask.generalized_description}

## Outcome
{intelligence.outcome}

## Cognitive Patterns Observed
{patterns_str}

## Agent Reflection Thoughts
{reflection_str}

## Turn Summary
{turns_summary}

## Task
Trace backward through the agent's decisions to identify the causal chain that led to this outcome.

For failures and recoveries: identify what went wrong and why.
For clean successes: identify what decisions were critical to the success.
For inefficient successes: identify what choices caused unnecessary work.

Return ONLY a JSON object — no other text:
{{
  "root_causes": ["the fundamental decision or assumption that drove the outcome"],
  "contributing_factors": ["conditions that amplified or enabled the root cause"],
  "causal_chain": ["1: first decision/event", "2: consequence", "3: final outcome"]
}}
"""
```

- [ ] **Step 4: Create attribution.py**

```python
from __future__ import annotations

import json
import re

from fm.llm import call_claude
from fm.models import Subtask, SubtaskAttribution, SubtaskIntelligence
from fm.prompts.attribution import build_attribution_prompt


def _parse_attribution(raw: str) -> SubtaskAttribution | None:
    json_match = re.search(r"\{[\s\S]*\}", raw)
    if not json_match:
        return None
    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return None

    try:
        return SubtaskAttribution(
            root_causes=data.get("root_causes", []),
            contributing_factors=data.get("contributing_factors", []),
            causal_chain=data.get("causal_chain", []),
        )
    except (KeyError, TypeError):
        return None


def extract_attribution(
    subtask: Subtask,
    intelligence: SubtaskIntelligence,
    *,
    model: str = "sonnet",
) -> SubtaskAttribution | None:
    """Run the Decision Attribution Analyzer on a subtask."""
    prompt = build_attribution_prompt(subtask, intelligence)
    raw = call_claude(prompt, model=model)
    if raw is None:
        return None
    return _parse_attribution(raw)
```

- [ ] **Step 5: Run test to verify it passes**

```bash
cd /home/brummerv/future_memory && .venv/bin/pytest tests/test_attribution.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /home/brummerv/future_memory && git add src/fm/prompts/attribution.py src/fm/attribution.py tests/test_attribution.py && git commit -m "feat: add Decision Attribution Analyzer"
```

---

## Task 7: Contextual Learning Generator + rebuild extractor.py

**Files:**
- Create: `src/fm/prompts/tips_gen.py`
- Rewrite: `src/fm/extractor.py`
- Test: `tests/test_extractor_pipeline.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_extractor_pipeline.py
import json
from unittest.mock import patch, call
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
    responses = [None, INTELLIGENCE_RESPONSE, ATTRIBUTION_RESPONSE, TIPS_RESPONSE]
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/brummerv/future_memory && .venv/bin/pytest tests/test_extractor_pipeline.py -v
```

Expected: tests fail (old extractor.py doesn't use the new pipeline).

- [ ] **Step 3: Create prompts/tips_gen.py**

```python
from __future__ import annotations

import json
from fm.models import Subtask, SubtaskAttribution, SubtaskIntelligence

_TIP_SCHEMA = """\
{
  "tips": [
    {
      "category": "strategy" | "recovery" | "optimization",
      "content": "Actionable guidance in imperative form",
      "purpose": "Why this tip exists — what problem it prevents",
      "steps": ["Step 1", "Step 2"],
      "trigger": "Specific condition under which this applies",
      "negative_example": "What NOT to do — null if not applicable",
      "priority": "critical" | "high" | "medium" | "low",
      "task_context": "Domain context — null if generic"
    }
  ]
}"""


def build_tips_generation_prompt(
    subtask: Subtask,
    intelligence: SubtaskIntelligence,
    attribution: SubtaskAttribution,
    *,
    project: str,
) -> str:
    patterns_str = ", ".join(intelligence.cognitive_patterns) if intelligence.cognitive_patterns else "none"
    root_causes_str = "\n".join(f"  - {c}" for c in attribution.root_causes) if attribution.root_causes else "  (none)"
    causal_chain_str = "\n".join(f"  {s}" for s in attribution.causal_chain) if attribution.causal_chain else "  (none)"

    tip_count_guidance = {
        "clean_success": "1-2 strategy tips capturing what worked. If execution was unremarkable, return [].",
        "inefficient_success": "1-2 optimization tips identifying the more efficient approach.",
        "recovery": "1-3 recovery tips focused on the root cause and how to avoid or recover from it.",
        "failure": "1-2 recovery tips on the failure mode, if a clear pattern exists.",
    }.get(intelligence.outcome, "2-4 tips as appropriate.")

    return f"""\
You are generating structured tips from a subtask analysis in project "{project}".

## Subtask (generalised)
{subtask.generalized_description}

## Outcome
{intelligence.outcome}

## Cognitive Patterns
{patterns_str}

## Root Causes
{root_causes_str}

## Causal Chain
{causal_chain_str}

## Task
Generate concrete, actionable tips from this subtask. Tip count guidance: {tip_count_guidance}

Requirements:
- **Actionable**: Stated in imperative form ("When X, do Y")
- **Generalized**: No project-specific names, file paths, or IDs — must transfer to other contexts
- **Non-obvious**: Skip tips for basic programming practices everyone already knows
- **Prioritized**: critical=prevents failure/data loss, high=significant improvement, medium=useful, low=minor

Priority guide for outcomes:
- recovery tips from failures → high or critical
- optimization tips → medium
- strategy tips from clean success → medium or high

## Output Format

Return ONLY a JSON object matching this schema — no other text:
{_TIP_SCHEMA}

If no genuine non-obvious tips exist, return {{"tips": []}}.
"""
```

- [ ] **Step 4: Rewrite extractor.py**

```python
from __future__ import annotations

import json
import re
import sys

from fm.attribution import extract_attribution
from fm.intelligence import extract_intelligence
from fm.llm import call_claude
from fm.models import Subtask, SubtaskAttribution, SubtaskIntelligence, Tip, Turn
from fm.prompts.tips_gen import build_tips_generation_prompt
from fm.segmenter import segment_session


def _parse_tips_json(
    raw: str,
    *,
    session_id: str,
    project: str,
    subtask: Subtask,
) -> list[Tip]:
    json_match = re.search(r"\{[\s\S]*\}", raw)
    if not json_match:
        return []
    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return []

    tips_data = data.get("tips", [])
    if not isinstance(tips_data, list):
        return []

    tips = []
    for item in tips_data:
        try:
            tip = Tip(
                category=item["category"],
                content=item["content"],
                purpose=item.get("purpose", ""),
                steps=item.get("steps", []),
                trigger=item.get("trigger", ""),
                negative_example=item.get("negative_example"),
                priority=item.get("priority", "medium"),
                source_session_id=session_id,
                source_project=project,
                task_context=item.get("task_context"),
                subtask_id=subtask.id,
                subtask_description=subtask.generalized_description,
            )
            tips.append(tip)
        except (KeyError, ValueError):
            continue

    return tips


def _extract_tips_from_subtask(
    subtask: Subtask,
    intelligence: SubtaskIntelligence,
    attribution: SubtaskAttribution,
    *,
    session_id: str,
    project: str,
    model: str,
) -> list[Tip]:
    prompt = build_tips_generation_prompt(
        subtask, intelligence, attribution, project=project
    )
    raw = call_claude(prompt, model=model)
    if raw is None:
        return []
    return _parse_tips_json(raw, session_id=session_id, project=project, subtask=subtask)


def extract_tips_from_session(
    turns: list[Turn],
    *,
    session_id: str,
    project: str,
    model: str = "sonnet",
) -> list[Tip]:
    """Extract tips from a session using the full three-stage pipeline per subtask."""
    if not turns:
        return []

    subtasks = segment_session(turns, session_id=session_id, model=model)

    all_tips: list[Tip] = []
    for subtask in subtasks:
        intelligence = extract_intelligence(subtask, model=model)
        if intelligence is None:
            print(f"  Warning: intelligence extraction failed for subtask {subtask.id}", file=sys.stderr)
            continue

        attribution = extract_attribution(subtask, intelligence, model=model)
        if attribution is None:
            print(f"  Warning: attribution failed for subtask {subtask.id}", file=sys.stderr)
            continue

        tips = _extract_tips_from_subtask(
            subtask, intelligence, attribution,
            session_id=session_id, project=project, model=model,
        )
        all_tips.extend(tips)

    return all_tips
```

- [ ] **Step 5: Run test to verify it passes**

```bash
cd /home/brummerv/future_memory && .venv/bin/pytest tests/test_extractor_pipeline.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 6: Run full test suite to check for regressions**

```bash
cd /home/brummerv/future_memory && .venv/bin/pytest -v
```

Expected: all tests PASS. Fix any failures before continuing.

- [ ] **Step 7: Delete old prompts/extract.py**

```bash
cd /home/brummerv/future_memory && rm src/fm/prompts/extract.py
```

- [ ] **Step 8: Commit**

```bash
cd /home/brummerv/future_memory && git add src/fm/extractor.py src/fm/prompts/tips_gen.py tests/test_extractor_pipeline.py && git rm src/fm/prompts/extract.py && git commit -m "feat: rebuild extractor with 3-stage pipeline — segment, intelligence, attribution, tips"
```

---

## Task 8: Update store.py — add subtask columns and migration

**Files:**
- Modify: `src/fm/store.py`
- Test: add to `tests/test_store.py` (create if absent)

- [ ] **Step 1: Write failing test**

```python
# tests/test_store.py
import tempfile
from pathlib import Path
from fm.store import TipStore
from fm.models import Tip

def _make_tip(**kwargs) -> Tip:
    defaults = dict(
        category="strategy",
        content="test tip",
        purpose="testing",
        steps=["step 1"],
        trigger="when testing",
        priority="medium",
        source_session_id="sess-1",
        source_project="test",
    )
    defaults.update(kwargs)
    return Tip(**defaults)

def test_add_tip_with_subtask_fields():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = TipStore(Path(tmpdir) / "tips.db")
        tip = _make_tip(
            subtask_id="s1",
            subtask_description="Agent configures Python environment",
        )
        store.add_tip(tip)
        retrieved = store.get_tip(tip.id)
        assert retrieved is not None
        assert retrieved.subtask_id == "s1"
        assert retrieved.subtask_description == "Agent configures Python environment"

def test_add_tip_without_subtask_fields():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = TipStore(Path(tmpdir) / "tips.db")
        tip = _make_tip()  # no subtask fields
        store.add_tip(tip)
        retrieved = store.get_tip(tip.id)
        assert retrieved is not None
        assert retrieved.subtask_id is None
        assert retrieved.subtask_description is None

def test_embed_key_uses_subtask_description_when_present():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = TipStore(Path(tmpdir) / "tips.db")
        tip = _make_tip(
            subtask_id="s1",
            subtask_description="Agent debugs SSL failure",
        )
        key = store.get_embedding_key(tip)
        assert key == "Agent debugs SSL failure"

def test_embed_key_falls_back_to_content_trigger():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = TipStore(Path(tmpdir) / "tips.db")
        tip = _make_tip(content="do X", trigger="when Y")
        key = store.get_embedding_key(tip)
        assert key == "do X when Y"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/brummerv/future_memory && .venv/bin/pytest tests/test_store.py -v
```

Expected: failures on missing `subtask_id`/`subtask_description` columns and missing `get_embedding_key`.

- [ ] **Step 3: Update store.py schema and methods**

Update `_SCHEMA` — add `subtask_id` and `subtask_description` columns to the `tips` table definition:

```python
_SCHEMA = """
CREATE TABLE IF NOT EXISTS tips (
    id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    content TEXT NOT NULL,
    purpose TEXT,
    steps TEXT,
    trigger TEXT,
    negative_example TEXT,
    priority TEXT NOT NULL,
    source_session_id TEXT NOT NULL,
    source_project TEXT,
    task_context TEXT,
    subtask_id TEXT,
    subtask_description TEXT,
    embedding BLOB,
    embedding_provider TEXT,
    created_at TEXT NOT NULL
);
...
```

Add `migrate_add_subtask_columns()` method to `TipStore`:

```python
def migrate_add_subtask_columns(self) -> None:
    """Add subtask_id and subtask_description columns if they don't exist (idempotent)."""
    existing = {row[1] for row in self._conn.execute("PRAGMA table_info(tips)").fetchall()}
    if "subtask_id" not in existing:
        self._conn.execute("ALTER TABLE tips ADD COLUMN subtask_id TEXT")
    if "subtask_description" not in existing:
        self._conn.execute("ALTER TABLE tips ADD COLUMN subtask_description TEXT")
    self._conn.commit()
```

Add `get_embedding_key()` method:

```python
def get_embedding_key(self, tip: Tip) -> str:
    """Return the text to embed for retrieval — subtask description when available, else content+trigger."""
    if tip.subtask_description:
        return tip.subtask_description
    return f"{tip.content} {tip.trigger}"
```

Update `add_tip()` to include the new columns:

```python
def add_tip(
    self,
    tip: Tip,
    *,
    embedding: list[float] | None = None,
    embedding_provider: str | None = None,
) -> None:
    embedding_blob = _pack_embedding(embedding) if embedding else None
    self._conn.execute(
        """INSERT INTO tips (
            id, category, content, purpose, steps, trigger,
            negative_example, priority, source_session_id,
            source_project, task_context, subtask_id, subtask_description,
            embedding, embedding_provider, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            tip.id, tip.category, tip.content, tip.purpose,
            json.dumps(tip.steps), tip.trigger, tip.negative_example,
            tip.priority, tip.source_session_id, tip.source_project,
            tip.task_context, tip.subtask_id, tip.subtask_description,
            embedding_blob, embedding_provider,
            tip.created_at,
        ),
    )
    self._conn.commit()
```

Update `_row_to_tip()` to include the new fields:

```python
def _row_to_tip(self, row: sqlite3.Row) -> Tip:
    return Tip(
        id=row["id"],
        category=row["category"],
        content=row["content"],
        purpose=row["purpose"] or "",
        steps=json.loads(row["steps"]) if row["steps"] else [],
        trigger=row["trigger"] or "",
        negative_example=row["negative_example"],
        priority=row["priority"],
        source_session_id=row["source_session_id"],
        source_project=row["source_project"] or "",
        task_context=row["task_context"],
        subtask_id=row["subtask_id"],
        subtask_description=row["subtask_description"],
        created_at=row["created_at"],
    )
```

Call `migrate_add_subtask_columns()` in `__init__` after `executescript`:

```python
def __init__(self, db_path: Path) -> None:
    self.db_path = db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    self._conn = sqlite3.connect(str(db_path))
    self._conn.row_factory = sqlite3.Row
    self._conn.executescript(_SCHEMA)
    self.migrate_add_subtask_columns()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/brummerv/future_memory && .venv/bin/pytest tests/test_store.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Run full test suite**

```bash
cd /home/brummerv/future_memory && .venv/bin/pytest -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /home/brummerv/future_memory && git add src/fm/store.py tests/test_store.py && git commit -m "feat: add subtask_id/subtask_description to store; get_embedding_key uses subtask description when present"
```

---

## Task 9: Update cli.py — use subtask description as embed key; add fm db migrate

**Files:**
- Modify: `src/fm/cli.py`

- [ ] **Step 1: Update embed key in `fm tips embed` command**

Find the line in `cli.py` that builds the embedding text for tips:

```python
texts = [f"{tip.content} {tip.trigger}" for tip in tips]
```

Replace with:

```python
texts = [store.get_embedding_key(tip) for tip in tips]
```

This applies in both the `extract-all` command (inline embedding) and the `tips embed` command.

Verify both locations — search for `tip.content` and `tip.trigger` used together in cli.py:

```bash
grep -n "tip.content\|tip.trigger" /home/brummerv/future_memory/src/fm/cli.py
```

Update all occurrences where `content + trigger` is concatenated for embedding.

- [ ] **Step 2: Add `fm db migrate` command**

After the `telemetry` command and before `main.add_command(tips)`, add:

```python
@main.group()
def db() -> None:
    """Database maintenance commands."""
    pass


@db.command("migrate")
@click.option("--db", type=click.Path(path_type=Path), default=_DEFAULT_DB)
@click.option("--clear-tips", is_flag=True, default=False,
              help="Delete all existing session-level tips and processed_sessions (required before re-extraction with new pipeline).")
def db_migrate(db: Path, clear_tips: bool) -> None:
    """Apply schema migrations to the database."""
    store = TipStore(db)
    store.migrate_add_subtask_columns()
    click.echo("Schema migration complete: subtask_id and subtask_description columns added.")

    if clear_tips:
        count = store._conn.execute("SELECT COUNT(*) FROM tips").fetchone()[0]
        store._conn.execute("DELETE FROM tips")
        store._conn.execute("DELETE FROM processed_sessions")
        store._conn.commit()
        click.echo(f"Cleared {count} tips and all processed_sessions. Ready for fresh extraction.")
    else:
        click.echo("Pass --clear-tips to clear old session-level tips and processed_sessions before re-running extract-all.")


main.add_command(db)
```

- [ ] **Step 3: Run full test suite**

```bash
cd /home/brummerv/future_memory && .venv/bin/pytest -v
```

Expected: all tests PASS.

- [ ] **Step 4: Smoke test the new commands**

```bash
/home/brummerv/future_memory/.venv/bin/fm db migrate
```

Expected: `Schema migration complete: subtask_id and subtask_description columns added.` (idempotent — safe to run on existing DB).

- [ ] **Step 5: Commit**

```bash
cd /home/brummerv/future_memory && git add src/fm/cli.py && git commit -m "feat: use subtask_description as embedding key; add fm db migrate --clear-tips"
```

---

## Task 10: Migration and re-extraction

This task is operational, not code. Run once when ready to re-extract from scratch.

> ⚠️ This deletes all existing tips and extraction history. You will re-extract from 3,800+ sessions using the new pipeline. Extraction will be significantly slower (~4 LLM calls per session vs 1 before). Use `--model haiku` for the full run to keep costs manageable — the `--model` flag already exists on `extract-all`. Make sure you want to do this before running.

- [ ] **Step 1: Run migration**

```bash
/home/brummerv/future_memory/.venv/bin/fm db migrate --clear-tips
```

Expected output:
```
Schema migration complete: subtask_id and subtask_description columns added.
Cleared N tips and all processed_sessions. Ready for fresh extraction.
```

- [ ] **Step 2: Verify DB is empty**

```bash
/home/brummerv/future_memory/.venv/bin/fm telemetry
```

Expected: `Total retrievals: 0`, tips count 0.

- [ ] **Step 3: Test extraction on a single session before running extract-all**

Find a session JSONL with a reasonable number of turns:

```bash
ls -lS ~/.claude/projects/*/*.jsonl | tail -20
```

Pick a mid-sized file (~50-200KB) and run extraction on it:

```bash
/home/brummerv/future_memory/.venv/bin/fm extract ~/.claude/projects/<project>/<session>.jsonl
```

Verify: tips are produced, each has `subtask_id` and `subtask_description` set.

```bash
/home/brummerv/future_memory/.venv/bin/fm tips list | head -40
```

- [ ] **Step 4: Run extract-all**

```bash
/home/brummerv/future_memory/.venv/bin/fm extract-all
```

Leave running. Will take significantly longer than before due to ~4 LLM calls per session.

---

## Self-Review

**Spec coverage:**

| Paper requirement | Task |
|---|---|
| Phase A: subtask segmentation | Task 4 |
| Generalised descriptions (entity abstraction, action normalization, context removal) | Task 3 (prompt) |
| Stage 1: Trajectory Intelligence Extractor | Task 5 |
| Stage 2: Decision Attribution Analyzer | Task 6 |
| Stage 3: Contextual Learning Generator | Task 7 |
| Subtask description as retrieval embedding key | Task 8 + 9 |
| Store schema updated for subtask provenance | Task 8 |
| Migration path from old session-level tips | Task 9 + 10 |
| Three tip categories (strategy/recovery/optimization) | Unchanged |
| τ ≥ 0.6, top-5 retrieval | Unchanged |

**No placeholders:** Verified — all steps contain actual code or commands.

**Type consistency:** `Subtask` defined in Task 1, used in Tasks 4–7. `SubtaskIntelligence`/`SubtaskAttribution` defined Task 1, used Tasks 5–7. `Tip.subtask_id`/`subtask_description` defined Task 1, written in Task 7 extractor, stored in Task 8.
