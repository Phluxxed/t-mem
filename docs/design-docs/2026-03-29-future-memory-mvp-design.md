# future_memory MVP Design

Trajectory-informed memory generation for self-improving agent systems.
Based on [arXiv:2603.10600](https://arxiv.org/abs/2603.10600) (IBM Research, Feb 2026).

## Problem

I (Claude Code) don't learn from past sessions. If I struggle with a particular pattern today, I'll struggle with the same pattern tomorrow unless someone manually updates my prompts. If I discover an efficient strategy, I can't automatically apply it to similar future tasks. The paper demonstrates that extracting structured tips from agent execution trajectories and injecting relevant ones at runtime produces measurable improvements (up to +14.3pp scenario goal completion on held-out tasks).

## MVP Scope

Implement the paper's core value loop: **trajectory → tips → store → retrieve → inject**.

Specifically:
- Task-level tip extraction only (no subtask decomposition)
- Single-pass extraction (combined intelligence/attribution/generation, not three separate LLM calls)
- Cosine similarity retrieval at τ ≥ 0.6, top-5 tips
- `UserPromptSubmit` hook only (no compact re-injection, no tool-call hooks)
- Extraction via `claude` CLI (no direct API calls)

## Components

### 1. Trajectory Parser (`fm.parser`)

Reads Claude Code JSONL session logs from `~/.claude/projects/` and produces a structured intermediate representation.

**Input**: A `.jsonl` session file.

**Keeps**:
- User prompts (non-meta, non-tool-result `type=user` entries)
- Assistant responses: text blocks, thinking blocks, tool_use blocks
- Tool results: stdout/stderr from `toolUseResult`
- Compact boundaries (marks where context was lost)
- Timestamps, session metadata (cwd, gitBranch)

**Drops**:
- `file-history-snapshot` entries
- `stop_hook_summary` system entries
- `turn_duration` system entries
- `isMeta=true` user entries (skill expansions, command metadata)
- ANSI escape codes, system-reminder tags

**Output**: A list of turns, reconstructed from `parentUuid` links (not assumed chronological).

```python
@dataclass
class Action:
    tool_name: str
    tool_input: dict
    result_stdout: str | None
    result_stderr: str | None
    success: bool

@dataclass
class Turn:
    user_prompt: str
    thinking: list[str]
    actions: list[Action]
    response_text: str
    timestamp: str
    cwd: str
```

### 2. Tip Extractor (`fm.extractor`)

Takes parsed turns and produces structured tips via the `claude` CLI.

**Single-pass extraction**: Combines the paper's three stages (trajectory intelligence extraction, decision attribution, tip generation) into one prompt call. The prompt instructs the model to:

1. Analyse the trajectory for: what went well, what failed, what was inefficient, what was recovered from
2. For failures/inefficiencies, trace back to the causal decision
3. Generate tips in each applicable category

**Tip schema**:

```python
@dataclass
class Tip:
    id: str                       # uuid
    category: str                 # "strategy" | "recovery" | "optimization"
    content: str                  # the actionable guidance
    purpose: str                  # why this tip exists
    steps: list[str]              # concrete implementation steps
    trigger: str                  # when this tip applies
    negative_example: str | None  # what NOT to do
    priority: str                 # "critical" | "high" | "medium" | "low"
    source_session_id: str        # provenance back to the JSONL
    source_project: str           # which project this came from
    task_context: str | None      # domain/application context
    created_at: str
```

**Prompt design**: The extraction prompt includes the full parsed trajectory (or chunked by turns if too large) and asks for JSON output matching the tip schema. Includes 2-3 few-shot examples derived from the paper's checkout/cart/payment examples.

**Large session handling**: If a parsed session exceeds what a single `claude` call can handle, chunk by groups of turns. Each chunk includes context about the session's overall purpose from the first few turns.

### 3. Tip Store (`fm.store`)

SQLite database at `~/.future_memory/tips.db`.

```sql
CREATE TABLE tips (
    id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    content TEXT NOT NULL,
    purpose TEXT,
    steps TEXT,                    -- JSON array
    trigger TEXT,
    negative_example TEXT,
    priority TEXT NOT NULL,
    source_session_id TEXT NOT NULL,
    source_project TEXT,
    task_context TEXT,
    embedding BLOB,
    embedding_provider TEXT,       -- "voyage" | "huggingface" | NULL
    created_at TEXT NOT NULL
);

CREATE TABLE processed_sessions (
    session_id TEXT PRIMARY KEY,
    jsonl_path TEXT NOT NULL,
    processed_at TEXT NOT NULL,
    tip_count INTEGER
);
```

`processed_sessions` tracks what's been extracted so `fm extract --all` skips already-processed sessions.

When a tip is stored, the `content + trigger` fields are concatenated and embedded. The provider name is stored alongside the vector since different models produce different vector spaces.

### 4. Tip Retriever (`fm.retriever`)

Given a task description (the user's prompt):

1. Embed it via the embedding provider chain
2. Load tip embeddings from SQLite (matching the same provider)
3. Compute cosine similarity
4. Filter to τ ≥ 0.6
5. Return top-5 by score
6. Format for prompt injection

**Embedding provider chain** (tried in order):
1. **Voyage AI** — primary, best quality
2. **HuggingFace Inference API** — free tier fallback, `sentence-transformers/all-MiniLM-L6-v2`
3. **NULL** — last resort, fall back to keyword overlap scoring

The retriever only compares embeddings from the same provider. If the store has a mix (e.g. Voyage was down during one extraction), it queries each provider's embeddings separately and merges results by score.

**Output format**:
```
[PRIORITY: HIGH] Strategy Tip:
When setting up a new Python project, verify the virtual
environment is created and activated before installing dependencies.

Apply when: Task involves project setup or dependency installation
Steps:
1. Check for existing venv
2. Create with python -m venv if missing
3. Verify activation before pip install

Source: session abc123 (2026-03-28)
---
```

**Fallback** (no embeddings available): Simple keyword overlap scoring against tip `content + trigger` fields.

### 5. CLI Interface

```
fm extract <session.jsonl>          # Run extraction pipeline on a session
fm extract --all                    # Process all unprocessed sessions
fm retrieve "task description"      # Get relevant tips as formatted text
fm hook-retrieve                    # Hook entrypoint: reads JSON from stdin, outputs tips to stdout
fm tips list                        # Browse stored tips
fm tips show <id>                   # Inspect a specific tip with provenance
```

### 6. Hook Integration

In `~/.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "command": "fm hook-retrieve",
        "timeout": 5000
      }
    ]
  }
}
```

The hook receives the user's prompt via **stdin as JSON**:
```json
{
  "session_id": "abc123",
  "transcript_path": "/path/to/session.jsonl",
  "cwd": "/current/working/dir",
  "hook_event_name": "UserPromptSubmit",
  "prompt": "the user's actual message"
}
```

`fm hook-retrieve` parses the JSON from stdin, extracts the `prompt` field, runs retrieval, and writes matching tips as **plain text to stdout** — which gets automatically injected into context.

5-second timeout — if retrieval is slow or fails, the session proceeds without tips.

Tips are searched globally across all projects (a strategy learned in one project may help in another).

## Data Flow

```
Session JSONL ──→ Parser ──→ Turns ──→ Extractor (claude CLI) ──→ Tips ──→ Store (SQLite + embeddings)
                                                                              │
User Prompt ──→ Hook fires ──→ Retriever ──→ cosine similarity ──→ top-5 ──→ injected into context
```

## Tech Stack

- **Language**: Python 3.12+, type hints on all signatures
- **Storage**: SQLite (stdlib)
- **Embeddings**: Voyage AI (primary), HuggingFace Inference API (fallback)
- **LLM calls**: `claude` CLI via subprocess (until API key access is available)
- **CLI framework**: `argparse` or `click` (TBD during implementation)
- **Dependencies**: `voyageai`, `requests` (for HuggingFace), `numpy` (for cosine similarity)

## Future Phases

What we're deliberately skipping relative to the paper, and why it matters:

### Phase 2: Baseline Metrics & Evaluation
`fm baseline` command that analyses existing session logs and produces a "before" snapshot: error rates (failed tool calls), recovery patterns (error → retry → success), efficiency metrics (tool calls per turn, repeated operations), and session length. Running this before and after tip injection gives a measurable delta. Critical for demonstrating value to other users.

### Phase 2b: Incremental Session Re-extraction
Currently `extract-all` skips sessions that have already been processed, even if the session has continued and grown significantly. Track `last_processed_size` (or line count) alongside the session ID and re-extract sessions that have grown beyond a threshold since last processing. Ensures tips from ongoing sessions aren't lost.

### Phase 3: Subtask-Level Extraction (Paper §3.1.4)
The paper's two-phase pipeline segments trajectories into logical subtasks (authentication, data retrieval, processing, etc.) and extracts tips per-subtask. This enables cross-task transfer — an auth tip from a Spotify task helps with a Venmo task. Our task-level tips won't transfer as well across dissimilar tasks that share subtask patterns.

### Phase 3: Tip Storage & Management (Paper §3.2)
- **Description generalisation**: Abstract entity-specific details, normalise verbs, strip context qualifiers
- **Semantic clustering**: Group tips with similar generalised descriptions using hierarchical agglomerative clustering at ~0.85 threshold
- **Tip consolidation**: LLM-based merging of redundant tips within clusters, conflict resolution (success-derived tips trump failure-derived), synthesis of complementary tips

Without this, our tip store will grow linearly and accumulate near-duplicates over time.

### Phase 4: LLM-Guided Retrieval (Paper §3.3.2)
Uses an LLM at retrieval time to reason about task context, detect application domain, prioritise tip categories, and construct metadata-filtered queries. The paper shows this drives +7.2pp SGC over cosine-only retrieval. More expensive (extra LLM call per prompt) but significantly better at cross-variant consistency.

### Phase 5: Additional Hook Points
- **Compact events**: Re-inject critical tips that got compacted away
- **Tool call hooks**: Inject tips relevant to specific tools being used
- **Session start**: Inject high-priority tips for the current project

### Phase 6: Direct API Integration
Replace `claude` CLI subprocess calls with direct Anthropic API calls for extraction. The CLI already supports model selection (e.g. `--model haiku` for bulk extraction, `--model sonnet` for complex attribution), so the main benefits of API access are batching, programmatic error handling, and cost control.

### Phase 7: Multi-Agent / Cross-Agent Learning
The paper mentions extending to multi-agent systems with cross-agent attribution and agent-role-aware guidance. Relevant if this gets plugged into Codex or other agent systems.
