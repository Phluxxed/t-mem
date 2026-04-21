# t-mem

Trajectory-informed memory generation for self-improving agent systems. Extracts actionable tips from Claude Code session logs and injects relevant ones into future sessions via a hook.

Based on [arXiv:2603.10600](https://arxiv.org/abs/2603.10600) (IBM Research, Feb 2026).

> **Built for Claude Code.** The extraction pipeline parses Claude Code's JSONL session format, and the hook integration uses Claude Code's `UserPromptSubmit` and `PreCompact` hooks. Adapting to another agent requires replacing the session parser (`src/fm/parser.py`) and the hook scripts — the tip store, embeddings, and retrieval logic are agent-agnostic.

> **Your data stays local.** The tip database lives at `~/.future_memory/tips.db` on your machine. Two things are sent to [Voyage AI](https://www.voyageai.com/) for embedding: generalized tip descriptions (entity names stripped by the extraction pipeline) and abstracted retrieval queries (entity names stripped by a Haiku call before embedding). Raw session content, raw prompts, and tip text are never sent externally.

## Example

An end-user flow in Claude Code looks like this:

1. You work on a task and eventually discover something useful, like "restart the dev server after regenerating Prisma client or tests will keep using the old schema."
2. When the session compacts, the `PreCompact` hook runs `fm extract` on that session transcript and saves the distilled tip to the local tip database.
3. A day later, you start a new prompt such as "fix the failing auth tests after the schema change".
4. Before Claude sees your prompt, the `UserPromptSubmit` hook runs `fm hook-retrieve` and finds the earlier Prisma tip because it matches the new task.
5. That tip is injected into the prompt context, so Claude starts with the relevant fix in mind instead of rediscovering it from scratch.

In short: useful lessons from previous sessions are automatically extracted after the session, then automatically surfaced at the start of future related tasks.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for environment management
- [Claude Code](https://claude.ai/code) (for hook integration)
- A [Voyage AI](https://www.voyageai.com/) API key (free tier is sufficient)

## Setup

```bash
# 1. Clone and install
git clone https://github.com/Phluxxed/t-mem.git
cd t-mem
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# 2. Configure environment
cp .env.example .env
# Edit .env and set VOYAGE_API_KEY=your-key-here
```

The database is created automatically at `~/.future_memory/tips.db` on first run.

## Usage

### Extraction

```bash
# Extract tips from a single session
fm extract ~/.claude/projects/<project>/<session>.jsonl

# Extract all unprocessed sessions (supports --since 30d / 2w / 6h)
fm extract-all --since 30d
```

### Retrieval

```bash
# Retrieve relevant tips for a task description
fm retrieve "your task description"
```

### Tip management

```bash
fm tips list                      # List all tips (optionally --category strategy|recovery|optimization)
fm tips show <id-prefix>          # Show full detail for a tip
fm tips embed                     # Backfill embeddings for tips missing them
fm tips embed --force             # Re-embed all tips with abstraction (run once after upgrading)
fm tips consolidate               # Deduplicate near-identical tips via LLM synthesis
fm tips consolidate --dry-run     # Preview merges without applying
fm tips backfill-titles           # Generate titles for untitled tips
```

### Monitoring

```bash
fm dashboard                      # Rich corpus health and retrieval activity summary
fm telemetry                      # Raw retrieval stats
```

### Database maintenance

```bash
fm db migrate                     # Apply schema migrations
fm db migrate --clear-tips        # Wipe tips and sessions (use before fresh re-extraction)
```

### Baseline metrics

```bash
fm baseline                       # Capture pre-injection error/recovery metrics snapshot
```

## Hook Integration

The `.claude/hooks/` directory contains shell scripts for Claude Code integration. Copy them to `~/.claude/hooks/` and make them executable:

```bash
cp .claude/hooks/future-memory-retrieve.sh ~/.claude/hooks/
cp .claude/hooks/future-memory-pre-compact.sh ~/.claude/hooks/
chmod +x ~/.claude/hooks/future-memory-retrieve.sh
chmod +x ~/.claude/hooks/future-memory-pre-compact.sh
```

Then add to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "bash ~/.claude/hooks/future-memory-retrieve.sh",
            "timeout": 10
          }
        ]
      }
    ],
    "PreCompact": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "bash ~/.claude/hooks/future-memory-pre-compact.sh",
            "timeout": 300
          }
        ]
      }
    ]
  }
}
```

- **`UserPromptSubmit`** — retrieves relevant tips and injects them before each prompt
- **`PreCompact`** — triggers background tip extraction from the session transcript and clears the injection cache so tips aren't repeated after compaction

## Tests

```bash
pytest tests/ -v
```
