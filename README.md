# t-mem

Trajectory-informed memory generation for self-improving agent systems. Extracts actionable tips from Claude Code session logs and injects relevant ones into future sessions via a hook.

Based on [arXiv:2603.10600](https://arxiv.org/abs/2603.10600) (IBM Research, Feb 2026).

## Setup

```bash
# Install
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Configure environment
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

The `hooks/` directory contains shell scripts for Claude Code integration. Copy them to `~/.claude/hooks/` and make them executable:

```bash
cp hooks/future-memory-retrieve.sh ~/.claude/hooks/
cp hooks/future-memory-pre-compact.sh ~/.claude/hooks/
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
