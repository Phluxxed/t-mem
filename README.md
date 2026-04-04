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

Inject relevant tips automatically at the start of each Claude Code prompt.

Add to `~/.claude/settings.json`:

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

To suppress duplicate injections within a session, also add a `PreCompact` hook:

```json
"PreCompact": [
  {
    "matcher": "",
    "hooks": [
      {
        "type": "command",
        "command": "fm session clear-injections $SESSION_ID"
      }
    ]
  }
]
```

## Tests

```bash
pytest tests/ -v
```
