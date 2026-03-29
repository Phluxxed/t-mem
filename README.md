# future_memory

Trajectory-informed memory generation for self-improving agent systems. Extracts actionable tips from Claude Code session logs and injects relevant ones into future sessions via a hook.

Based on [arXiv:2603.10600](https://arxiv.org/abs/2603.10600) (IBM Research, Feb 2026).

## Setup

```bash
# Install
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Set your Voyage AI key
export VOYAGE_API_KEY=your-key-here
```

## Usage

```bash
# Extract tips from a session
fm extract ~/.claude/projects/<project>/<session>.jsonl

# Extract all unprocessed sessions
fm extract-all

# Retrieve tips manually
fm retrieve "your task description"

# List stored tips
fm tips list

# Show a specific tip
fm tips show <id-prefix>
```

## Hook Integration

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

## Tests

```bash
pytest tests/ -v
```
