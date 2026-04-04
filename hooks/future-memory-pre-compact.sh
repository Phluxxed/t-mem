#!/bin/bash
# Triggered by Claude Code's PreCompact hook.
# Extracts tips from the session transcript in the background,
# then clears the injection cache so tips aren't re-injected after compaction.

PAYLOAD=$(cat)
TRANSCRIPT_PATH=$(python3 -c "import json,sys; print(json.loads(sys.argv[1]).get('transcript_path',''))" "$PAYLOAD")
SESSION_ID=$(python3 -c "import json,sys; print(json.loads(sys.argv[1]).get('session_id',''))" "$PAYLOAD")

if [ -z "$TRANSCRIPT_PATH" ]; then
    echo "t-mem: no transcript_path in hook payload, skipping extraction" >&2
    exit 0
fi

LOG_DIR="$HOME/.future_memory/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/extract-${SESSION_ID:-unknown}-$(date +%Y%m%dT%H%M%S).log"

echo "t-mem: launching background extraction for session ${SESSION_ID} (log: $LOG_FILE)" >&2
nohup "$HOME/future_memory/.venv/bin/fm" extract "$TRANSCRIPT_PATH" >"$LOG_FILE" 2>&1 &

if [ -n "$SESSION_ID" ]; then
    echo "t-mem: clearing injection cache for session $SESSION_ID..." >&2
    "$HOME/future_memory/.venv/bin/fm" session clear-injections "$SESSION_ID" >&2
fi

exit 0
