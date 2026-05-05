#!/bin/bash
# Inject relevant tips into the current prompt context.
# Called by Claude Code's UserPromptSubmit hook — stdin is JSON with a "prompt" key,
# stdout is prepended to the session context.

exec "$HOME/future_memory/.venv/bin/fm" hook-retrieve
