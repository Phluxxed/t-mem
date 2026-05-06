Run `fm extract` on the current session's transcript so tips can be extracted without waiting for a compact.

Steps:
1. Find the most recently modified `.jsonl` in `~/.claude/projects/$(pwd | tr '/_' '--')/` — that's the current session's transcript.
2. Run `~/future_memory/.venv/bin/fm extract <path>` against it.
3. Report which transcript was processed and the extraction summary `fm` prints.

If the project dir doesn't exist or contains no `.jsonl` files, tell the user no transcript was found and stop.
