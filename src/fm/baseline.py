from __future__ import annotations

import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fm.models import Action, Turn
from fm.parser import parse_session


def _session_metrics(turns: list[Turn]) -> dict[str, Any]:
    """Compute metrics for a single session. Returns None-safe counts."""
    all_actions: list[Action] = [a for t in turns for a in t.actions]
    total_actions = len(all_actions)
    failed_actions = sum(1 for a in all_actions if not a.success)

    # Recovery: error followed by success on same tool (adjacent actions, any turn)
    error_sequences = 0
    recovered_sequences = 0
    retry_counts: list[int] = []

    i = 0
    while i < len(all_actions):
        if not all_actions[i].success:
            error_sequences += 1
            tool = all_actions[i].tool_name
            retries = 0
            j = i + 1
            recovered = False
            while j < len(all_actions) and all_actions[j].tool_name == tool:
                retries += 1
                if all_actions[j].success:
                    recovered = True
                    break
                j += 1
            if recovered:
                recovered_sequences += 1
                retry_counts.append(retries)
            i = j if recovered else i + 1
        else:
            i += 1

    # Repeated ops: same tool called consecutively within a turn
    repeated_op_turns = 0
    for turn in turns:
        for k in range(1, len(turn.actions)):
            if turn.actions[k].tool_name == turn.actions[k - 1].tool_name:
                repeated_op_turns += 1
                break

    return {
        "turns": len(turns),
        "total_actions": total_actions,
        "failed_actions": failed_actions,
        "error_sequences": error_sequences,
        "recovered_sequences": recovered_sequences,
        "retry_counts": retry_counts,
        "repeated_op_turns": repeated_op_turns,
    }


def _find_sessions(before: datetime, projects_dir: Path) -> list[Path]:
    """Find all JSONL session files modified before the cutoff date."""
    return [
        f for f in projects_dir.rglob("*.jsonl")
        if datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc) < before
    ]


def compute_baseline(
    *,
    before: datetime,
    sample: int = 500,
    projects_dir: Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Parse a sample of pre-cutoff sessions and return aggregate metrics."""
    if projects_dir is None:
        projects_dir = Path.home() / ".claude" / "projects"

    all_files = _find_sessions(before, projects_dir)
    available = len(all_files)

    if sample > 0 and len(all_files) > sample:
        rng = random.Random(seed)
        files = rng.sample(all_files, sample)
    else:
        files = all_files

    agg = {
        "sessions": 0,
        "turns": 0,
        "total_actions": 0,
        "failed_actions": 0,
        "error_sequences": 0,
        "recovered_sequences": 0,
        "all_retries": [],
        "repeated_op_turns": 0,
        "session_lengths": [],
    }
    skipped = 0

    for path in files:
        try:
            turns = parse_session(path)
        except Exception:
            skipped += 1
            continue

        if not turns:
            skipped += 1
            continue

        m = _session_metrics(turns)
        agg["sessions"] += 1
        agg["turns"] += m["turns"]
        agg["total_actions"] += m["total_actions"]
        agg["failed_actions"] += m["failed_actions"]
        agg["error_sequences"] += m["error_sequences"]
        agg["recovered_sequences"] += m["recovered_sequences"]
        agg["all_retries"].extend(m["retry_counts"])
        agg["repeated_op_turns"] += m["repeated_op_turns"]
        agg["session_lengths"].append(m["turns"])

    sessions = agg["sessions"]
    if sessions == 0:
        raise ValueError("No valid sessions found before the cutoff date.")

    lengths = sorted(agg["session_lengths"])
    p50 = lengths[len(lengths) // 2]
    p90 = lengths[int(len(lengths) * 0.9)]

    error_rate = agg["failed_actions"] / agg["total_actions"] if agg["total_actions"] else 0.0
    recovery_rate = agg["recovered_sequences"] / agg["error_sequences"] if agg["error_sequences"] else 0.0
    avg_retries = sum(agg["all_retries"]) / len(agg["all_retries"]) if agg["all_retries"] else 0.0

    return {
        "computed_at": datetime.now(tz=timezone.utc).isoformat(),
        "cutoff_date": before.date().isoformat(),
        "sessions_analyzed": sessions,
        "sessions_available": available,
        "sessions_skipped": skipped,
        "metrics": {
            "error_rate": round(error_rate, 4),
            "recovery_rate": round(recovery_rate, 4),
            "avg_retries_after_error": round(avg_retries, 2),
            "avg_actions_per_turn": round(agg["total_actions"] / agg["turns"], 2) if agg["turns"] else 0.0,
            "avg_turns_per_session": round(agg["turns"] / sessions, 1),
            "session_length_p50": p50,
            "session_length_p90": p90,
            "repeated_op_rate": round(agg["repeated_op_turns"] / agg["turns"], 4) if agg["turns"] else 0.0,
        },
    }
