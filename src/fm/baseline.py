from __future__ import annotations

import gc
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fm.models import Action, Turn
from fm.parser import parse_session_lean as parse_session

# Skip session JSONLs larger than this when aggregating. Even 5MB JSONLs can
# OOM the process when a single line carries a megabytes-long tool result
# (e.g. `cat` of a large file). 2MB is conservative enough to be safe and
# excludes only ~1% of all session files (30 of ~3000 in our corpus).
_MAX_SESSION_BYTES = 2 * 1024 * 1024


def _parse_iso_ts(ts: str) -> datetime | None:
    """Parse a Claude Code Turn timestamp (ISO 8601 with Z or offset).

    Returns None on empty/unparseable input. We need this because the eval
    methodology is turn-level, not file-level: a single user JSONL spans
    pre- and post-implementation history when it's been resumed across the
    cutoff date, so we filter individual turns by Turn.timestamp rather than
    bucketing the whole file by mtime.
    """
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


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

    # Optimization metric: same-CHUNK re-reads. Dedup key is (path, offset, limit)
    # so that paginated reads of different chunks of the same file don't get
    # mis-counted (e.g. loci, or any large-file pagination workflow).
    # Strategy metric: edits to files that were never Read first in this session.
    # Walk in trajectory order so the "prior" sets grow naturally as we go.
    paths_ever_read: set[str] = set()
    read_chunks_seen: set[tuple[str, Any, Any]] = set()
    total_reads = 0
    same_file_rereads = 0
    total_edits = 0
    edits_without_read = 0
    for a in all_actions:
        if a.tool_name == "Read":
            total_reads += 1
            ti = a.tool_input or {}
            path = ti.get("file_path")
            if path:
                chunk_key = (path, ti.get("offset"), ti.get("limit"))
                if chunk_key in read_chunks_seen:
                    same_file_rereads += 1
                else:
                    read_chunks_seen.add(chunk_key)
                paths_ever_read.add(path)
        elif a.tool_name in ("Edit", "NotebookEdit"):
            total_edits += 1
            path = (a.tool_input or {}).get("file_path")
            if path and path not in paths_ever_read:
                edits_without_read += 1

    return {
        "turns": len(turns),
        "total_actions": total_actions,
        "failed_actions": failed_actions,
        "error_sequences": error_sequences,
        "recovered_sequences": recovered_sequences,
        "retry_counts": retry_counts,
        "repeated_op_turns": repeated_op_turns,
        "total_reads": total_reads,
        "same_file_rereads": same_file_rereads,
        "total_edits": total_edits,
        "edits_without_read": edits_without_read,
    }


def _is_user_session(f: Path) -> bool:
    """True for user-driven session JSONLs, False for subagent transcripts.

    Subagent transcripts (`agent-*.jsonl`) are task-driven runs spawned by a
    parent agent — different shape, different purpose, and they share the
    parent's session_id so they can't be cleanly attributed to tip injection.
    They're excluded from baseline/snapshot pools so metrics measure
    user-driven session improvement, not subagent behaviour.
    """
    return not f.name.startswith("agent-")


def _find_user_sessions(projects_dir: Path) -> list[Path]:
    """Find every user-driven session JSONL, regardless of mtime.

    Date filtering happens at the turn level inside `_aggregate_sessions`, not
    at the file level, because a single resumed JSONL contains turns from
    both before and after the implementation cutoff. mtime-based filtering
    would mis-bucket every resumed file as "post" and leave only never-resumed
    subagent transcripts in the "pre" pool — which is what was happening.
    """
    return [
        f for f in projects_dir.rglob("*.jsonl")
        if _is_user_session(f)
    ]


def _aggregate_sessions(
    files: list[Path],
    min_turns: int = 0,
    *,
    before: datetime | None = None,
    after: datetime | None = None,
) -> dict[str, Any]:
    """Parse a list of session files and return raw aggregate counts.

    If ``before`` is set, only turns with timestamp < before are counted.
    If ``after`` is set, only turns with timestamp >= after are counted.
    Turns whose timestamp is missing or unparseable are dropped from filtered
    runs (real Claude Code data always carries timestamps; this only affects
    synthetic test fixtures or malformed JSONLs).
    """
    agg: dict[str, Any] = {
        "sessions": 0,
        "turns": 0,
        "total_actions": 0,
        "failed_actions": 0,
        "error_sequences": 0,
        "recovered_sequences": 0,
        "all_retries": [],
        "repeated_op_turns": 0,
        "session_lengths": [],
        "total_reads": 0,
        "same_file_rereads": 0,
        "total_edits": 0,
        "edits_without_read": 0,
        "skipped": 0,
    }
    from rich.progress import Progress, BarColumn, TextColumn, MofNCompleteColumn, TimeRemainingColumn

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        TextColumn("• {task.fields[current]}"),
    )
    with progress:
        task_id = progress.add_task(
            f"Parsing {len(files)} sessions",
            total=len(files),
            current="",
        )
        for i, path in enumerate(files):
            progress.update(task_id, current=path.name[:48], advance=1)

            # Periodic explicit GC: pymalloc holds arenas across iterations even
            # after refcounts hit zero. Across hundreds of files this can build up
            # enough residue to OOM. A forced collect every 100 files releases the
            # generational pools the regular heuristic schedule misses.
            if i and i % 100 == 0:
                gc.collect()
            try:
                if path.stat().st_size > _MAX_SESSION_BYTES:
                    agg["skipped"] += 1
                    continue
                turns = parse_session(path)
            except Exception:
                agg["skipped"] += 1
                continue
            # Turn-level timestamp filter: keeps the eval methodology honest
            # for resumed sessions that span the cutoff.
            if before is not None or after is not None:
                filtered: list[Turn] = []
                for t in turns:
                    ts = _parse_iso_ts(t.timestamp)
                    if ts is None:
                        continue
                    if before is not None and ts >= before:
                        continue
                    if after is not None and ts < after:
                        continue
                    filtered.append(t)
                turns = filtered
            if not turns or len(turns) < min_turns:
                agg["skipped"] += 1
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
            agg["total_reads"] += m["total_reads"]
            agg["same_file_rereads"] += m["same_file_rereads"]
            agg["total_edits"] += m["total_edits"]
            agg["edits_without_read"] += m["edits_without_read"]
    return agg


def _metrics_from_agg(agg: dict[str, Any]) -> dict[str, Any]:
    """Compute final metric dict from raw aggregates."""
    sessions = agg["sessions"]
    lengths = sorted(agg["session_lengths"])
    p50 = lengths[len(lengths) // 2]
    p90 = lengths[int(len(lengths) * 0.9)]
    error_rate = agg["failed_actions"] / agg["total_actions"] if agg["total_actions"] else 0.0
    recovery_rate = agg["recovered_sequences"] / agg["error_sequences"] if agg["error_sequences"] else 0.0
    avg_retries = sum(agg["all_retries"]) / len(agg["all_retries"]) if agg["all_retries"] else 0.0
    same_file_reread_rate = (
        agg["same_file_rereads"] / agg["total_reads"] if agg["total_reads"] else 0.0
    )
    edit_without_read_rate = (
        agg["edits_without_read"] / agg["total_edits"] if agg["total_edits"] else 0.0
    )
    return {
        "error_rate": round(error_rate, 4),
        "recovery_rate": round(recovery_rate, 4),
        "avg_retries_after_error": round(avg_retries, 2),
        "avg_actions_per_turn": round(agg["total_actions"] / agg["turns"], 2) if agg["turns"] else 0.0,
        "avg_turns_per_session": round(agg["turns"] / sessions, 1),
        "session_length_p50": p50,
        "session_length_p90": p90,
        "repeated_op_rate": round(agg["repeated_op_turns"] / agg["turns"], 4) if agg["turns"] else 0.0,
        "same_file_reread_rate": round(same_file_reread_rate, 4),
        "edit_without_read_rate": round(edit_without_read_rate, 4),
    }


def _sample_files(all_files: list[Path], sample: int, seed: int) -> list[Path]:
    if sample > 0 and len(all_files) > sample:
        rng = random.Random(seed)
        return rng.sample(all_files, sample)
    return all_files


def compute_baseline(
    *,
    before: datetime,
    sample: int = 500,
    min_turns: int = 0,
    projects_dir: Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Parse a sample of pre-cutoff sessions and return aggregate metrics."""
    if projects_dir is None:
        projects_dir = Path.home() / ".claude" / "projects"

    all_files = _find_user_sessions(projects_dir)
    available = len(all_files)
    files = _sample_files(all_files, sample, seed)
    agg = _aggregate_sessions(files, min_turns=min_turns, before=before)

    if agg["sessions"] == 0:
        raise ValueError("No valid sessions found before the cutoff date.")

    return {
        "computed_at": datetime.now(tz=timezone.utc).isoformat(),
        "cutoff_date": before.date().isoformat(),
        "sessions_analyzed": agg["sessions"],
        "sessions_available": available,
        "sessions_skipped": agg["skipped"],
        "metrics": _metrics_from_agg(agg),
    }


def compute_snapshot(
    *,
    after: datetime,
    sample: int = 500,
    min_turns: int = 0,
    files: list[Path] | None = None,
    projects_dir: Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Parse a sample of post-cutoff sessions and return aggregate metrics.

    If ``files`` is provided, use that list directly instead of scanning the
    file system — caller is responsible for filtering by date/source.
    """
    if files is not None:
        all_files = files
    else:
        if projects_dir is None:
            projects_dir = Path.home() / ".claude" / "projects"
        all_files = _find_user_sessions(projects_dir)

    available = len(all_files)
    sampled = _sample_files(all_files, sample, seed)
    agg = _aggregate_sessions(sampled, min_turns=min_turns, after=after)

    if agg["sessions"] == 0:
        raise ValueError("No valid sessions found after the cutoff date.")

    return {
        "computed_at": datetime.now(tz=timezone.utc).isoformat(),
        "after_date": after.date().isoformat(),
        "sessions_analyzed": agg["sessions"],
        "sessions_available": available,
        "sessions_skipped": agg["skipped"],
        "metrics": _metrics_from_agg(agg),
    }
