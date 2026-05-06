"""Discover top failure / inefficiency patterns in pre-implementation sessions.

Two passes:
  1. Mechanical aggregation — tool errors, same-file re-reads, edit-without-read,
     repeated bash commands, recovery sequences.
  2. LLM categorization (Haiku, parallel) — anti-patterns that don't have an
     explicit error signature.

Combined into a ranked report we can use to pick which metrics are worth
implementing in baseline.py.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
import signal
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fm.llm import call_claude_async
from fm.parser import parse_session


CUTOFF = datetime(2026, 3, 30, tzinfo=timezone.utc)
PROJECTS = Path.home() / ".claude" / "projects"
SAMPLE_SIZE = 50
MIN_TURNS = 5
MAX_FILE_BYTES = 5 * 1024 * 1024  # skip > 5MB files (memory blowups)
SEED = 42
LLM_BATCH = 2  # parallel Haiku calls; each spawns a claude CLI subprocess (memory-bound)
PARSE_TIMEOUT_S = 8  # per-file parser timeout — protects against pathological JSONLs


class _ParseTimeout(Exception):
    """Raised when parse_session exceeds PARSE_TIMEOUT_S."""


def _timeout_handler(signum, frame):
    raise _ParseTimeout()


# ---------------- Mechanical pass ----------------

def _norm_error(stderr: str) -> str:
    """Normalize an error message to a signature for aggregation."""
    if not stderr:
        return ""
    s = stderr.strip().split("\n")[0][:120]
    s = re.sub(r"/[^\s'\"]+", "<path>", s)
    s = re.sub(r"\b\d+\b", "N", s)
    return s


def _read_path(action) -> str | None:
    if action.tool_name == "Read":
        return (action.tool_input or {}).get("file_path")
    return None


def _edit_path(action) -> str | None:
    if action.tool_name in ("Edit", "Write", "NotebookEdit"):
        return (action.tool_input or {}).get("file_path")
    return None


def _bash_cmd(action) -> str | None:
    if action.tool_name == "Bash":
        cmd = (action.tool_input or {}).get("command", "")
        # collapse whitespace, take first 80 chars as signature
        cmd = re.sub(r"\s+", " ", cmd).strip()[:120]
        return cmd
    return None


def analyze_session_mechanical(turns) -> dict:
    out = {
        "tool_errors": [],          # (tool, error_signature)
        "recovery_pairs": [],       # (failed_tool, next_tool_after_error)
        "same_file_rereads": [],    # path (each extra read counts)
        "edit_without_read": [],    # (tool, path)
        "bash_repeats": [],         # cmd_signature
        "consec_same_tool": [],     # tool name
        "long_unrecovered_errors": [],  # tool name (error followed only by non-recoveries)
    }
    actions = [a for t in turns for a in t.actions]
    files_read_at: dict[str, int] = {}  # path -> first index
    bash_seen: dict[str, int] = {}      # cmd -> first index
    last_tool: str | None = None

    for i, a in enumerate(actions):
        # consecutive same tool
        if last_tool and a.tool_name == last_tool:
            out["consec_same_tool"].append(a.tool_name)
        last_tool = a.tool_name

        # error tracking
        if not a.success:
            err = _norm_error(a.result_stderr or a.result_stdout or "")
            out["tool_errors"].append((a.tool_name, err))
            # what's the next non-same-failure action?
            next_action = actions[i + 1] if i + 1 < len(actions) else None
            if next_action:
                out["recovery_pairs"].append((a.tool_name, next_action.tool_name))

        # same-file read
        rp = _read_path(a)
        if rp:
            if rp in files_read_at:
                out["same_file_rereads"].append(rp)
            else:
                files_read_at[rp] = i

        # edit without prior read
        ep = _edit_path(a)
        if ep and ep not in files_read_at:
            out["edit_without_read"].append((a.tool_name, ep))

        # repeated bash command
        bc = _bash_cmd(a)
        if bc:
            if bc in bash_seen:
                out["bash_repeats"].append(bc)
            else:
                bash_seen[bc] = i

    return out


# ---------------- LLM pass ----------------

LLM_PROMPT = """\
You are analysing a Claude Code agent session for FAILURE and INEFFICIENCY patterns.

Identify up to 3 patterns where the agent either failed/recovered, OR could have been more efficient. Focus on patterns that REPEAT or that represent clearly suboptimal behaviour.

Examples of patterns worth reporting:
  - "ran tests with broken setup" (didn't verify env first)
  - "kept retrying same broken command without changing anything"
  - "edited file without reading it first → produced wrong patch"
  - "asked user for info it could have figured out"
  - "ran commands sequentially when parallel would have worked"
  - "rediscovered the same fact in multiple turns"
  - "used per-item loop where bulk operation existed"

For each pattern return:
  - "label": short snake_case identifier (max 4 words)
  - "description": one-sentence plain English
  - "severity": "minor" | "major"
  - "recurring": true if seen 2+ times in this session, else false

Return ONLY valid JSON:
{"patterns": [{"label": "...", "description": "...", "severity": "...", "recurring": ...}]}

If no notable patterns, return {"patterns": []}.

------ SESSION ------
""".strip()


def compact_session_summary(turns, max_turns: int = 25, max_action_chars: int = 180) -> str:
    """Produce a compact text summary of a session for LLM analysis."""
    lines: list[str] = []
    for ti, t in enumerate(turns[:max_turns], 1):
        if t.user_prompt:
            lines.append(f"\n[Turn {ti}] USER: {t.user_prompt[:200]}")
        else:
            lines.append(f"\n[Turn {ti}]")

        # compact thinking signal (first 80 chars of first thinking block, if any)
        if t.thinking:
            think = t.thinking[0][:160].replace("\n", " ")
            lines.append(f"  think: {think}")

        for a in t.actions:
            mark = "✗" if not a.success else "·"
            ti_keys = list((a.tool_input or {}).keys())[:4]
            args_preview = " ".join(f"{k}={str((a.tool_input or {}).get(k, ''))[:60]!r}" for k in ti_keys)
            line = f"  {mark} {a.tool_name}({args_preview})"
            if not a.success:
                err = (a.result_stderr or a.result_stdout or "")[:120].replace("\n", " ")
                line += f" → ERR: {err}"
            lines.append(line[:max_action_chars])

    if len(turns) > max_turns:
        lines.append(f"\n... ({len(turns) - max_turns} more turns truncated)")
    return "\n".join(lines)


def parse_patterns_json(raw: str) -> list[dict]:
    if not raw:
        return []
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return []
    try:
        data = json.loads(m.group())
    except json.JSONDecodeError:
        return []
    return data.get("patterns", [])


async def classify_session(turns) -> list[dict]:
    summary = compact_session_summary(turns)
    prompt = f"{LLM_PROMPT}\n{summary}"
    raw = await call_claude_async(prompt, model="haiku")
    return parse_patterns_json(raw or "")


async def classify_summary(summary: str) -> list[dict]:
    """Classify a pre-built summary (used in streaming pipeline to avoid re-parsing)."""
    prompt = f"{LLM_PROMPT}\n{summary}"
    raw = await call_claude_async(prompt, model="haiku")
    return parse_patterns_json(raw or "")


# ---------------- Aggregation & report ----------------

def normalize_label(label: str) -> str:
    """Normalize an LLM-generated pattern label to snake_case for clustering."""
    s = re.sub(r"[^a-z0-9_]+", "_", label.lower()).strip("_")
    return re.sub(r"_+", "_", s)


def report_mechanical(per_session: list[dict]) -> None:
    n = len(per_session)
    print(f"\n=================== MECHANICAL PASS ({n} sessions) ===================\n")

    err_counter = Counter()
    for r in per_session:
        for tool, sig in r["tool_errors"]:
            err_counter[(tool, sig)] += 1
    print(f"-- Top tool errors (signature) --")
    for (tool, sig), c in err_counter.most_common(15):
        print(f"  {c:4d}  {tool:20s} {sig}")

    rec_counter = Counter()
    for r in per_session:
        for failed, nxt in r["recovery_pairs"]:
            rec_counter[(failed, nxt)] += 1
    print(f"\n-- Top recovery transitions (failed_tool → next_tool) --")
    for (failed, nxt), c in rec_counter.most_common(10):
        print(f"  {c:4d}  {failed} → {nxt}")

    print(f"\n-- Aggregate counts --")
    total_rereads = sum(len(r["same_file_rereads"]) for r in per_session)
    sessions_with_reread = sum(1 for r in per_session if r["same_file_rereads"])
    print(f"  same-file re-reads:     {total_rereads} total, in {sessions_with_reread}/{n} sessions")

    total_edit_no_read = sum(len(r["edit_without_read"]) for r in per_session)
    sessions_edit_no_read = sum(1 for r in per_session if r["edit_without_read"])
    print(f"  edit-without-read:      {total_edit_no_read} total, in {sessions_edit_no_read}/{n} sessions")

    total_bash_repeat = sum(len(r["bash_repeats"]) for r in per_session)
    sessions_bash_repeat = sum(1 for r in per_session if r["bash_repeats"])
    print(f"  repeated bash cmds:     {total_bash_repeat} total, in {sessions_bash_repeat}/{n} sessions")

    consec_counter = Counter()
    for r in per_session:
        for tool in r["consec_same_tool"]:
            consec_counter[tool] += 1
    print(f"\n-- Top consecutive-same-tool calls --")
    for tool, c in consec_counter.most_common(10):
        print(f"  {c:4d}  {tool}")


def report_llm(per_session_patterns: list[list[dict]]) -> None:
    n = len(per_session_patterns)
    print(f"\n=================== LLM PASS ({n} sessions) ===================\n")

    label_counter = Counter()
    severity_counter: dict[str, Counter] = defaultdict(Counter)
    description_examples: dict[str, list[str]] = defaultdict(list)

    for patterns in per_session_patterns:
        for p in patterns:
            label = normalize_label(p.get("label", ""))
            if not label:
                continue
            label_counter[label] += 1
            severity_counter[label][p.get("severity", "?")] += 1
            desc = p.get("description", "")[:140]
            if desc and len(description_examples[label]) < 3:
                description_examples[label].append(desc)

    print(f"-- Top LLM-identified patterns --")
    for label, c in label_counter.most_common(20):
        sevs = severity_counter[label]
        sev_str = " ".join(f"{k}={v}" for k, v in sevs.most_common())
        print(f"\n  [{c:3d}x] {label}    ({sev_str})")
        for desc in description_examples[label]:
            print(f"        · {desc}")


# ---------------- Main ----------------

async def main():
    print(f"Discovering pre-implementation failure patterns")
    print(f"  cutoff:    < {CUTOFF.date()}")
    print(f"  min_turns: {MIN_TURNS}")
    print(f"  sample:    {SAMPLE_SIZE} (seed {SEED})")

    # Enumerate pre-cutoff files (mtime + size filter, no parsing yet)
    all_pre: list[Path] = []
    too_big = 0
    for f in PROJECTS.rglob("*.jsonl"):
        st = f.stat()
        if datetime.fromtimestamp(st.st_mtime, tz=timezone.utc) >= CUTOFF:
            continue
        if st.st_size > MAX_FILE_BYTES:
            too_big += 1
            continue
        all_pre.append(f)
    print(f"\n  pre-cutoff files (≤{MAX_FILE_BYTES // 1024 // 1024}MB): {len(all_pre)}  (skipped {too_big} oversized)")

    # Stream pipeline: parse → analyze → summarise → DISCARD turns immediately.
    # We never hold more than one session's parsed turns in memory at a time.
    rng = random.Random(SEED)
    rng.shuffle(all_pre)

    mechanical: list[dict] = []
    session_summaries: list[tuple[Path, str]] = []   # path + compact LLM-friendly summary
    parsed_count = 0
    skipped_short = 0
    skipped_error = 0
    skipped_timeout = 0

    signal.signal(signal.SIGALRM, _timeout_handler)
    for f in all_pre:
        if len(mechanical) >= SAMPLE_SIZE:
            break
        signal.alarm(PARSE_TIMEOUT_S)
        try:
            turns = parse_session(f)
        except _ParseTimeout:
            skipped_timeout += 1
            print(f"    [timeout] {f.name} (>{PARSE_TIMEOUT_S}s) — skipping", flush=True)
            continue
        except Exception:
            skipped_error += 1
            continue
        finally:
            signal.alarm(0)
        parsed_count += 1
        if len(turns) < MIN_TURNS:
            skipped_short += 1
            del turns
            continue

        # Process immediately while turns are in scope, store only the small derived data.
        mechanical.append(analyze_session_mechanical(turns))
        session_summaries.append((f, compact_session_summary(turns)))
        del turns  # release ~MB-scale memory before next iteration

        if len(mechanical) % 5 == 0:
            print(f"    ...{len(mechanical)}/{SAMPLE_SIZE} substantive sessions analysed (parsed {parsed_count})", flush=True)

    print(f"  parsed:    {parsed_count}  (errors: {skipped_error}, timeouts: {skipped_timeout}, < {MIN_TURNS} turns: {skipped_short})")
    print(f"  collected: {len(mechanical)}\n")

    if len(mechanical) == 0:
        print("No substantive sessions found. Aborting.")
        return

    report_mechanical(mechanical)

    # LLM pass (parallel batches) — uses pre-computed summaries, no re-parsing
    llm_patterns: list[list[dict]] = []
    if os.environ.get("SKIP_LLM", "").lower() in ("1", "true", "yes"):
        print("\nSkipping LLM pass (SKIP_LLM set).", flush=True)
    else:
        print(f"\nRunning LLM pass (Haiku, batches of {LLM_BATCH})...", flush=True)
        for batch_start in range(0, len(session_summaries), LLM_BATCH):
            batch = session_summaries[batch_start:batch_start + LLM_BATCH]
            results = await asyncio.gather(
                *[classify_summary(summary) for _, summary in batch],
                return_exceptions=True,
            )
            for r in results:
                if isinstance(r, BaseException):
                    llm_patterns.append([])
                else:
                    llm_patterns.append(r)
            done = batch_start + len(batch)
            print(f"  ...{done}/{len(session_summaries)} sessions classified", flush=True)

        report_llm(llm_patterns)

    # Persist raw results so we can iterate without re-running LLM
    out_path = Path(__file__).resolve().parent / "discover_failures_results.json"
    out_path.write_text(json.dumps({
        "computed_at": datetime.now(tz=timezone.utc).isoformat(),
        "config": {
            "cutoff": CUTOFF.isoformat(),
            "min_turns": MIN_TURNS,
            "sample_size": len(mechanical),
            "seed": SEED,
        },
        "sessions": [str(p) for p, _ in session_summaries],
        "mechanical": [
            {k: (list(v) if isinstance(v, list) else v) for k, v in m.items()}
            for m in mechanical
        ],
        "llm_patterns": llm_patterns,
    }, indent=2))
    print(f"\nRaw results saved to {out_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
