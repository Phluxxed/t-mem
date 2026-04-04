from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import click

from fm.baseline import compute_baseline
from fm.embeddings import embed_texts_batch, get_available_provider
from fm.extractor import extract_tips_from_session
from fm.llm import call_claude
from fm.parser import parse_session
from fm.retriever import format_tips, retrieve_tips
from fm.store import TipStore

_DEFAULT_DB = Path.home() / ".future_memory" / "tips.db"

_SINCE_UNITS = {"h": "hours", "d": "days", "w": "weeks"}


def _parse_since(value: str) -> datetime | None:
    """Parse a duration string like '30d', '2w', '6h' into a UTC cutoff datetime."""
    if len(value) < 2:
        return None
    unit = value[-1].lower()
    if unit not in _SINCE_UNITS:
        return None
    try:
        n = int(value[:-1])
    except ValueError:
        return None
    return datetime.now(tz=timezone.utc) - timedelta(**{_SINCE_UNITS[unit]: n})


@click.group()
def main() -> None:
    """future_memory — trajectory-informed tips for self-improving agents."""
    pass


@main.command()
@click.argument("jsonl_path", type=click.Path(exists=True, path_type=Path))
@click.option("--db", type=click.Path(path_type=Path), default=_DEFAULT_DB)
@click.option("--model", default="sonnet", help="Claude model for extraction.")
@click.option("--min-turns", default=5, help="Skip sessions with fewer turns than this.")
@click.option("--max-turns", default=None, type=int, help="Cap new turns processed per run; remainder picked up next time.")
def extract(jsonl_path: Path, db: Path, model: str, min_turns: int, max_turns: int | None) -> None:
    """Extract tips from a Claude Code session log."""
    store = TipStore(db)

    session_id, turns = parse_session(jsonl_path, return_session_id=True)
    if not session_id:
        session_id = jsonl_path.stem

    if not turns:
        click.echo("No turns found in session.")
        store.mark_session_processed(session_id, str(jsonl_path), tip_count=0, last_turn_count=0)
        return

    if len(turns) < min_turns:
        click.echo(f"Session too short ({len(turns)} turns < {min_turns}), skipping.")
        return

    last_turn_count = store.get_last_turn_count(session_id)
    new_turns = turns[last_turn_count:]

    if not new_turns:
        click.echo(f"No new turns since last extraction ({len(turns)} turns already processed).")
        return

    capped = max_turns is not None and len(new_turns) > max_turns
    batch = new_turns[:max_turns] if capped else new_turns
    new_watermark = last_turn_count + len(batch)

    project = jsonl_path.parent.name

    suffix = f" (capped at {max_turns}; {len(new_turns) - len(batch)} turns queued for next run)" if capped else ""
    click.echo(f"Extracting tips from {len(batch)} new turns (of {len(turns)} total, model: {model}){suffix}...")
    tips = extract_tips_from_session(
        batch, session_id=session_id, project=project, model=model
    )

    provider = get_available_provider()
    missing_embeddings = 0
    if tips:
        texts = [store.get_embedding_key(tip) for tip in tips]
        embeddings = embed_texts_batch(texts, provider=provider)
        for tip, emb in zip(tips, embeddings):
            if emb:
                store.add_tip(tip, embedding=emb.vector, embedding_provider=emb.provider)
            else:
                store.add_tip(tip)
                missing_embeddings += 1

    store.mark_session_processed(session_id, str(jsonl_path), tip_count=len(tips), last_turn_count=new_watermark)
    msg = f"Extracted {len(tips)} tips from session {session_id}."
    if missing_embeddings:
        msg += f" Warning: {missing_embeddings}/{len(tips)} tips saved without embeddings (run 'fm tips embed' to backfill)."
    click.echo(msg)


@main.command("extract-all")
@click.option("--db", type=click.Path(path_type=Path), default=_DEFAULT_DB)
@click.option("--model", default="sonnet", help="Claude model for extraction.")
@click.option("--min-turns", default=5, help="Skip sessions with fewer turns than this.")
@click.option("--since", default=None, help="Only process sessions modified within this window, e.g. '30d', '2w', '6h'.")
def extract_all(db: Path, model: str, min_turns: int, since: str | None) -> None:
    """Process all unprocessed Claude Code sessions."""
    projects_dir = Path.home() / ".claude" / "projects"
    if not projects_dir.exists():
        click.echo("No Claude Code projects directory found.")
        return

    cutoff: datetime | None = None
    if since:
        cutoff = _parse_since(since)
        if cutoff is None:
            click.echo(f"Invalid --since value '{since}'. Use e.g. '30d', '2w', '6h'.", err=True)
            return

    store = TipStore(db)
    total_tips = 0
    processed = 0
    skipped_short = 0
    skipped_old = 0

    for project_dir in projects_dir.iterdir():
        if not project_dir.is_dir():
            continue
        for jsonl_file in project_dir.glob("*.jsonl"):
            if cutoff is not None:
                mtime = datetime.fromtimestamp(jsonl_file.stat().st_mtime, tz=timezone.utc)
                if mtime < cutoff:
                    skipped_old += 1
                    continue

            session_id, turns = parse_session(jsonl_file, return_session_id=True)
            if not session_id:
                session_id = jsonl_file.stem

            if not turns:
                store.mark_session_processed(session_id, str(jsonl_file), tip_count=0, last_turn_count=0)
                continue

            if len(turns) < min_turns:
                skipped_short += 1
                continue

            last_turn_count = store.get_last_turn_count(session_id)
            new_turns = turns[last_turn_count:]
            if not new_turns:
                continue

            click.echo(f"Processing {project_dir.name}/{jsonl_file.name} ({len(new_turns)} new / {len(turns)} total)...")

            project = project_dir.name
            tips = extract_tips_from_session(
                new_turns, session_id=session_id, project=project, model=model
            )

            provider = get_available_provider()
            missing_embeddings = 0
            if tips:
                texts = [store.get_embedding_key(tip) for tip in tips]
                embeddings = embed_texts_batch(texts, provider=provider)
                for tip, emb in zip(tips, embeddings):
                    if emb:
                        store.add_tip(tip, embedding=emb.vector, embedding_provider=emb.provider)
                    else:
                        store.add_tip(tip)
                        missing_embeddings += 1
                if missing_embeddings:
                    click.echo(f"  Warning: {missing_embeddings}/{len(tips)} tips saved without embeddings.")

            store.mark_session_processed(
                session_id, str(jsonl_file), tip_count=len(tips), last_turn_count=len(turns)
            )
            total_tips += len(tips)
            processed += 1

    msg = f"Processed {processed} sessions, extracted {total_tips} tips total."
    if skipped_short:
        msg += f" Skipped {skipped_short} short session(s) (< {min_turns} turns)."
    if skipped_old:
        msg += f" Skipped {skipped_old} old session(s) (outside --since window)."
    click.echo(msg)


@main.command()
@click.argument("query")
@click.option("--db", type=click.Path(path_type=Path), default=_DEFAULT_DB)
@click.option("--threshold", default=0.6, help="Cosine similarity threshold.")
@click.option("--top-k", default=5, help="Max tips to return.")
@click.option("--verbose", "-v", is_flag=True, help="Show debug info.")
def retrieve(query: str, db: Path, threshold: float, top_k: int, verbose: bool) -> None:
    """Retrieve relevant tips for a task description."""
    store = TipStore(db)
    if verbose:
        from fm.embeddings import get_available_provider, embed_text as _et
        from fm.retriever import _cosine_similarity
        provider = get_available_provider()
        click.echo(f"Provider: {provider}", err=True)
        result = _et(query, provider=provider)
        click.echo(f"Query embedded: {result is not None}", err=True)
        if result:
            click.echo(f"Query vector dims: {len(result.vector)}, first 5: {result.vector[:5]}", err=True)
        stored = store.get_tips_with_embeddings(provider=provider or "voyage")
        click.echo(f"Stored tips with provider '{provider}': {len(stored)}", err=True)
        if result and stored:
            for s in stored:
                sim = _cosine_similarity(result.vector, s["embedding"])
                click.echo(f"  sim={sim:.4f} dims={len(s['embedding'])} | {s['content'][:60]}", err=True)
    tips = retrieve_tips(query, store, threshold=threshold, top_k=top_k)
    output = format_tips(tips)
    if output:
        click.echo(output)
    else:
        click.echo("No matching tips found.")


@main.command("hook-retrieve")
@click.option("--db", type=click.Path(path_type=Path), default=_DEFAULT_DB)
@click.option("--threshold", default=0.6)
@click.option("--top-k", default=5)
def hook_retrieve(db: Path, threshold: float, top_k: int) -> None:
    """Hook entrypoint: reads JSON from stdin, outputs tips to stdout."""
    try:
        raw = sys.stdin.read()
        data = json.loads(raw)
        prompt = data.get("prompt", "")
    except (json.JSONDecodeError, KeyError) as e:
        print(f"hook-retrieve: failed to parse stdin payload: {e}", file=sys.stderr)
        return

    if not prompt:
        return

    session_id = data.get("session_id") or None

    store = TipStore(db)
    tips = retrieve_tips(prompt, store, threshold=threshold, top_k=top_k, session_id=session_id)
    output = format_tips(tips)
    if output:
        click.echo(output)
        if session_id and tips:
            store.record_injections(session_id, [t.id for t in tips])


@main.group()
def tips() -> None:
    """Manage stored tips."""
    pass


@tips.command("list")
@click.option("--db", type=click.Path(path_type=Path), default=_DEFAULT_DB)
@click.option("--category", type=click.Choice(["strategy", "recovery", "optimization"]))
def tips_list(db: Path, category: str | None) -> None:
    """List all stored tips."""
    store = TipStore(db)
    all_tips = store.list_tips(category=category)
    if not all_tips:
        click.echo("No tips stored.")
        return
    for tip in all_tips:
        click.echo(f"[{tip.id[:8]}] [{tip.priority}] {tip.category}: {tip.content[:80]}")


@tips.command("show")
@click.argument("tip_id")
@click.option("--db", type=click.Path(path_type=Path), default=_DEFAULT_DB)
def tips_show(tip_id: str, db: Path) -> None:
    """Show details of a specific tip."""
    store = TipStore(db)
    all_tips = store.list_tips()
    matches = [t for t in all_tips if t.id.startswith(tip_id)]
    if not matches:
        click.echo(f"No tip found matching '{tip_id}'.")
        return
    if len(matches) > 1:
        click.echo(f"Multiple tips match '{tip_id}'. Be more specific.")
        return
    tip = matches[0]
    click.echo(format_tips([tip]))
    click.echo(f"\nID: {tip.id}")
    click.echo(f"Project: {tip.source_project}")
    click.echo(f"Session: {tip.source_session_id}")
    click.echo(f"Created: {tip.created_at}")


@tips.command("embed")
@click.option("--db", type=click.Path(path_type=Path), default=_DEFAULT_DB)
def tips_embed(db: Path) -> None:
    """Backfill embeddings for tips that are missing them."""
    store = TipStore(db)
    all_tips = store.list_tips()
    provider = get_available_provider()
    if not provider:
        click.echo("No embedding provider available. Set VOYAGE_API_KEY or check network.")
        return

    unembedded = [
        tip for tip in all_tips
        if not (store.get_tip_with_embedding(tip.id) or {}).get("embedding")
    ]

    if not unembedded:
        click.echo("All tips already have embeddings.")
        return

    click.echo(f"Embedding {len(unembedded)} tip(s) using {provider}...")
    texts = [store.get_embedding_key(tip) for tip in unembedded]
    results = embed_texts_batch(texts, provider=provider)

    updated = 0
    failed = 0
    for tip, embedding_result in zip(unembedded, results):
        if embedding_result:
            store._conn.execute(
                "UPDATE tips SET embedding = ?, embedding_provider = ? WHERE id = ?",
                (
                    __import__("struct").pack(f"{len(embedding_result.vector)}f", *embedding_result.vector),
                    embedding_result.provider,
                    tip.id,
                ),
            )
            updated += 1
        else:
            click.echo(f"  Warning: failed to embed tip {tip.id} ({store.get_embedding_key(tip)[:60]})", err=True)
            failed += 1
    store._conn.commit()

    msg = f"Embedded {updated} tip(s)."
    if failed:
        msg += f" {failed} failed — run 'fm tips embed' again to retry."
    click.echo(msg)


@tips.command("consolidate")
@click.option("--db", type=click.Path(path_type=Path), default=_DEFAULT_DB)
@click.option("--threshold", default=0.88, help="Cosine similarity threshold for candidate clusters.")
@click.option("--dry-run", is_flag=True, help="Show proposed merges without applying.")
@click.option("--model", default="sonnet", help="Claude model for merge decisions.")
@click.option("--limit", default=0, help="Cap number of clusters to process (0 = all).")
def tips_consolidate(db: Path, threshold: float, dry_run: bool, model: str, limit: int) -> None:
    """Deduplicate near-identical tips via LLM-guided synthesis."""
    from fm.consolidator import find_clusters, decide_merge, apply_merge

    store = TipStore(db)
    clusters = find_clusters(store, threshold=threshold)

    if not clusters:
        click.echo("No candidate clusters found — nothing to consolidate.")
        return

    click.echo(f"Found {len(clusters)} candidate cluster(s) at threshold {threshold}.")
    if limit:
        clusters = clusters[:limit]
        click.echo(f"(limited to {limit} clusters)")
    if dry_run:
        click.echo("(dry-run — no changes will be applied)\n")

    merged = 0
    kept = 0
    failed = 0

    for i, cluster in enumerate(clusters, 1):
        ids_preview = ", ".join(t.id[:8] for t in cluster.tips)
        click.echo(f"Cluster {i}/{len(clusters)} ({len(cluster.tips)} tips, max_sim={cluster.max_similarity:.3f}):")
        for tip in cluster.tips:
            click.echo(f"  [{tip.id[:8]}] [{tip.priority}] {tip.category}: {tip.content[:80]}")

        result = decide_merge(cluster, model=model)
        if result is None:
            click.echo(f"  → LLM call failed, skipping.")
            failed += 1
            continue

        if result.action == "keep":
            click.echo(f"  → KEEP SEPARATE: {result.reasoning}")
            kept += 1
        else:
            assert result.canonical_tip is not None
            click.echo(f"  → MERGE: {result.reasoning}")
            click.echo(f"     canonical: [{result.canonical_tip.priority}] {result.canonical_tip.content[:80]}")
            if not dry_run:
                apply_merge(result, cluster, store)
            merged += 1

        click.echo("")

    summary = f"Done. {merged} merged, {kept} kept separate, {failed} failed."
    if dry_run and merged:
        summary += " Run without --dry-run to apply."
    click.echo(summary)


@tips.command("backfill-titles")
@click.option("--db", type=click.Path(path_type=Path), default=_DEFAULT_DB)
@click.option("--model", default="haiku", help="Claude model for title generation.")
def tips_backfill_titles(db: Path, model: str) -> None:
    """Generate titles for tips that don't have one yet."""
    store = TipStore(db)
    untitled = [t for t in store.list_tips() if not t.title]
    if not untitled:
        click.echo("All tips already have titles.")
        return
    click.echo(f"Generating titles for {len(untitled)} tips (model: {model})...")
    updated = 0
    for tip in untitled:
        prompt = (
            f"Generate a 5-8 word imperative title for this tip.\n"
            f"Return ONLY the title — no punctuation, no quotes, no explanation.\n\n"
            f"Tip: {tip.content}"
        )
        raw = call_claude(prompt, model=model)
        if raw is None:
            click.echo(f"  Warning: failed to generate title for {tip.id[:8]}", err=True)
            continue
        title = raw.strip().strip('"').strip("'")[:80]
        store._conn.execute("UPDATE tips SET title = ? WHERE id = ?", (title, tip.id))
        store._conn.commit()
        updated += 1
        click.echo(f"  [{tip.id[:8]}] {title}")
    click.echo(f"Done. {updated}/{len(untitled)} titles generated.")


def _to_aest(utc_iso: str) -> str:
    """Convert a UTC ISO timestamp (from the DB) to AEST (UTC+10) display string."""
    from datetime import datetime, timezone, timedelta
    AEST = timezone(timedelta(hours=10))
    dt = datetime.fromisoformat(utc_iso.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(AEST).strftime("%Y-%m-%d %H:%M")


@main.command("dashboard")
@click.option("--db", type=click.Path(path_type=Path), default=_DEFAULT_DB)
def dashboard(db: Path) -> None:
    """Show a summary dashboard of corpus health and retrieval activity."""
    from rich.console import Console
    from rich.table import Table
    from rich import box
    from rich.text import Text

    store = TipStore(db)
    s = store.get_dashboard_stats()
    console = Console()

    PRIORITY_ORDER = ["critical", "high", "medium", "low"]
    CATEGORY_ORDER = ["strategy", "recovery", "optimization"]

    def _bar(value: int, total: int, width: int = 20) -> str:
        if total == 0:
            return "░" * width
        filled = round(value / total * width)
        return "█" * filled + "░" * (width - filled)

    # ── Corpus ──────────────────────────────────────────────────────────────
    corpus = s["corpus"]
    active = corpus["active"]
    merged = corpus["merged"]
    embedded = corpus["embedded"]

    console.rule("[bold]future_memory dashboard[/bold]")
    console.print()

    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    t.add_column(style="bold cyan", no_wrap=True)
    t.add_column()
    t.add_column(style="dim")

    t.add_row("Active tips", str(active), "")
    t.add_row("Merged (soft-deleted)", str(merged), "")
    embed_pct = f"{embedded/active*100:.0f}%" if active else "—"
    t.add_row("Embedding coverage", f"{embedded}/{active}", embed_pct)

    sessions = s["sessions"]
    yield_pct = f"{sessions['with_tips']/sessions['total']*100:.0f}%" if sessions["total"] else "—"
    t.add_row("Sessions processed", str(sessions["total"]), f"{sessions['with_tips']} yielded tips ({yield_pct})")

    console.print(t)

    # ── By category ─────────────────────────────────────────────────────────
    console.print("[bold]By category[/bold]")
    cat_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    cat_table.add_column(style="cyan", width=14)
    cat_table.add_column(justify="right", width=5)
    cat_table.add_column(width=22)
    cat_table.add_column(style="dim")
    for cat in CATEGORY_ORDER:
        count = corpus["by_category"].get(cat, 0)
        cat_table.add_row(cat, str(count), _bar(count, active), f"{count/active*100:.0f}%" if active else "—")
    console.print(cat_table)

    # ── By priority ─────────────────────────────────────────────────────────
    PRIORITY_STYLE = {"critical": "red", "high": "yellow", "medium": "white", "low": "dim"}
    console.print("[bold]By priority[/bold]")
    pri_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    pri_table.add_column(width=10)
    pri_table.add_column(justify="right", width=5)
    pri_table.add_column(width=22)
    pri_table.add_column(style="dim")
    for pri in PRIORITY_ORDER:
        count = corpus["by_priority"].get(pri, 0)
        style = PRIORITY_STYLE.get(pri, "white")
        pri_table.add_row(
            Text(pri, style=style), str(count),
            _bar(count, active),
            f"{count/active*100:.0f}%" if active else "—",
        )
    console.print(pri_table)

    # ── Retrieval ────────────────────────────────────────────────────────────
    retrieval = s["retrieval"]
    total_ret = retrieval["total"]
    hit_tips = retrieval["tips_with_hits"]
    never_hit = retrieval["never_hit"]
    hit_rate = f"{hit_tips/active*100:.0f}%" if active else "—"

    console.print("[bold]Retrieval[/bold]")
    ret_summary = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    ret_summary.add_column(style="bold cyan", no_wrap=True)
    ret_summary.add_column()
    ret_summary.add_column(style="dim")
    ret_summary.add_row("Total retrievals", str(total_ret), "")
    ret_summary.add_row("Tips with ≥1 hit", f"{hit_tips}/{active}", hit_rate)
    ret_summary.add_row("Never retrieved", str(never_hit), "potential dead weight" if never_hit > active * 0.5 else "")
    console.print(ret_summary)

    if retrieval["top_tips"]:
        console.print("[bold]Top tips[/bold]")
        top_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
        top_table.add_column("Hits", justify="right", style="cyan", width=5)
        top_table.add_column("Avg", justify="right", style="dim", width=5)
        top_table.add_column("Pri", width=8)
        top_table.add_column("Category", width=12)
        top_table.add_column("Trigger", style="dim", max_width=45)
        top_table.add_column("Title")
        for t_row in retrieval["top_tips"]:
            pri = t_row["priority"]
            label = t_row["title"] if t_row["title"] else t_row["content"][:60]
            top_table.add_row(
                str(t_row["hits"]),
                f"{t_row['avg_score']:.2f}",
                Text(pri, style=PRIORITY_STYLE.get(pri, "white")),
                t_row["category"],
                t_row["trigger"],
                label,
            )
        console.print(top_table)

    # ── Recent sessions ──────────────────────────────────────────────────────
    if sessions["recent"]:
        console.print("[bold]Recent extractions[/bold]")
        sess_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        sess_table.add_column(style="dim", width=16)
        sess_table.add_column(justify="right", width=5)
        sess_table.add_column(style="cyan")
        for row in sessions["recent"]:
            date = _to_aest(row["processed_at"])
            path = Path(row["jsonl_path"])
            label = f"{path.parent.name}/{path.stem[:24]}"
            sess_table.add_row(date, str(row["tip_count"]), label)
        console.print(sess_table)

    console.print()


@main.command("telemetry")
@click.option("--db", type=click.Path(path_type=Path), default=_DEFAULT_DB)
def telemetry(db: Path) -> None:
    """Show tip retrieval stats."""
    store = TipStore(db)
    stats = store.get_retrieval_stats()

    click.echo(f"Total retrievals: {stats['total_retrievals']}")
    click.echo(f"Tips never retrieved: {stats['never_retrieved']}")

    if stats["top_tips"]:
        click.echo("\nTop retrieved tips:")
        for t in stats["top_tips"]:
            click.echo(
                f"  [{t['hits']}x, avg {t['avg_score']:.2f}] "
                f"[{t['priority'].upper()} {t['category']}] "
                f"{t['content'][:80]}"
            )

    if stats["recent"]:
        click.echo("\nRecent retrievals:")
        for r in stats["recent"]:
            date = _to_aest(r["retrieved_at"])
            click.echo(
                f"  {date} | score={r['similarity_score']:.2f} | "
                f"{r['content'][:60]}"
            )


@main.group()
def session() -> None:
    """Session-level commands."""
    pass


@session.command("clear-injections")
@click.argument("session_id")
@click.option("--db", type=click.Path(path_type=Path), default=_DEFAULT_DB)
def session_clear_injections(session_id: str, db: Path) -> None:
    """Clear injection cache for a session (called automatically on PreCompact)."""
    store = TipStore(db)
    store.clear_session_injections(session_id)


main.add_command(session)


@main.group()
def db() -> None:
    """Database maintenance commands."""
    pass


@db.command("migrate")
@click.option("--db", type=click.Path(path_type=Path), default=_DEFAULT_DB)
@click.option("--clear-tips", is_flag=True, default=False,
              help="Delete all existing session-level tips and processed_sessions (required before re-extraction with new pipeline).")
def db_migrate(db: Path, clear_tips: bool) -> None:
    """Apply schema migrations to the database."""
    store = TipStore(db)
    store.migrate_add_subtask_columns()
    store.migrate_add_consolidation_columns()
    click.echo("Schema migration complete: subtask columns and consolidation columns added.")

    if clear_tips:
        count = store._conn.execute("SELECT COUNT(*) FROM tips").fetchone()[0]
        store._conn.execute("DELETE FROM tips")
        store._conn.execute("DELETE FROM processed_sessions")
        store._conn.commit()
        click.echo(f"Cleared {count} tips and all processed_sessions. Ready for fresh extraction.")
    else:
        click.echo("Pass --clear-tips to clear old session-level tips and processed_sessions before re-running extract-all.")


main.add_command(db)
main.add_command(tips)


_DEFAULT_BASELINE = Path.home() / ".future_memory" / "baseline.json"
_INJECTION_START = "2026-03-29"


@main.command()
@click.option("--before", default=_INJECTION_START, help="Cutoff date (YYYY-MM-DD). Sessions before this are treated as pre-injection.")
@click.option("--sample", default=500, help="Max sessions to sample (0 = all).")
@click.option("--output", type=click.Path(path_type=Path), default=_DEFAULT_BASELINE, help="Where to save the snapshot JSON.")
def baseline(before: str, sample: int, output: Path) -> None:
    """Capture a baseline metrics snapshot from pre-injection sessions."""
    try:
        cutoff = datetime.strptime(before, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        click.echo(f"Invalid --before date '{before}'. Use YYYY-MM-DD format.", err=True)
        return

    click.echo(f"Computing baseline from sessions before {before} (sample={sample or 'all'})...")

    try:
        snapshot = compute_baseline(before=cutoff, sample=sample)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(snapshot, indent=2))

    m = snapshot["metrics"]
    click.echo(f"Sessions analyzed: {snapshot['sessions_analyzed']} of {snapshot['sessions_available']} available")
    click.echo(f"  Error rate:              {m['error_rate']:.1%}")
    click.echo(f"  Recovery rate:           {m['recovery_rate']:.1%}")
    click.echo(f"  Avg retries after error: {m['avg_retries_after_error']:.1f}")
    click.echo(f"  Avg actions/turn:        {m['avg_actions_per_turn']:.1f}")
    click.echo(f"  Avg turns/session:       {m['avg_turns_per_session']:.1f}")
    click.echo(f"  Session length p50/p90:  {m['session_length_p50']} / {m['session_length_p90']}")
    click.echo(f"  Repeated op rate:        {m['repeated_op_rate']:.1%}")
    click.echo(f"Saved to {output}")
