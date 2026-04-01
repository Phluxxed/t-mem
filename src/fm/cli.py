from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from fm.embeddings import embed_texts_batch, get_available_provider
from fm.extractor import extract_tips_from_session
from fm.parser import parse_session
from fm.retriever import format_tips, retrieve_tips
from fm.store import TipStore

_DEFAULT_DB = Path.home() / ".future_memory" / "tips.db"


@click.group()
def main() -> None:
    """future_memory — trajectory-informed tips for self-improving agents."""
    pass


@main.command()
@click.argument("jsonl_path", type=click.Path(exists=True, path_type=Path))
@click.option("--db", type=click.Path(path_type=Path), default=_DEFAULT_DB)
@click.option("--model", default="sonnet", help="Claude model for extraction.")
def extract(jsonl_path: Path, db: Path, model: str) -> None:
    """Extract tips from a Claude Code session log."""
    store = TipStore(db)

    session_id, turns = parse_session(jsonl_path, return_session_id=True)
    if not session_id:
        session_id = jsonl_path.stem

    if store.is_session_processed(session_id):
        click.echo(f"Session {session_id} already processed, skipping.")
        return

    if not turns:
        click.echo("No turns found in session.")
        store.mark_session_processed(session_id, str(jsonl_path), tip_count=0)
        return

    project = jsonl_path.parent.name

    click.echo(f"Extracting tips from {len(turns)} turns (model: {model})...")
    tips = extract_tips_from_session(
        turns, session_id=session_id, project=project, model=model
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

    store.mark_session_processed(session_id, str(jsonl_path), tip_count=len(tips))
    msg = f"Extracted {len(tips)} tips from session {session_id}."
    if missing_embeddings:
        msg += f" Warning: {missing_embeddings}/{len(tips)} tips saved without embeddings (run 'fm tips embed' to backfill)."
    click.echo(msg)


@main.command("extract-all")
@click.option("--db", type=click.Path(path_type=Path), default=_DEFAULT_DB)
@click.option("--model", default="sonnet", help="Claude model for extraction.")
def extract_all(db: Path, model: str) -> None:
    """Process all unprocessed Claude Code sessions."""
    projects_dir = Path.home() / ".claude" / "projects"
    if not projects_dir.exists():
        click.echo("No Claude Code projects directory found.")
        return

    store = TipStore(db)
    total_tips = 0
    processed = 0

    for project_dir in projects_dir.iterdir():
        if not project_dir.is_dir():
            continue
        for jsonl_file in project_dir.glob("*.jsonl"):
            session_id = jsonl_file.stem
            if store.is_session_processed(session_id):
                continue

            click.echo(f"Processing {project_dir.name}/{jsonl_file.name}...")
            session_id, turns = parse_session(jsonl_file, return_session_id=True)
            if not session_id:
                session_id = jsonl_file.stem

            if not turns:
                store.mark_session_processed(session_id, str(jsonl_file), tip_count=0)
                continue

            project = project_dir.name
            tips = extract_tips_from_session(
                turns, session_id=session_id, project=project, model=model
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
                session_id, str(jsonl_file), tip_count=len(tips)
            )
            total_tips += len(tips)
            processed += 1

    click.echo(f"Processed {processed} sessions, extracted {total_tips} tips total.")


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

    store = TipStore(db)
    tips = retrieve_tips(prompt, store, threshold=threshold, top_k=top_k)
    output = format_tips(tips)
    if output:
        click.echo(output)


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
            date = r["retrieved_at"][:16].replace("T", " ")
            click.echo(
                f"  {date} | score={r['similarity_score']:.2f} | "
                f"{r['content'][:60]}"
            )


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
    click.echo("Schema migration complete: subtask_id and subtask_description columns added.")

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
