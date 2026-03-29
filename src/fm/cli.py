from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from fm.embeddings import embed_text, get_available_provider
from fm.extractor import extract_tips_from_session
from fm.models import Tip
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
    for tip in tips:
        embedding_result = embed_text(
            f"{tip.content} {tip.trigger}", provider=provider
        )
        if embedding_result:
            store.add_tip(
                tip,
                embedding=embedding_result.vector,
                embedding_provider=embedding_result.provider,
            )
        else:
            store.add_tip(tip)

    store.mark_session_processed(session_id, str(jsonl_path), tip_count=len(tips))
    click.echo(f"Extracted {len(tips)} tips from session {session_id}.")


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
            for tip in tips:
                embedding_result = embed_text(
                    f"{tip.content} {tip.trigger}", provider=provider
                )
                if embedding_result:
                    store.add_tip(
                        tip,
                        embedding=embedding_result.vector,
                        embedding_provider=embedding_result.provider,
                    )
                else:
                    store.add_tip(tip)

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
def retrieve(query: str, db: Path, threshold: float, top_k: int) -> None:
    """Retrieve relevant tips for a task description."""
    store = TipStore(db)
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
    except (json.JSONDecodeError, KeyError):
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


main.add_command(tips)
