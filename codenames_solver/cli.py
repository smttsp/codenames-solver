from __future__ import annotations

import click
import nltk

from codenames_solver.config import EMBEDDING_BATCH_SIZE, MAX_WORD_LEN, MIN_WORD_LEN
from codenames_solver.embedder import Embedder
from codenames_solver.solver import Solver
from codenames_solver.vectordb import VectorDB


@click.group()
def cli() -> None:
    """Codenames Solver — AI-powered spymaster clue suggestions."""


@cli.command()
@click.option("--force", is_flag=True, help="Re-embed even if DB already has words.")
def train(force: bool) -> None:
    """Embed the English word corpus and persist it to the local vector DB."""
    db = VectorDB()

    if db.count() > 0 and not force:
        click.echo(
            f"DB already contains {db.count():,} words. Pass --force to retrain."
        )
        return

    click.echo("Downloading NLTK words corpus...")
    nltk.download("words", quiet=True)
    from nltk.corpus import words as nltk_words

    raw = nltk_words.words()
    filtered = sorted(
        {
            w.lower()
            for w in raw
            if w.isalpha() and MIN_WORD_LEN <= len(w) <= MAX_WORD_LEN
        }
    )
    click.echo(
        f"Filtered corpus: {len(filtered):,} words. Starting embedding (this may take a few minutes)..."
    )

    embedder = Embedder()
    embeddings = embedder.encode(
        filtered, batch_size=EMBEDDING_BATCH_SIZE, show_progress=True
    )
    db.upsert(filtered, embeddings)

    click.echo(f"Done. DB now contains {db.count():,} words.")


@cli.command()
@click.option(
    "--team", "team_words", required=True, help="Comma-separated words for your team."
)
@click.option(
    "--avoid",
    "avoid_words",
    default="",
    help="Comma-separated opponent + neutral words.",
)
@click.option(
    "--assassin", "assassin_words", default="", help="Comma-separated assassin word(s)."
)
@click.option(
    "--top", default=5, show_default=True, help="Number of clue suggestions to show."
)
@click.option(
    "--max-count",
    default=4,
    show_default=True,
    help="Max words a single clue may cover.",
)
def solve(
    team_words: str,
    avoid_words: str,
    assassin_words: str,
    top: int,
    max_count: int,
) -> None:
    """Find the best clue words for a manually specified board state."""
    targets = [w.strip() for w in team_words.split(",") if w.strip()]
    avoids = [w.strip() for w in avoid_words.split(",") if w.strip()]
    assassins = [w.strip() for w in assassin_words.split(",") if w.strip()]

    if not targets:
        raise click.UsageError("--team requires at least one word.")

    db = VectorDB()
    if db.count() == 0:
        raise click.ClickException("Vector DB is empty. Run `codenames train` first.")

    embedder = Embedder()
    solver = Solver(embedder, db)

    click.echo(f"\nYour words : {', '.join(targets)}")
    if avoids:
        click.echo(f"Avoid      : {', '.join(avoids)}")
    if assassins:
        click.echo(f"Assassin   : {', '.join(assassins)}")
    click.echo()

    suggestions = solver.suggest(
        targets, avoids, assassins, max_clues=top, max_count=max_count
    )

    if not suggestions:
        click.echo("No suggestions found. Try broadening your word list.")
        return

    click.echo("Top clues:")
    for i, s in enumerate(suggestions, 1):
        covered = ", ".join(s.target_words)
        click.echo(
            f"  {i}. {s.clue.upper():20s} ({s.count})  →  {covered}   [score: {s.score:.3f}]"
        )


@cli.command()
@click.option(
    "--image",
    required=True,
    type=click.Path(exists=True),
    help="Path to board screenshot.",
)
@click.option(
    "--team",
    default="blue",
    show_default=True,
    type=click.Choice(["blue", "red"]),
    help="Your team color.",
)
@click.option(
    "--top", default=5, show_default=True, help="Number of clue suggestions to show."
)
@click.option(
    "--max-count",
    default=4,
    show_default=True,
    help="Max words a single clue may cover.",
)
def infer(image: str, team: str, top: int, max_count: int) -> None:
    """Parse a board screenshot and suggest clues automatically."""
    from codenames_solver.board_parser import parse_screenshot

    click.echo(f"Parsing screenshot with GPT-4o ({image})...")
    board = parse_screenshot(image, team=team)

    team_words = board.blue if team == "blue" else board.red
    opponent_words = board.red if team == "blue" else board.blue
    avoid_words = opponent_words + board.other

    click.echo(f"\nYour words ({team}) : {', '.join(team_words) or '—'}")
    click.echo(f"Opponent           : {', '.join(opponent_words) or '—'}")
    click.echo(f"Neutral            : {', '.join(board.other) or '—'}")
    click.echo(f"Assassin           : {', '.join(board.black) or '—'}")

    if not team_words:
        raise click.ClickException(
            "No words found for your team. Check the screenshot or team color."
        )

    db = VectorDB()
    if db.count() == 0:
        raise click.ClickException("Vector DB is empty. Run `codenames train` first.")

    embedder = Embedder()
    solver = Solver(embedder, db)

    click.echo("\nFinding clues...")
    suggestions = solver.suggest(
        team_words, avoid_words, board.black, max_clues=top, max_count=max_count
    )

    if not suggestions:
        click.echo("No suggestions found.")
        return

    click.echo("\nTop clues:")
    for i, s in enumerate(suggestions, 1):
        covered = ", ".join(s.target_words)
        click.echo(
            f"  {i}. {s.clue.upper():20s} ({s.count})  →  {covered}   [score: {s.score:.3f}]"
        )
