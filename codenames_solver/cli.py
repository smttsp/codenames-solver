from __future__ import annotations

import click
import nltk

from codenames_solver.config import (
    EMBEDDING_BATCH_SIZE,
    MAX_WORD_LEN,
    MIN_TRAINING_FREQ,
    MIN_WORD_LEN,
)
from codenames_solver.corpus import word_freq
from codenames_solver.embedder import Embedder
from codenames_solver.solver import Solver
from codenames_solver.vectordb import VectorDB


def _create_solver(rerank: bool = False) -> Solver:
    db = VectorDB()
    if db.count() == 0:
        raise click.ClickException("Vector DB is empty. Run `codenames train` first.")
    reranker = None
    if rerank:
        from codenames_solver.reranker import LLMReranker
        reranker = LLMReranker()
    return Solver(Embedder(), db, reranker=reranker)


def _print_suggestions(suggestions: list, /) -> None:
    if not suggestions:
        click.echo("No suggestions found.")
        return
    click.echo("\nTop clues:")
    for i, s in enumerate(suggestions, 1):
        covered = ", ".join(s.target_words)
        click.echo(
            f"  {i}. {s.clue.upper():20s} ({s.count})  →  {covered}   [score: {s.score:.3f}]"
        )


@click.group()
def cli() -> None:
    """Codenames Solver — AI-powered spymaster clue suggestions."""


@cli.command()
@click.option("--force", is_flag=True, help="Re-embed all words even if already in DB.")
@click.option("--limit", default=35000, show_default=True, help="Max words to train on (0 = no limit).")
def train(force: bool, limit: int) -> None:
    """Embed the English word corpus and persist it to the local vector DB."""
    db = VectorDB()

    click.echo("Downloading NLTK corpora...")
    nltk.download("words", quiet=True)
    from nltk.corpus import words as nltk_words

    # Drop proper nouns: the NLTK words corpus capitalises them, so we filter
    # before lowercasing. Also enforce length bounds.
    valid_words = {
        w.lower()
        for w in nltk_words.words()
        if w.isalpha() and w[0].islower() and MIN_WORD_LEN <= len(w) <= MAX_WORD_LEN
    }

    # word_freq() downloads brown/reuters/webtext and builds the FreqDist.
    fdist = word_freq()
    # Frequency floor strips obscure / archaic dictionary residue.
    valid_words = {w for w in valid_words if fdist.get(w, 0) >= MIN_TRAINING_FREQ}
    filtered = sorted(valid_words, key=lambda w: fdist.get(w, 0), reverse=True)

    if limit > 0:
        filtered = filtered[:limit]

    if force:
        new_words = filtered
    else:
        already_in_db = db.existing_ids(filtered)
        new_words = [w for w in filtered if w not in already_in_db]
        if not new_words:
            click.echo(f"All {len(filtered):,} words already in DB. Nothing to do.")
            return
        if already_in_db:
            click.echo(f"Skipping {len(already_in_db):,} existing words, embedding {len(new_words):,} new words...")
        else:
            click.echo(f"Corpus: {len(new_words):,} words. Starting embedding (this may take a few minutes)...")

    embedder = Embedder()
    embeddings = embedder.encode(
        new_words, batch_size=EMBEDDING_BATCH_SIZE, show_progress=True
    )
    db.upsert(new_words, embeddings)

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
@click.option("--rerank", is_flag=True, help="Use LLM reranker for higher quality clues.")
def solve(
    team_words: str,
    avoid_words: str,
    assassin_words: str,
    top: int,
    max_count: int,
    rerank: bool,
) -> None:
    """Find the best clue words for a manually specified board state."""
    targets = [w.strip() for w in team_words.split(",") if w.strip()]
    avoids = [w.strip() for w in avoid_words.split(",") if w.strip()]
    assassins = [w.strip() for w in assassin_words.split(",") if w.strip()]

    if not targets:
        raise click.UsageError("--team requires at least one word.")

    click.echo(f"\nYour words : {', '.join(targets)}")
    if avoids:
        click.echo(f"Avoid      : {', '.join(avoids)}")
    if assassins:
        click.echo(f"Assassin   : {', '.join(assassins)}")
    click.echo()

    solver = _create_solver(rerank=rerank)
    if rerank:
        click.echo("Reranking with LLM...")
    suggestions = solver.suggest(targets, avoids, assassins, max_clues=top, max_count=max_count)
    _print_suggestions(suggestions)


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
@click.option("--rerank", is_flag=True, help="Use LLM reranker for higher quality clues.")
def infer(image: str, team: str, top: int, max_count: int, rerank: bool) -> None:
    """Parse a board screenshot and suggest clues automatically."""
    from codenames_solver.board_parser import BoardParser

    click.echo(f"Parsing screenshot with GPT-4o ({image})...")
    board = BoardParser().parse(image)

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

    click.echo("\nFinding clues...")
    solver = _create_solver(rerank=rerank)
    if rerank:
        click.echo("Reranking with LLM...")
    suggestions = solver.suggest(
        team_words, avoid_words, board.black, max_clues=top, max_count=max_count
    )
    _print_suggestions(suggestions)
