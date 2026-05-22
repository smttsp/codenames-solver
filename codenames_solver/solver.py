from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

from codenames_solver.config import (
    ASSASSIN_PENALTY,
    COVERAGE_MARGIN,
    MIN_CANDIDATE_FREQ,
)
from codenames_solver.corpus import word_freq
from codenames_solver.embedder import Embedder
from codenames_solver.vectordb import VectorDB

if TYPE_CHECKING:
    from codenames_solver.reranker import LLMReranker


@dataclass
class ClueSuggestion:
    clue: str
    count: int
    target_words: list[str]
    score: float


@dataclass
class _BoardEmbeddings:
    # Per-target stacks of constituent embeddings ("big ben" -> 2 rows).
    target_groups: list[np.ndarray]
    # One vector per target (mean of constituents, normalised) — used for retrieval.
    target_centroids: np.ndarray
    # All avoid / assassin constituents flattened.
    avoid: np.ndarray
    assassin: np.ndarray


def _is_legal_clue(word: str, board_tokens: set[str]) -> bool:
    """Reject clues that share a substring with any board word token.

    Codenames forbids clues that contain (or are contained in) a board word.
    We split multi-word board entries ("big ben") into tokens and check both
    directions against every token.
    """
    for tok in board_tokens:
        if word == tok or tok in word or word in tok:
            return False
    return True


class Solver:
    def __init__(self, embedder: Embedder, db: VectorDB, reranker: LLMReranker | None = None) -> None:
        self.embedder = embedder
        self.db = db
        self.reranker = reranker

    def _encode_board(
        self,
        target_words: list[str],
        avoid_words: list[str],
        assassin_words: list[str],
    ) -> _BoardEmbeddings:
        target_parts = [w.split() for w in target_words]
        avoid_parts = [w.split() for w in avoid_words]
        assassin_parts = [w.split() for w in assassin_words]

        flat: list[str] = []
        for parts in target_parts + avoid_parts + assassin_parts:
            flat.extend(parts)
        embs = self.embedder.encode(flat)

        idx = 0
        target_groups: list[np.ndarray] = []
        for parts in target_parts:
            target_groups.append(embs[idx : idx + len(parts)])
            idx += len(parts)

        n_avoid = sum(len(p) for p in avoid_parts)
        avoid_embs = embs[idx : idx + n_avoid]
        idx += n_avoid

        n_assassin = sum(len(p) for p in assassin_parts)
        assassin_embs = embs[idx : idx + n_assassin]

        dim = embs.shape[1]
        target_centroids = np.zeros((len(target_words), dim), dtype=np.float32)
        for i, group in enumerate(target_groups):
            c = group.mean(axis=0)
            c /= max(float(np.linalg.norm(c)), 1e-8)
            target_centroids[i] = c

        return _BoardEmbeddings(
            target_groups=target_groups,
            target_centroids=target_centroids,
            avoid=avoid_embs,
            assassin=assassin_embs,
        )

    def _gather_candidates(
        self,
        target_centroids: np.ndarray,
        board_words: set[str],
        max_count: int,
        candidates_per_query: int,
    ) -> list[str]:
        n_t = len(target_centroids)
        candidates: set[str] = set()
        for k in range(min(max_count, n_t), 0, -1):
            for indices in combinations(range(n_t), k):
                subset = target_centroids[list(indices)]
                centroid = subset.mean(axis=0)
                centroid /= max(float(np.linalg.norm(centroid)), 1e-8)
                results = self.db.query(
                    centroid, n_results=candidates_per_query, exclude=board_words
                )
                candidates.update(w for w, _ in results)
        return list(candidates)

    def _score_candidate(
        self,
        word: str,
        cand_emb: np.ndarray,
        embs: _BoardEmbeddings,
        target_words: list[str],
    ) -> ClueSuggestion | None:
        t_sims = np.array(
            [float((g @ cand_emb).max()) for g in embs.target_groups],
            dtype=np.float32,
        )

        avoid_max = float((embs.avoid @ cand_emb).max()) if embs.avoid.size else 0.0
        raw_assassin = (
            float((embs.assassin @ cand_emb).max()) if embs.assassin.size else 0.0
        )
        danger = max(avoid_max, raw_assassin)

        order = np.argsort(-t_sims)
        threshold = danger + COVERAGE_MARGIN
        best_k = 0
        for j in range(len(target_words)):
            if float(t_sims[order[j]]) > threshold:
                best_k = j + 1
            else:
                break

        if best_k == 0:
            return None

        covered = [target_words[int(order[i])] for i in range(best_k)]
        # Sum-based reward so multi-coverage actually wins; subtract per-target
        # danger cost so risky clues don't ride high target sims.
        sum_target_sim = float(t_sims[order[:best_k]].sum())
        assassin_extra = max(0.0, raw_assassin - avoid_max) * ASSASSIN_PENALTY
        score = sum_target_sim - best_k * danger - assassin_extra

        return ClueSuggestion(
            clue=word, count=best_k, target_words=covered, score=score
        )

    def suggest(
        self,
        target_words: list[str],
        avoid_words: list[str],
        assassin_words: list[str],
        max_clues: int = 5,
        max_count: int = 4,
        candidates_per_query: int = 75,
        reranker_top_k: int = 75,
    ) -> list[ClueSuggestion]:
        target_words = [w.lower() for w in target_words]
        avoid_words = [w.lower() for w in avoid_words]
        assassin_words = [w.lower() for w in assassin_words]
        board_words = set(target_words + avoid_words + assassin_words)
        # Token-level set: includes "big" and "ben" separately so the substring
        # filter can match clues that collide with either half.
        board_tokens = {tok for bw in board_words for tok in bw.split()}

        embs = self._encode_board(target_words, avoid_words, assassin_words)
        candidates = self._gather_candidates(
            embs.target_centroids, board_words, max_count, candidates_per_query
        )
        if not candidates:
            return []

        fdist = word_freq()
        candidates = [
            w
            for w in candidates
            if _is_legal_clue(w, board_tokens)
            and fdist.get(w, 0) >= MIN_CANDIDATE_FREQ
        ]
        if not candidates:
            return []

        found_ids, cand_embs = self.db.get_embeddings(candidates)
        suggestions: list[ClueSuggestion] = []
        for word, cand_emb in zip(found_ids, cand_embs):
            s = self._score_candidate(word, cand_emb, embs, target_words)
            if s is not None:
                suggestions.append(s)
        suggestions.sort(key=lambda x: -x.score)

        if self.reranker is not None:
            top_candidates = [s.clue for s in suggestions[:reranker_top_k]]
            suggestions = self.reranker.rerank(
                top_candidates, target_words, avoid_words, assassin_words,
                avoid_embs=embs.avoid,
                assassin_embs=embs.assassin,
                get_embedding_fn=self.embedder.encode,
            )

        return suggestions[:max_clues]
