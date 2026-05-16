from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from codenames_solver.config import ASSASSIN_PENALTY, AVOID_PENALTY
from codenames_solver.embedder import Embedder
from codenames_solver.vectordb import VectorDB


@dataclass
class ClueSuggestion:
    clue: str
    count: int
    target_words: list[str]
    score: float


@dataclass
class _BoardEmbeddings:
    target: np.ndarray
    avoid: np.ndarray
    assassin: np.ndarray


class Solver:
    def __init__(self, embedder: Embedder, db: VectorDB) -> None:
        self.embedder = embedder
        self.db = db

    def _encode_board(
        self,
        target_words: list[str],
        avoid_words: list[str],
        assassin_words: list[str],
    ) -> _BoardEmbeddings:
        all_words = target_words + avoid_words + assassin_words
        embs = self.embedder.encode(all_words)
        n_t, n_a = len(target_words), len(avoid_words)
        return _BoardEmbeddings(
            target=embs[:n_t],
            avoid=embs[n_t : n_t + n_a],
            assassin=embs[n_t + n_a :],
        )

    def _gather_candidates(
        self,
        target_embs: np.ndarray,
        board_words: set[str],
        max_count: int,
        candidates_per_query: int,
    ) -> list[str]:
        n_t = len(target_embs)
        candidates: set[str] = set()
        for k in range(min(max_count, n_t), 0, -1):
            for indices in combinations(range(n_t), k):
                subset = target_embs[list(indices)]
                centroid = subset.mean(axis=0)
                centroid /= np.linalg.norm(centroid)
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
    ) -> ClueSuggestion:
        t_sims = embs.target @ cand_emb

        danger = 0.0
        if embs.avoid.size:
            danger = max(danger, float((embs.avoid @ cand_emb).max()) * AVOID_PENALTY)
        if embs.assassin.size:
            danger = max(danger, float((embs.assassin @ cand_emb).max()) * ASSASSIN_PENALTY)

        order = np.argsort(-t_sims)
        best_score = float("-inf")
        best_k = 1
        best_covered = [target_words[int(order[0])]]

        for j in range(1, len(target_words) + 1):
            score = float(t_sims[order[:j]].mean()) - danger
            if score > best_score:
                best_score = score
                best_k = j
                best_covered = [target_words[int(order[i])] for i in range(j)]

        return ClueSuggestion(clue=word, count=best_k, target_words=best_covered, score=best_score)

    def suggest(
        self,
        target_words: list[str],
        avoid_words: list[str],
        assassin_words: list[str],
        max_clues: int = 5,
        max_count: int = 4,
        candidates_per_query: int = 30,
    ) -> list[ClueSuggestion]:
        target_words = [w.lower() for w in target_words]
        avoid_words = [w.lower() for w in avoid_words]
        assassin_words = [w.lower() for w in assassin_words]
        board_words = set(target_words + avoid_words + assassin_words)

        embs = self._encode_board(target_words, avoid_words, assassin_words)
        candidates = self._gather_candidates(
            embs.target, board_words, max_count, candidates_per_query
        )

        if not candidates:
            return []

        found_ids, cand_embs = self.db.get_embeddings(candidates)
        suggestions = [
            self._score_candidate(word, cand_emb, embs, target_words)
            for word, cand_emb in zip(found_ids, cand_embs)
        ]
        suggestions.sort(key=lambda x: (-x.count, -x.score))
        return suggestions[:max_clues]
