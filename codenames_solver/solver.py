from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

from codenames_solver.config import ASSASSIN_PENALTY, AVOID_DANGER_TOP_K
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
    target: np.ndarray
    avoid: np.ndarray
    assassin: np.ndarray


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

        raw_assassin = float((embs.assassin @ cand_emb).max()) if embs.assassin.size else 0.0
        if embs.avoid.size:
            avoid_sims = embs.avoid @ cand_emb
            avoid_max = float(avoid_sims.max())
            # Lenient threshold for coverage counting: ignore the single closest avoid word
            # so one rogue avoid word doesn't prevent counting legitimate target coverage.
            k = min(AVOID_DANGER_TOP_K, len(avoid_sims))
            avoid_lenient = float(np.partition(avoid_sims, -k)[-k])
        else:
            avoid_max = 0.0
            avoid_lenient = 0.0

        # Coverage threshold is lenient (top-k avoid) to count more covered targets.
        coverage_danger = max(avoid_lenient, raw_assassin)
        # Score penalty uses strict max avoid so risky clues (e.g. HEAT near FIRE) rank lower.
        score_danger = max(avoid_max, raw_assassin)

        order = np.argsort(-t_sims)

        best_k = 0
        for j in range(len(target_words)):
            if float(t_sims[order[j]]) > coverage_danger:
                best_k = j + 1
            else:
                break
        best_k = max(1, best_k)

        best_covered = [target_words[int(order[i])] for i in range(best_k)]
        avg_target_sim = float(t_sims[order[:best_k]].mean())
        assassin_extra = max(0.0, raw_assassin - avoid_max) * ASSASSIN_PENALTY
        best_score = avg_target_sim - score_danger - assassin_extra

        return ClueSuggestion(clue=word, count=best_k, target_words=best_covered, score=best_score)

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

        if self.reranker is not None:
            top_candidates = [s.clue for s in suggestions[:reranker_top_k]]
            suggestions = self.reranker.rerank(
                top_candidates, target_words, avoid_words, assassin_words,
                avoid_embs=embs.avoid,
                assassin_embs=embs.assassin,
                get_embedding_fn=self.embedder.encode,
            )

        return suggestions[:max_clues]
