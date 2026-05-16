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


class Solver:
    def __init__(self, embedder: Embedder, db: VectorDB) -> None:
        self.embedder = embedder
        self.db = db

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

        # Encode all board words in one API call
        all_board = target_words + avoid_words + assassin_words
        board_embs = self.embedder.encode(all_board)

        n_t = len(target_words)
        n_a = len(avoid_words)
        target_embs = board_embs[:n_t]
        avoid_embs = board_embs[n_t : n_t + n_a]
        assassin_embs = board_embs[n_t + n_a :]

        # Query DB with centroids of all k-subsets of target words
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

        if not candidates:
            return []

        # Batch-fetch all candidate embeddings from DB (one round-trip)
        cand_list = list(candidates)
        found_ids, cand_embs = self.db.get_embeddings(cand_list)

        suggestions: list[ClueSuggestion] = []
        for word, cand_emb in zip(found_ids, cand_embs):
            t_sims = target_embs @ cand_emb  # (n_t,)
            a_sims = avoid_embs @ cand_emb  # (n_a,) or (0,)
            assr_sims = assassin_embs @ cand_emb  # (n_assr,) or (0,)

            danger = 0.0
            if a_sims.size:
                danger = max(danger, float(a_sims.max()) * AVOID_PENALTY)
            if assr_sims.size:
                danger = max(danger, float(assr_sims.max()) * ASSASSIN_PENALTY)

            # Find optimal k: how many targets to claim with this clue
            order = np.argsort(-t_sims)
            best_score = float("-inf")
            best_k = 1
            best_covered = [target_words[int(order[0])]]

            for j in range(1, n_t + 1):
                mean_sim = float(t_sims[order[:j]].mean())
                score = mean_sim - danger
                if score > best_score:
                    best_score = score
                    best_k = j
                    best_covered = [target_words[int(order[i])] for i in range(j)]

            suggestions.append(
                ClueSuggestion(
                    clue=word, count=best_k, target_words=best_covered, score=best_score
                )
            )

        # Primary: more words covered is better; secondary: higher score
        suggestions.sort(key=lambda x: (-x.count, -x.score))
        return suggestions[:max_clues]
