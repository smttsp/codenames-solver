from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from anthropic import Anthropic

from codenames_solver.config import RERANKER_MODEL
from codenames_solver.solver import ClueSuggestion

_SYSTEM = """\
You are an expert Codenames spymaster. Two equally important rules:

1. Cover AS MANY team words as possible with one clue.
2. NEVER give a clue that connects to an opponent or neutral word — your team will
   guess it and lose their turn. This is a HARD constraint, not a soft one.
   Examples of forbidden clues: HEAT when FIRE is avoid, OCEAN when PACIFIC is avoid,
   BIRD when PARROT is avoid. If a clue relates to ANY avoid word, score it 0–2.\
"""

_EVAL_TMPL = """\
Board state:
  Team words   (guess these): {targets}
  Avoid words  (MUST NOT connect to these): {avoids}
  Assassin     (instant loss): {assassins}

For each candidate clue:
1. Check FIRST whether it connects to any avoid/assassin word. If yes, score 0–2.
2. Otherwise list all team words it genuinely connects to.
3. Score 0–10: penalise avoid-word proximity heavily, reward covering multiple team words.

Candidates:
{candidates}

Return a JSON array: "clue", "covers" (team words only), "score" (0–10), "reason".
Return ONLY the JSON array.\
"""

_GEN_TMPL = """\
Board state:
  Team words   (guess these): {targets}
  Avoid words  (MUST NOT connect to these): {avoids}
  Assassin     (instant loss): {assassins}

Suggest {n} single-word clues covering AS MANY team words as possible.
STRICT RULE: discard any idea that also connects to an avoid/assassin word.
Think laterally: categories, themes, compound words, cultural references.
Do NOT suggest any word already on the board.

Return a JSON array: "clue", "covers" (team words only), "score" (0–10), "reason".
Return ONLY the JSON array.\
"""


def _parse_items(raw: str, target_set: set[str]) -> list[ClueSuggestion]:
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    items = json.loads(raw)
    suggestions: list[ClueSuggestion] = []
    for item in items:
        clue = item["clue"].lower()
        covers = [w.lower() for w in item.get("covers", []) if w.lower() in target_set]
        raw_score = float(item.get("score", 0)) / 10.0
        boosted = raw_score + (len(covers) - 1) * 0.25
        suggestions.append(ClueSuggestion(clue=clue, count=len(covers), target_words=covers, score=boosted))
    return suggestions


class LLMReranker:
    def __init__(self, model: str = RERANKER_MODEL) -> None:
        self._client = Anthropic()
        self._model = model

    def _call(self, prompt: str) -> str:
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()

    def rerank(
        self,
        candidates: list[str],
        target_words: list[str],
        avoid_words: list[str],
        assassin_words: list[str],
        avoid_embs: np.ndarray | None = None,
        assassin_embs: np.ndarray | None = None,
        get_embedding_fn=None,
        n_generated: int = 40,
    ) -> list[ClueSuggestion]:
        if not candidates and not target_words:
            return []

        target_set = set(target_words)
        board_words = target_set | set(avoid_words) | set(assassin_words)

        avoids_str = ", ".join(avoid_words) if avoid_words else "none"
        assassins_str = ", ".join(assassin_words) if assassin_words else "none"
        targets_str = ", ".join(target_words)

        eval_prompt = _EVAL_TMPL.format(
            targets=targets_str, avoids=avoids_str, assassins=assassins_str,
            candidates="\n".join(f"- {c}" for c in candidates),
        )
        gen_prompt = _GEN_TMPL.format(
            targets=targets_str, avoids=avoids_str, assassins=assassins_str,
            n=n_generated,
        )

        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_eval = pool.submit(self._call, eval_prompt)
            fut_gen = pool.submit(self._call, gen_prompt)
            eval_raw, gen_raw = fut_eval.result(), fut_gen.result()

        all_suggestions = _parse_items(eval_raw, target_set) + _parse_items(gen_raw, target_set)

        # Merge: best score per clue, exclude board words and zero-coverage clues
        merged: dict[str, ClueSuggestion] = {}
        for s in all_suggestions:
            if s.clue in board_words or s.count == 0:
                continue
            if s.clue not in merged or s.score > merged[s.clue].score:
                merged[s.clue] = s

        # Apply embedding-based avoid penalty so the LLM can't smuggle in risky clues.
        # Any clue whose max avoid similarity exceeds its max target similarity is penalised.
        if avoid_embs is not None and avoid_embs.size and get_embedding_fn is not None:
            clue_list = list(merged.keys())
            clue_embs = get_embedding_fn(clue_list)
            for clue, emb in zip(clue_list, clue_embs):
                max_avoid = float((avoid_embs @ emb).max())
                max_assassin = float((assassin_embs @ emb).max()) if (assassin_embs is not None and assassin_embs.size) else 0.0
                danger = max(max_avoid, max_assassin)
                s = merged[clue]
                # Subtract how much the clue exceeds the danger threshold (zero if safe)
                avoid_penalty = max(0.0, danger - 0.4) * 2.0
                merged[clue] = ClueSuggestion(
                    clue=s.clue, count=s.count, target_words=s.target_words,
                    score=s.score - avoid_penalty,
                )

        return sorted(merged.values(), key=lambda x: -x.score)
