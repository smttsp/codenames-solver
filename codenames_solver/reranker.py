from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from anthropic import Anthropic

from codenames_solver.config import RERANKER_MODEL
from codenames_solver.solver import ClueSuggestion

_SYSTEM = """\
You are an expert Codenames spymaster. Covering MORE team words with one clue is the
primary goal. A clue covering 3 words weakly beats a perfect clue for 1 word.
Think about categories, themes, double meanings, cultural references, compound words,
idiomatic phrases — any genuine link counts.\
"""

_EVAL_TMPL = """\
Board state:
  Team words   (guess these): {targets}
  Avoid words  (opponent + neutral): {avoids}
  Assassin     (instant loss): {assassins}

Evaluate each candidate clue. For each:
1. List ALL team words it genuinely connects to — stretch for indirect links.
2. Flag if it's dangerously close to any avoid/assassin word.
3. Score 0–10: reward both connection strength and number of team words covered.

Candidates:
{candidates}

Return a JSON array with objects: "clue", "covers" (list of team words), "score" (0–10), "reason" (one sentence).
Return ONLY the JSON array.\
"""

_GEN_TMPL = """\
Board state:
  Team words   (guess these): {targets}
  Avoid words  (opponent + neutral): {avoids}
  Assassin     (instant loss): {assassins}

Suggest {n} single-word clues that each connect to AS MANY team words as possible.
Prioritise clues covering 3 or 4 team words. Think laterally: categories, themes,
compound words, cultural references, shared properties.
Do NOT suggest any word from the board.

For each clue return: "clue", "covers" (list of team words), "score" (0–10), "reason" (one sentence).
Return ONLY a JSON array.\
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
        n_generated: int = 20,
    ) -> list[ClueSuggestion]:
        if not candidates and not target_words:
            return []

        target_set = set(target_words)
        board_words = target_set | set(avoid_words) | set(assassin_words)

        avoids_str = ", ".join(avoid_words) if avoid_words else "none"
        assassins_str = ", ".join(assassin_words) if assassin_words else "none"
        targets_str = ", ".join(target_words)

        eval_prompt = _EVAL_TMPL.format(
            targets=targets_str,
            avoids=avoids_str,
            assassins=assassins_str,
            candidates="\n".join(f"- {c}" for c in candidates),
        )
        gen_prompt = _GEN_TMPL.format(
            targets=targets_str,
            avoids=avoids_str,
            assassins=assassins_str,
            n=n_generated,
        )

        # Run both LLM calls in parallel
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_eval = pool.submit(self._call, eval_prompt)
            fut_gen = pool.submit(self._call, gen_prompt)
            eval_raw = fut_eval.result()
            gen_raw = fut_gen.result()

        eval_suggestions = _parse_items(eval_raw, target_set)
        gen_suggestions = _parse_items(gen_raw, target_set)

        # Merge: keep best score per clue, exclude board words
        merged: dict[str, ClueSuggestion] = {}
        for s in eval_suggestions + gen_suggestions:
            if s.clue in board_words:
                continue
            if s.clue not in merged or s.score > merged[s.clue].score:
                merged[s.clue] = s

        return sorted(merged.values(), key=lambda x: -x.score)
