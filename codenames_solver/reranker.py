from __future__ import annotations

import json

from anthropic import Anthropic

from codenames_solver.config import RERANKER_MODEL
from codenames_solver.solver import ClueSuggestion

_SYSTEM = """\
You are an expert Codenames spymaster evaluator. You understand the game deeply:
a clue word must connect to your team's words through meaning, category, or association,
while being safely unrelated to opponent, neutral, and especially assassin words.\
"""

_USER_TMPL = """\
Board state:
  Team words   (your team must guess these): {targets}
  Avoid words  (opponent + neutral — dangerous if guessed): {avoids}
  Assassin     (instant loss if guessed): {assassins}

Evaluate each candidate clue below. For each, decide:
- Which team words does it connect to? (list only genuine connections, most relevant first)
- Is it dangerously close to any avoid or assassin word?
- Overall quality score: 0–10  (10 = brilliant multi-word clue, 0 = useless or dangerous)

Candidate clues:
{candidates}

Return a JSON array — one object per candidate — with keys:
  "clue"    : the word (lowercase)
  "covers"  : list of team words it connects to
  "score"   : float 0–10
  "reason"  : one sentence

Return ONLY the JSON array, no markdown fences or extra text.\
"""


class LLMReranker:
    def __init__(self, model: str = RERANKER_MODEL) -> None:
        self._client = Anthropic()
        self._model = model

    def rerank(
        self,
        candidates: list[str],
        target_words: list[str],
        avoid_words: list[str],
        assassin_words: list[str],
    ) -> list[ClueSuggestion]:
        if not candidates:
            return []

        target_set = set(target_words)
        prompt = _USER_TMPL.format(
            targets=", ".join(target_words),
            avoids=", ".join(avoid_words) if avoid_words else "none",
            assassins=", ".join(assassin_words) if assassin_words else "none",
            candidates="\n".join(f"- {c}" for c in candidates),
        )

        message = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = message.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        items = json.loads(raw)
        suggestions: list[ClueSuggestion] = []
        for item in items:
            clue = item["clue"].lower()
            covers = [w.lower() for w in item.get("covers", []) if w.lower() in target_set]
            score = float(item.get("score", 0)) / 10.0
            suggestions.append(
                ClueSuggestion(clue=clue, count=len(covers), target_words=covers, score=score)
            )

        suggestions.sort(key=lambda x: (-x.count, -x.score))
        return suggestions
