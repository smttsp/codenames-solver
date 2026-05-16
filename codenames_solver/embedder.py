from __future__ import annotations

import numpy as np
from openai import OpenAI

from codenames_solver.config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL


class Embedder:
    def __init__(self, model: str = EMBEDDING_MODEL) -> None:
        self._client = OpenAI()
        self._model = model

    def encode(
        self,
        texts: list[str],
        batch_size: int = EMBEDDING_BATCH_SIZE,
        show_progress: bool = False,
    ) -> np.ndarray:
        all_embeddings: list[list[float]] = []

        ranges = range(0, len(texts), batch_size)
        if show_progress:
            from tqdm import tqdm

            ranges = tqdm(ranges, desc="Embedding", unit="batch")  # type: ignore[assignment]

        for i in ranges:
            batch = texts[i : i + batch_size]
            response = self._client.embeddings.create(input=batch, model=self._model)
            batch_embs = [
                e.embedding for e in sorted(response.data, key=lambda x: x.index)
            ]
            all_embeddings.extend(batch_embs)

        arr = np.array(all_embeddings, dtype=np.float32)
        # Normalize to unit vectors for cosine similarity via dot product
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr /= np.maximum(norms, 1e-8)
        return arr
