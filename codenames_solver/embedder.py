from __future__ import annotations

import numpy as np
import voyageai

from codenames_solver.config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL

# Voyage AI max inputs per request
_VOYAGE_BATCH_LIMIT = 128


class Embedder:
    def __init__(self, model: str = EMBEDDING_MODEL) -> None:
        self._client = voyageai.Client()
        self._model = model

    def encode(
        self,
        texts: list[str],
        batch_size: int = EMBEDDING_BATCH_SIZE,
        show_progress: bool = False,
    ) -> np.ndarray:
        effective_batch = min(batch_size, _VOYAGE_BATCH_LIMIT)
        all_embeddings: list[list[float]] = []

        ranges = range(0, len(texts), effective_batch)
        if show_progress:
            from tqdm import tqdm

            ranges = tqdm(ranges, desc="Embedding", unit="batch")  # type: ignore[assignment]

        for i in ranges:
            batch = texts[i : i + effective_batch]
            result = self._client.embed(batch, model=self._model, input_type="query")
            all_embeddings.extend(result.embeddings)

        arr = np.array(all_embeddings, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr /= np.maximum(norms, 1e-8)
        return arr
