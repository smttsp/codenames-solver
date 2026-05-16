from __future__ import annotations

import numpy as np
import voyageai

from codenames_solver.config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL

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
        all_embeddings: list[list[float]] = []

        ranges = range(0, len(texts), batch_size)
        if show_progress:
            from tqdm import tqdm

            ranges = tqdm(ranges, desc="Embedding", unit="batch")  # type: ignore[assignment]

        for i in ranges:
            batch = texts[i : i + batch_size]
            result = self._client.embed(batch, model=self._model)
            all_embeddings.extend(result.embeddings)

        arr = np.array(all_embeddings, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr /= np.maximum(norms, 1e-8)
        return arr
