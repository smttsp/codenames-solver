from __future__ import annotations

import chromadb
import numpy as np
from tqdm import tqdm

from codenames_solver.config import CHROMA_DIR, COLLECTION_NAME


class VectorDB:
    def __init__(self, persist_dir: str | None = None) -> None:
        path = persist_dir or str(CHROMA_DIR)
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=path)
        self._col = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self._col.count()

    def upsert(self, words: list[str], embeddings: np.ndarray, batch_size: int = 5000) -> None:
        batches = range(0, len(words), batch_size)
        for i in tqdm(batches, desc="Pushing to DB", unit="batch"):
            batch_words = words[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size].tolist()
            self._col.upsert(
                ids=batch_words,
                embeddings=batch_embeddings,
                documents=batch_words,
            )

    def query(
        self,
        embedding: np.ndarray,
        n_results: int = 50,
        exclude: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        total = self._col.count()
        if total == 0:
            return []
        extra = len(exclude) if exclude else 0
        fetch = min(n_results + extra, total)

        results = self._col.query(
            query_embeddings=[embedding.tolist()],
            n_results=fetch,
            include=["documents", "distances"],
        )

        items: list[tuple[str, float]] = []
        for word, dist in zip(results["documents"][0], results["distances"][0]):
            if exclude and word in exclude:
                continue
            # ChromaDB cosine distance = 1 - cosine_similarity
            items.append((word, 1.0 - float(dist)))
            if len(items) >= n_results:
                break
        return items

    def existing_ids(self, words: list[str], batch_size: int = 5000) -> set[str]:
        found: set[str] = set()
        for i in range(0, len(words), batch_size):
            batch = words[i : i + batch_size]
            result = self._col.get(ids=batch, include=[])
            found.update(result["ids"])
        return found

    def get_embeddings(self, words: list[str]) -> tuple[list[str], np.ndarray]:
        result = self._col.get(ids=words, include=["embeddings"])
        found_ids: list[str] = result["ids"]
        embeddings = np.array(result["embeddings"], dtype=np.float32)
        return found_ids, embeddings
