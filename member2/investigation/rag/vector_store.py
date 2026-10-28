
import faiss
import numpy as np


class VectorStore:
    """Simple in-memory vector store using FAISS."""

    def __init__(self, dimension: int):
        if dimension <= 0:
            raise ValueError("Dimension must be positive")

        self.index = faiss.IndexFlatIP(dimension)
        self.chunks = []

    def add(self, chunks: list[dict], embeddings: list[list[float]]):
        """Store chunks and their embedding vectors."""
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have one embedding")

        if not chunks:
            return

        vectors = np.asarray(embeddings, dtype="float32")

        if vectors.ndim != 2:
            raise ValueError("Embeddings must be a 2D array")

        if vectors.shape[1] != self.index.d:
            raise ValueError("Embedding dimension does not match index")

        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings contain invalid numbers")

        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_embedding: list[float], top_k: int = 3) -> list[dict]:
        """Return the most similar stored chunks."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        if not self.chunks:
            return []

        query = np.asarray([query_embedding], dtype="float32")

        if query.shape[1] != self.index.d:
            raise ValueError("Query embedding dimension does not match index")

        if not np.isfinite(query).all():
            raise ValueError("Query embedding contains invalid numbers")

        scores, indices = self.index.search(
            query,
            min(top_k, len(self.chunks)),
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            results.append({
                **self.chunks[idx],
                "similarity": float(score),
            })

        return results
