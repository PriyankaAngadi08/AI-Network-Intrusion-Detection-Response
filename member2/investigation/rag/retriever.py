
from investigation.rag.document_loader import load_documents
from investigation.rag.chunker import chunk_documents
from investigation.rag.embeddings import embed_texts
from investigation.rag.vector_store import VectorStore


class CybersecurityRetriever:
    """Load, index, and retrieve cybersecurity knowledge."""

    def __init__(self):
        self.store = None
        self.chunk_count = 0

    def build_index(self):
        """Load documents and build an in-memory search index."""
        documents = load_documents()
        chunks = chunk_documents(documents)

        if not chunks:
            self.store = None
            self.chunk_count = 0
            return

        vectors = embed_texts([chunk["text"] for chunk in chunks])

        self.store = VectorStore(dimension=len(vectors[0]))
        self.store.add(chunks, vectors)
        self.chunk_count = len(chunks)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """Retrieve relevant knowledge chunks for a question."""
        if not query.strip():
            raise ValueError("Query must not be empty")

        if self.store is None:
            self.build_index()

        if self.store is None:
            return []

        query_vector = embed_texts([query])[0]
        return self.store.search(query_vector, top_k=top_k)
