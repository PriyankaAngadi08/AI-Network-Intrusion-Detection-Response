
from sentence_transformers import SentenceTransformer

# A small model suitable for a basic semantic-search prototype.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model = None


def get_model():
    """Load the embedding model once and reuse it."""
    global _model

    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)

    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Convert text strings into numerical embedding vectors."""
    if not texts:
        return []

    model = get_model()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return vectors.tolist()
