
def chunk_documents(
    documents: list[dict],
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[dict]:
    """Split documents into overlapping text chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    chunks = []

    for document in documents:
        text = document["text"].strip()
        source = document["source"]

        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))

            chunks.append({
                "source": source,
                "chunk_id": len(chunks) + 1,
                "text": text[start:end],
            })

            if end == len(text):
                break

            start = end - overlap

    return chunks
