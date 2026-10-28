
from investigation.rag.document_loader import load_documents
from investigation.rag.chunker import chunk_documents

documents = load_documents()
chunks = chunk_documents(documents, chunk_size=200, overlap=30)

print("Documents:", len(documents))
print("Chunks:", len(chunks))

for chunk in chunks:
    print("\nChunk ID:", chunk["chunk_id"])
    print("Source:", chunk["source"])
    print("Text:", chunk["text"][:80])

assert len(chunks) > 0
assert all(chunk["text"] for chunk in chunks)
assert all(chunk["source"] for chunk in chunks)

print("\nText chunker test passed!")
