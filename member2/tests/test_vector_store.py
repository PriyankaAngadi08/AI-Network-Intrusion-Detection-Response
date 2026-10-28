
from investigation.rag.embeddings import embed_texts
from investigation.rag.vector_store import VectorStore

chunks = [
    {
        "source": "network_scanning.md",
        "chunk_id": 1,
        "text": "Port scanning identifies open ports and network services.",
    },
    {
        "source": "dos.md",
        "chunk_id": 2,
        "text": "Denial of service attacks disrupt network availability.",
    },
]

embeddings = embed_texts([chunk["text"] for chunk in chunks])

store = VectorStore(dimension=len(embeddings[0]))
store.add(chunks, embeddings)

query = "How can I identify open network ports?"
query_embedding = embed_texts([query])[0]

results = store.search(query_embedding, top_k=2)

for result in results:
    print(result["source"], result["similarity"], result["text"])

assert len(results) == 2
assert results[0]["source"] == "network_scanning.md"

print("Vector store test passed!")
