
from investigation.rag.retriever import CybersecurityRetriever

retriever = CybersecurityRetriever()

results = retriever.retrieve(
    "What should I investigate in a port scanning alert?",
    top_k=3,
)

print("Retrieved chunks:", len(results))

for item in results:
    print("\nSource:", item["source"])
    print("Similarity:", item["similarity"])
    print("Text:", item["text"][:200])

assert len(results) > 0
assert all(item["source"] for item in results)

print("\nRetriever test passed!")

