
from investigation.rag.document_loader import load_documents

documents = load_documents()

print("Documents loaded:", len(documents))

for document in documents:
    print("Source:", document["source"])
    print("Preview:", document["text"][:100])

assert any(
    document["source"] == "network_scanning.md"
    for document in documents
)

print("Document loader test passed!")
