
from investigation.rag.embeddings import embed_texts

texts = [
    "A port scan checks network ports.",
    "A denial of service attack disrupts network availability.",
]

vectors = embed_texts(texts)

print("Number of vectors:", len(vectors))
print("Vector dimensions:", len(vectors[0]))

assert len(vectors) == 2
assert len(vectors[0]) > 0
assert len(vectors[0]) == len(vectors[1])

print("Embedding test passed!")
