import chromadb
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="stores/chroma")
collection = client.get_or_create_collection("kb_finance")

questions = [
    "How long does a refund take?",
    "Can I get money back if I cancel my subscription early?",
    "My card was charged twice, what happens?",
]

for q in questions:
    q_embedding = embedder.encode([q]).tolist()
    results = collection.query(query_embeddings=q_embedding, n_results=2)

    print(f"Q: {q}")
    for doc, dist in zip(results['documents'][0], results['distances'][0]):
        print(f"   [{dist:.3f}] {doc[:80]}")
    print()
