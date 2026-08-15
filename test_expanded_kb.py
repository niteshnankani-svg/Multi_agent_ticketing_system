import chromadb
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="stores/chroma")

# a question our hand-written doc definitely does NOT cover
questions = {
    "finance": "I was charged in the wrong currency, can this be fixed?",
    "backend": "The mobile app freezes when I switch networks.",
}

for domain, q in questions.items():
    collection = client.get_or_create_collection(f"kb_{domain}")
    q_embedding = embedder.encode([q]).tolist()
    results = collection.query(query_embeddings=q_embedding, n_results=2)

    print(f"DOMAIN: {domain}")
    print(f"Q: {q}")
    for doc, dist in zip(results['documents'][0], results['distances'][0]):
        print(f"   [{dist:.3f}] {doc[:100]}")
    print()
