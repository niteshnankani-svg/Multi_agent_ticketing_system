import chromadb
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="stores/chroma")
collection = client.get_or_create_collection("ticket_cache")

CACHE_DISTANCE_THRESHOLD = 0.25  # tuned below - lower = stricter match

def check_cache(question: str):
    """Returns a cached answer if a near-identical question was answered before, else None."""
    if collection.count() == 0:
        return None
    q_embedding = embedder.encode([question]).tolist()
    results = collection.query(query_embeddings=q_embedding, n_results=1)

    if not results['distances'][0]:
        return None

    distance = results['distances'][0][0]
    if distance < CACHE_DISTANCE_THRESHOLD:
        return results['metadatas'][0][0]['answer']
    return None

def store_in_cache(question: str, answer: str):
    """Saves a question+answer pair so future similar questions can reuse it."""
    embedding = embedder.encode([question]).tolist()
    existing_id = f"cache_{collection.count()}"
    collection.upsert(
        ids=[existing_id],
        documents=[question],
        embeddings=embedding,
        metadatas=[{"answer": answer}],
    )
