from sentence_transformers import SentenceTransformer
import chromadb

embedder = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="stores/chroma")
collection = client.get_or_create_collection("ticket_cache")

# store the original question fresh
collection.upsert(
    ids=["debug_1"],
    documents=["How long does a refund take?"],
    embeddings=embedder.encode(["How long does a refund take?"]).tolist(),
    metadatas=[{"answer": "Refunds are processed within 5-7 business days."}],
)

# check the actual distance for a paraphrase
q_embedding = embedder.encode(["How many days does a refund usually take?"]).tolist()
results = collection.query(query_embeddings=q_embedding, n_results=1)

print("actual distance:", results['distances'][0][0])
