import chromadb
from sentence_transformers import SentenceTransformer

with open('data/kb/general/company_info.md') as f:
    doc = f.read()

chunks = [c.strip() for c in doc.split('\n\n') if c.strip()]
print(f"split into {len(chunks)} chunks:")
for i, c in enumerate(chunks):
    print(f"  [{i}] {c[:60]}...")

embedder = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="stores/chroma")
collection = client.get_or_create_collection("kb_general")

embeddings = embedder.encode(chunks).tolist()
ids = [f"general_info_{i}" for i in range(len(chunks))]

collection.upsert(ids=ids, documents=chunks, embeddings=embeddings)
print(f"\nsaved {len(chunks)} chunks to the general knowledge base")
