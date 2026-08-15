import chromadb
from sentence_transformers import SentenceTransformer

# ---- 1. load the document ----
with open('data/kb/finance/refund_policy.md') as f:
    doc = f.read()

# ---- 2. split into chunks ----
# simple version: split on blank lines (paragraphs)
chunks = [c.strip() for c in doc.split('\n\n') if c.strip()]
print(f"split into {len(chunks)} chunks:")
for i, c in enumerate(chunks):
    print(f"  [{i}] {c[:60]}...")

# ---- 3. load a small embedding model ----
# this turns text into a list of numbers that capture its meaning
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# ---- 4. create a searchable database, store the chunks ----
client = chromadb.PersistentClient(path="stores/chroma")
collection = client.get_or_create_collection("kb_finance")

embeddings = embedder.encode(chunks).tolist()
ids = [f"finance_refund_{i}" for i in range(len(chunks))]

collection.upsert(
    ids=ids,
    documents=chunks,
    embeddings=embeddings,
)

print(f"\nsaved {len(chunks)} chunks to the finance knowledge base")
