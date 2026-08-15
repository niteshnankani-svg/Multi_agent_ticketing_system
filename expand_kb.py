import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

df = pd.read_csv('data/raw/train_with_answers.csv')
df = df.dropna(subset=['text'])

embedder = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="stores/chroma")

def make_chunks(question, answer):
    """Splits a long answer into paragraph-sized chunks, each tagged with the question."""
    paragraphs = [p.strip() for p in answer.split('\n') if len(p.strip()) > 20]
    if not paragraphs:
        paragraphs = [answer.strip()]
    return [f"Q: {question[:200]}\nA: {p}" for p in paragraphs]

for category in ['finance', 'backend', 'internal', 'general']:
    subset = df[df['category'] == category].dropna(subset=['answer'])
    subset = subset[subset['answer'].str.len() > 25]   # reject thin/empty answers

    if len(subset) == 0:
        print(f"{category}: 0 usable answers found")
        continue

    subset = subset.sample(min(300, len(subset)), random_state=42)

    all_chunks = []
    for _, row in subset.iterrows():
        all_chunks.extend(make_chunks(row['text'], row['answer']))

    collection = client.get_or_create_collection(f"kb_{category}")
    ids = [f"{category}_realticket_{i}" for i in range(len(all_chunks))]
    embeddings = embedder.encode(all_chunks, show_progress_bar=True).tolist()

    collection.upsert(ids=ids, documents=all_chunks, embeddings=embeddings)
    print(f"{category}: {len(subset)} tickets -> {len(all_chunks)} chunks added to kb_{category}")
