import os
from dotenv import load_dotenv
import anthropic
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()

client_llm = anthropic.Anthropic()
embedder = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="stores/chroma")
collection = chroma_client.get_or_create_collection("kb_backend")

PROMPT_TEMPLATE = """You are a technical support agent for the Backend/Infrastructure team.
Answer the customer's question using ONLY the information in the context below.
If the context does not contain enough information to answer, say so honestly -
do not guess or make anything up.

CONTEXT:
{context}

CUSTOMER QUESTION:
{question}

Answer in 2-3 sentences, clearly and directly."""

def answer_backend_ticket(question: str) -> str:
    q_embedding = embedder.encode([question]).tolist()
    results = collection.query(query_embeddings=q_embedding, n_results=2)
    context = "\n\n".join(results['documents'][0])

    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    response = client_llm.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

if __name__ == "__main__":
    questions = [
        "I keep getting a 500 error when I try to save.",
        "My file won't upload, it's 30MB.",
        "What's your policy on data encryption at rest?",  # NOT in our KB - watch what happens
    ]
    for q in questions:
        print(f"Q: {q}")
        print(f"A: {answer_backend_ticket(q)}")
        print("-" * 60)
