import os
from dotenv import load_dotenv
import anthropic
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()  # reads ANTHROPIC_API_KEY from .env into memory

client_llm = anthropic.Anthropic()  # picks up the key automatically
embedder = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="stores/chroma")
collection = chroma_client.get_or_create_collection("kb_finance")

PROMPT_TEMPLATE = """You are a customer support agent for the Finance team.
Answer the customer's question using ONLY the information in the context below.
If the context does not contain enough information to answer, say so honestly -
do not guess or make anything up.

CONTEXT:
{context}

CUSTOMER QUESTION:
{question}

Answer in 2-3 sentences, clearly and directly."""

def answer_finance_ticket(question: str) -> str:
    # 1. find the most relevant chunk(s)
    q_embedding = embedder.encode([question]).tolist()
    results = collection.query(query_embeddings=q_embedding, n_results=2)
    context = "\n\n".join(results['documents'][0])

    # 2. build the prompt with real context, not guesses
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    # 3. ask Claude to answer, grounded in that context
    response = client_llm.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

# ---- test it ----
if __name__ == "__main__":
    questions = [
        "How long does a refund take?",
        "I was charged twice for my subscription, what do I do?",
        "Can you tell me about your data security certifications?",  # NOT in our KB - watch what happens
    ]
    for q in questions:
        print(f"Q: {q}")
        print(f"A: {answer_finance_ticket(q)}")
        print("-" * 60)
