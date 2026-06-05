import ollama

from rag.config import LLM_MODEL, MAX_EXCERPT_CHARS
from rag.vector_store import VectorStore

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based only on the provided context.
If the context does not contain enough information, say so clearly.
Keep answers concise."""


def _truncate(text: str, max_chars: int = MAX_EXCERPT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def build_prompt(question: str, chunks: list[dict]) -> str:
    if not chunks:
        return question

    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(f"[{i}] (source: {chunk['source']})\n{chunk['text']}")

    context = "\n\n".join(context_parts)
    return f"""Context:
{context}

Question: {question}

Answer based on the context above:"""


def format_sources(chunks: list[dict]) -> list[dict]:
    sources = []
    for i, chunk in enumerate(chunks, start=1):
        sources.append(
            {
                "rank": i,
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
                "distance": round(chunk["distance"], 4),
                "excerpt": _truncate(chunk["text"]),
            }
        )
    return sources


def generate_answer(question: str, chunks: list[dict]) -> str:
    user_prompt = build_prompt(question, chunks)
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response["message"]["content"]


def ask(question: str, store: VectorStore, top_k: int = 4) -> dict:
    chunks = store.query(question, top_k=top_k)
    answer = generate_answer(question, chunks)
    return {
        "question": question,
        "answer": answer,
        "sources": format_sources(chunks),
    }
