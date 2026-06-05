import shutil
from pathlib import Path

import ollama

from rag.config import DATA_DIR, TOP_K
from rag.document_loader import load_and_chunk, load_text_file
from rag.query import ask, format_sources
from rag.vector_store import VectorStore


class RagError(Exception):
    """Base error for RAG operations."""


class EmptyIndexError(RagError):
    """Raised when the vector store has no indexed chunks."""


class OllamaError(RagError):
    """Raised when Ollama is unavailable for chat generation."""


def _get_store() -> VectorStore:
    return VectorStore()


def _require_index(store: VectorStore) -> None:
    if store.count() == 0:
        raise EmptyIndexError(
            "No documents indexed. Run: python main.py ingest data/sample.txt"
        )


def ingest_document(document_path: str | Path, reset: bool = False) -> dict:
    doc_path = Path(document_path).resolve()
    if not doc_path.exists():
        raise FileNotFoundError(f"File not found: {doc_path}")

    chunks = load_and_chunk(doc_path)
    store = _get_store()

    if reset:
        store.reset()

    count = store.add_chunks(doc_path, chunks)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_DIR / doc_path.name
    if doc_path != dest.resolve():
        shutil.copy2(doc_path, dest)

    return {
        "file": doc_path.name,
        "chunks_created": len(chunks),
        "chunks_stored": count,
        "total_in_index": store.count(),
    }


def search_documents(question: str, top_k: int = TOP_K) -> dict:
    store = _get_store()
    _require_index(store)

    chunks = store.query(question, top_k=top_k)
    return {
        "question": question,
        "top_k": top_k,
        "chunks": format_sources(chunks),
    }


def ask_documents(question: str, top_k: int = TOP_K) -> dict:
    store = _get_store()
    _require_index(store)

    try:
        return ask(question, store, top_k=top_k)
    except Exception as exc:
        message = str(exc).lower()
        if "connection" in message or "refused" in message or "failed to connect" in message:
            raise OllamaError(
                "Ollama is not reachable. Start Ollama and run: ollama pull tinyllama"
            ) from exc
        raise


def get_index_status() -> dict:
    store = _get_store()
    return {
        "chunk_count": store.count(),
        "sources": store.list_sources(),
        "ready": store.count() > 0,
    }


def reset_index() -> dict:
    store = _get_store()
    previous = store.count()
    store.reset()
    return {"chunks_removed": previous, "chunk_count": store.count()}


def preview_document(document_path: str | Path, limit: int = 5) -> dict:
    doc_path = Path(document_path).resolve()
    if not doc_path.exists():
        raise FileNotFoundError(f"File not found: {doc_path}")

    if doc_path.suffix.lower() == ".txt":
        text = load_text_file(doc_path)
    else:
        from rag.document_loader import load_pdf_text

        text = load_pdf_text(doc_path)

    from rag.chunker import chunk_text

    chunks = chunk_text(text)
    previews = []
    for i, chunk in enumerate(chunks[:limit]):
        preview = chunk.replace("\n", " ")
        if len(preview) > 120:
            preview = preview[:117] + "..."
        previews.append({"index": i, "chars": len(chunk), "preview": preview})

    return {
        "file": doc_path.name,
        "chunk_count": len(chunks),
        "shown": len(previews),
        "chunks": previews,
    }
