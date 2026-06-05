from pathlib import Path

import chromadb

from rag.config import CHROMA_DIR, COLLECTION_NAME, TOP_K
from rag.embedder import embed_texts


class VectorStore:
    def __init__(self, collection_name: str = COLLECTION_NAME):
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_chunks(self, source_path: Path, chunks: list[str]) -> int:
        if not chunks:
            return 0

        embeddings = embed_texts(chunks)
        source = source_path.name
        ids = [f"{source}-{i}" for i in range(len(chunks))]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=[{"source": source, "chunk_index": i} for i in range(len(chunks))],
        )
        return len(chunks)

    def query(self, question: str, top_k: int = TOP_K) -> list[dict]:
        if self.collection.count() == 0:
            return []

        query_embedding = embed_texts([question])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        return [
            {
                "text": doc,
                "source": meta["source"],
                "chunk_index": meta["chunk_index"],
                "distance": dist,
            }
            for doc, meta, dist in zip(documents, metadatas, distances)
        ]

    def count(self) -> int:
        return self.collection.count()

    def list_sources(self) -> list[str]:
        if self.collection.count() == 0:
            return []

        results = self.collection.get(include=["metadatas"])
        sources = {meta["source"] for meta in results["metadatas"]}
        return sorted(sources)

    def reset(self) -> None:
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(name=name)
