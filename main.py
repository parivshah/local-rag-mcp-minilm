#!/usr/bin/env python3
"""Local RAG with MiniLM embeddings, Ollama chat, and optional MCP server."""

import argparse
import json
import sys
from pathlib import Path

from rag import service
from rag.service import EmptyIndexError, OllamaError


def _print_error(exc: Exception) -> None:
    print(f"Error: {exc}", file=sys.stderr)


def cmd_ingest(args: argparse.Namespace) -> None:
    try:
        result = service.ingest_document(args.document, reset=args.reset)
    except (FileNotFoundError, ValueError, RagError) as exc:
        _print_error(exc)
        sys.exit(1)

    print(f"Ingesting {result['file']} with recursive chunking + MiniLM embeddings...")
    print(f"  Created {result['chunks_created']} chunks")
    if args.reset:
        print("  Cleared existing vector store")
    print(f"  Stored {result['chunks_stored']} vectors (total in DB: {result['total_in_index']})")


def cmd_ask(args: argparse.Namespace) -> None:
    try:
        result = service.ask_documents(args.question, top_k=args.top_k)
    except EmptyIndexError as exc:
        _print_error(exc)
        sys.exit(1)
    except OllamaError as exc:
        _print_error(exc)
        sys.exit(1)

    status = service.get_index_status()
    print(f"Asking (index has {status['chunk_count']} chunks)...\n")
    print(result["answer"])

    if args.show_sources and result["sources"]:
        print("\nSources:")
        for source in result["sources"]:
            print(f"  [{source['rank']}] {source['source']} (distance: {source['distance']})")
            print(f"      {source['excerpt']}")


def cmd_search(args: argparse.Namespace) -> None:
    try:
        result = service.search_documents(args.question, top_k=args.top_k)
    except EmptyIndexError as exc:
        _print_error(exc)
        sys.exit(1)

    print(json.dumps(result, indent=2))


def cmd_preview(args: argparse.Namespace) -> None:
    try:
        result = service.preview_document(args.document, limit=args.limit)
    except FileNotFoundError as exc:
        _print_error(exc)
        sys.exit(1)

    print(f"Recursive split: {result['chunk_count']} chunks from {result['file']}\n")
    for chunk in result["chunks"]:
        print(f"  [{chunk['index']}] ({chunk['chars']} chars) {chunk['preview']}")
    remaining = result["chunk_count"] - result["shown"]
    if remaining > 0:
        print(f"\n  ... and {remaining} more chunks")


def cmd_status(_: argparse.Namespace) -> None:
    result = service.get_index_status()
    print(f"Vector store: {result['chunk_count']} chunks indexed")
    if result["sources"]:
        print("Sources:")
        for source in result["sources"]:
            print(f"  - {source}")


def cmd_reset(_: argparse.Namespace) -> None:
    result = service.reset_index()
    if result["chunks_removed"] == 0:
        print("Vector store is already empty.")
        return
    print(f"Cleared vector store ({result['chunks_removed']} chunks removed).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local RAG with all-MiniLM-L6-v2 embeddings and Ollama chat"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Load a document and store MiniLM vectors")
    ingest.add_argument("document", help="Path to a .txt or .pdf file")
    ingest.add_argument("--reset", action="store_true", help="Clear the vector store before ingesting")
    ingest.set_defaults(func=cmd_ingest)

    preview = sub.add_parser("preview", help="Preview recursive chunks without indexing")
    preview.add_argument("document", help="Path to a .txt or .pdf file")
    preview.add_argument("--limit", type=int, default=5, help="Max chunks to print")
    preview.set_defaults(func=cmd_preview)

    search = sub.add_parser("search", help="Search indexed documents (no LLM)")
    search.add_argument("question", help="Your search query")
    search.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve")
    search.set_defaults(func=cmd_search)

    ask_parser = sub.add_parser("ask", help="Ask a question about ingested documents")
    ask_parser.add_argument("question", help="Your question")
    ask_parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve")
    ask_parser.add_argument("--show-sources", action="store_true", help="Print retrieved sources")
    ask_parser.set_defaults(func=cmd_ask)

    status = sub.add_parser("status", help="Show how many chunks are indexed")
    status.set_defaults(func=cmd_status)

    reset = sub.add_parser("reset", help="Clear all vectors from the store")
    reset.set_defaults(func=cmd_reset)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
