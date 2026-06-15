Local RAG with MiniLM Embeddings and MCP

A fully local RAG pipeline that uses sentence-transformers (`all-MiniLM-L6-v2`) for
embeddings, Ollama (`tinyllama`) for answers, ChromaDB for storage, and an MCP server
so Cursor can query your documents as tools.

## What is different from the earlier RAG projects?

| Piece | Earlier projects | This project |
|-------|------------------|--------------|
| Embeddings | Ollama `nomic-embed-text` | Hugging Face `all-MiniLM-L6-v2` |
| Chat model | Ollama `tinyllama` | Ollama `tinyllama` |
| Interface | CLI only | CLI + MCP server |
| Vector index | Separate `.chroma/` | Fresh index (384-dim MiniLM vectors) |

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/download) running locally (chat model only)

```powershell
ollama pull tinyllama
```

MiniLM downloads automatically on first ingest (~80 MB via Hugging Face).

## Setup

```powershell
cd C:\KF\repos\projects\local-rag-mcp-minilm
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

First `pip install` may take a few minutes (PyTorch + sentence-transformers).

## CLI usage

```powershell
# Preview recursive chunks (no model download needed for splitting)
python main.py preview data\sample.txt

# Index a document (downloads MiniLM on first run)
python main.py ingest data\sample.txt

# Search without calling the LLM
python main.py search "What embedding model does this project use?"

# Ask a question
python main.py ask "What is recursive chunking?"

# Show retrieved sources with the answer
python main.py ask "What is recursive chunking?" --show-sources

# Check index status
python main.py status

# Clear the vector store
python main.py reset
```

## MCP server (Cursor)

1. Ingest at least one document via CLI first.
2. Add the server to `%USERPROFILE%\.cursor\mcp.json` (see `examples/cursor-mcp.json`).
3. Restart Cursor.
4. Use tools: `search_documents`, `ask_documents`, `rag_status`.

Example Cursor prompts:

- "Call rag_status and tell me what's indexed."
- "Use search_documents to find chunks about MiniLM."
- "Use ask_documents: What embedding model does this project use?"

## Architecture

Full diagram and component breakdown: [docs/architecture.md](docs/architecture.md) · [architecture diagram PNG](docs/local-rag-mcp-architecture-diagram.png)

```
                    ┌─────────────────────────────────────────┐
                    │           rag/service.py                │
                    │     (shared by CLI and MCP tools)       │
                    └─────────────────────────────────────────┘
                           ▲                    ▲
              ┌────────────┘                    └────────────┐
              │                                              │
       main.py CLI                              mcp_server.py (stdio)
   ingest · preview · search · ask          search_documents · ask_documents · rag_status
              │                                              │
              └──────────────────┬───────────────────────────┘
                                 ▼
Document → recursive chunker → MiniLM embed → ChromaDB (.chroma/)
Question → MiniLM search → top-K chunks → [optional] Ollama tinyllama → answer
```

| Path | Embedding | LLM | Entry point |
|------|-----------|-----|-------------|
| Ingest | MiniLM | — | `python main.py ingest` |
| Search | MiniLM | — | CLI `search` or MCP `search_documents` |
| Ask | MiniLM | TinyLlama | CLI `ask` or MCP `ask_documents` |

## Project layout

```
local-rag-mcp-minilm/
├── main.py                 # CLI entry point
├── mcp_server.py           # FastMCP tools for Cursor
├── docs/
│   ├── architecture.md                      # Diagrams, pipelines, component map
│   ├── local-rag-mcp-minilm-article.md      # Medium / LinkedIn article draft
│   ├── local-rag-mcp-architecture-diagram.png
│   └── local-rag-mcp-tools-table.png
├── rag/
│   ├── config.py           # models, chunk size, paths
│   ├── chunker.py          # RecursiveCharacterTextSplitter
│   ├── document_loader.py  # Load .txt / .pdf
│   ├── embedder.py         # all-MiniLM-L6-v2 via sentence-transformers
│   ├── vector_store.py     # ChromaDB
│   ├── query.py            # Retrieve + Ollama generate
│   └── service.py          # Shared logic for CLI + MCP
├── examples/
│   └── cursor-mcp.json     # MCP config snippet
├── data/
│   └── sample.txt
└── .chroma/                # Vector DB (auto-created)
```

## Configuration

Edit `rag/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Hugging Face embedding model |
| `LLM_MODEL` | `tinyllama` | Ollama chat model |
| `CHUNK_SIZE` | `500` | Target chunk length |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K` | `4` | Chunks retrieved per query |

## Notes

- Do not reuse `.chroma/` from projects that used Ollama embeddings — vector dimensions differ.
- `search_documents` only needs MiniLM + ChromaDB; `ask_documents` also needs Ollama running.
- First ingest is slower while MiniLM downloads and loads into memory.
