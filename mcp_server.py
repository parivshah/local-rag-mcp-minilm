"""
MCP server exposing local RAG tools to Cursor / Claude Desktop.

Run directly (for debugging):
    python mcp_server.py

Cursor launches this via ~/.cursor/mcp.json instead.
"""

from mcp.server.fastmcp import FastMCP

from rag import service
from rag.service import EmptyIndexError, OllamaError

mcp = FastMCP("local-rag-mcp-minilm")


def _error(message: str) -> dict:
    return {"ok": False, "error": message}


@mcp.tool()
def search_documents(question: str, top_k: int = 4) -> dict:
    """Search indexed documents and return matching chunks (no LLM answer)."""
    try:
        result = service.search_documents(question, top_k=top_k)
        return {"ok": True, **result}
    except EmptyIndexError as exc:
        return _error(str(exc))
    except Exception as exc:
        return _error(f"Search failed: {exc}")


@mcp.tool()
def ask_documents(question: str, top_k: int = 4) -> dict:
    """Ask a question and get an answer grounded in indexed documents."""
    try:
        result = service.ask_documents(question, top_k=top_k)
        return {"ok": True, **result}
    except EmptyIndexError as exc:
        return _error(str(exc))
    except OllamaError as exc:
        return _error(str(exc))
    except Exception as exc:
        return _error(f"Ask failed: {exc}")


@mcp.tool()
def rag_status() -> dict:
    """Return how many chunks are indexed and which source files exist."""
    try:
        result = service.get_index_status()
        return {"ok": True, **result}
    except Exception as exc:
        return _error(f"Status failed: {exc}")


if __name__ == "__main__":
    mcp.run(transport="stdio")
