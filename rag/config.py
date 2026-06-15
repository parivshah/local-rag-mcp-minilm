from pathlib import Path

EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384
LLM_MODEL = "llama3"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 4
MAX_EXCERPT_CHARS = 500

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_ROOT / ".chroma"
DATA_DIR = PROJECT_ROOT / "data"
COLLECTION_NAME = "minilm_chunks"
