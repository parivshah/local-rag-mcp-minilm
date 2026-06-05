from pathlib import Path

from pypdf import PdfReader

from rag.chunker import chunk_text


def load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_and_chunk(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text = load_pdf_text(path)
    elif suffix == ".txt":
        text = load_text_file(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use .txt or .pdf")

    if not text:
        raise ValueError(f"No text extracted from {path}")

    return chunk_text(text)
