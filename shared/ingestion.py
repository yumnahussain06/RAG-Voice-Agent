"""
Document ingestion: extract raw text from uploaded files and split it into
overlapping chunks ready for embedding.

Design notes:
- Chunking is word-based (not char-based) so chunk boundaries respect word
  edges, and overlap is small (~10% of chunk size by default) so we don't
  waste embedding budget re-encoding large repeated spans, while still
  preserving enough context continuity across chunk boundaries.
- Splitting first happens on paragraphs, then paragraphs are packed into
  chunks up to CHUNK_SIZE_WORDS, so we don't cut a sentence in half more
  than necessary.
"""

from io import BytesIO
from typing import List

from pypdf import PdfReader
from pypdf.errors import PdfReadError
from docx import Document as DocxDocument
from docx.opc.exceptions import PackageNotFoundError

from shared.config import CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS
from shared.errors import ServiceError


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Dispatch to the right extractor based on file extension."""
    ext = filename.lower().rsplit(".", 1)[-1]

    try:
        if ext == "pdf":
            return _extract_pdf(file_bytes)
        elif ext == "docx":
            return _extract_docx(file_bytes)
        elif ext == "txt":
            return file_bytes.decode("utf-8", errors="ignore")
        else:
            raise ServiceError("File ingestion", f"Unsupported file type: .{ext}")
    except (PdfReadError, PackageNotFoundError) as e:
        raise ServiceError("File ingestion", f"'{filename}' appears to be corrupted or unreadable.", e) from e
    except UnicodeDecodeError as e:
        raise ServiceError("File ingestion", f"'{filename}' could not be decoded as text.", e) from e


def _extract_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n\n".join(pages)


def _extract_docx(file_bytes: bytes) -> str:
    doc = DocxDocument(BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def chunk_text(
    text: str,
    chunk_size_words: int = CHUNK_SIZE_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> List[str]:
    """
    Split text into word-based chunks with a small overlap.

    Paragraphs are packed greedily into a chunk until adding the next
    paragraph would exceed chunk_size_words. If a single paragraph is
    itself longer than chunk_size_words, it's split on plain word
    boundaries. Overlap is carried forward by re-including the last
    `overlap_words` words of the previous chunk at the start of the next.
    """
    if overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be smaller than chunk_size_words")

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    words: List[str] = []
    for p in paragraphs:
        words.extend(p.split())
        words.append("\n")  # paragraph boundary marker

    chunks: List[str] = []
    start = 0
    n = len(words)

    if n == 0:
        return []

    while start < n:
        end = min(start + chunk_size_words, n)
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words).replace(" \n ", "\n").strip()
        if chunk:
            chunks.append(chunk)

        if end == n:
            break

        # advance, keeping a small overlap for context continuity
        start = end - overlap_words

    return chunks
