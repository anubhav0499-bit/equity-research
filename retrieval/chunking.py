"""
Smart Chunking — Semantic, Recursive, and Context-Aware strategies.

Strategy selection:
  recursive   — Hierarchical split on paragraph/sentence/word separators.
                Best for mixed text/table financial data.
  contextual  — Section-header-aware split for structured annual reports & 10-Ks.
                Preserves ITEM headers, MD&A, Risk Factors, etc. in chunk metadata.
  semantic    — Embedding-similarity split: break where adjacent-sentence cosine
                similarity drops below a threshold. Best for dense prose (MD&A,
                earnings narratives). Falls back to recursive if embed_fn absent.
  auto        — Heuristically picks the best strategy based on document signals.
"""

from __future__ import annotations
import math
import re
from typing import Callable, Literal, Optional

from loguru import logger

ChunkingMode = Literal["semantic", "recursive", "contextual", "auto"]

# ── Recursive splitter ──────────────────────────────────────────────────────

_SEPARATORS = [
    "\n## ", "\n# ",        # Markdown headings
    "\nITEM ", "\nPART ",   # 10-K structural headers
    "\n\n",                 # Paragraph breaks
    "\n",                   # Line breaks
    ". ",                   # Sentence ends
    " ",                    # Words
]


def recursive_split(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    separators: Optional[list[str]] = None,
) -> list[str]:
    """Hierarchical recursive character splitting."""
    return _recursive(text.strip(), separators or _SEPARATORS, chunk_size, chunk_overlap)


def _recursive(text: str, seps: list[str], size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text] if text.strip() else []
    sep = seps[0] if seps else " "
    parts = re.split(re.escape(sep), text) if seps else [text]
    chunks: list[str] = []
    current = ""
    for part in parts:
        candidate = current + (sep if current else "") + part
        if len(candidate) <= size:
            current = candidate
        else:
            if current.strip():
                if len(current) > size and len(seps) > 1:
                    chunks.extend(_recursive(current, seps[1:], size, overlap))
                else:
                    chunks.append(current.strip())
            current = part
    if current.strip():
        if len(current) > size and len(seps) > 1:
            chunks.extend(_recursive(current, seps[1:], size, overlap))
        else:
            chunks.append(current.strip())
    # Apply overlap: prepend trailing suffix from the previous chunk
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-overlap:]
            overlapped.append((tail + " " + chunks[i]).strip())
        return overlapped
    return chunks


# ── Contextual splitter (financial document-aware) ──────────────────────────

_SECTION_RE = re.compile(
    r"(?:^|\n)"
    r"(?:"
    r"ITEM\s+\d+[A-Z]?\.[^\n]{0,80}"          # ITEM 1A. Business Overview
    r"|PART\s+[IVX\d]+[\.\s][^\n]{0,60}"       # PART I. Financial Information
    r"|(?:Management['’s]*\s+Discussion)"  # MD&A
    r"|Risk\s+Factors"
    r"|Financial\s+Statements"
    r"|Notes?\s+to\s+(?:the\s+)?Financial"
    r"|\d{1,2}\.\s+[A-Z][A-Za-z ]{3,50}(?=\n)" # 1. REVENUE RECOGNITION
    r")",
    re.IGNORECASE | re.MULTILINE,
)


def contextual_split(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[dict]:
    """
    Split a financial document into section-aware chunks.
    Returns list of dicts with keys 'text' and 'section_header'.
    """
    boundaries = [m.start() for m in _SECTION_RE.finditer(text)]
    if not boundaries:
        return [{"text": c, "section_header": ""} for c in recursive_split(text, chunk_size, chunk_overlap)]

    result: list[dict] = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
        raw_header = text[start : start + 120].split("\n")[1][:80].strip()
        section_text = text[start:end].strip()
        for chunk in recursive_split(section_text, chunk_size, chunk_overlap):
            result.append({"text": chunk, "section_header": raw_header})
    return result


# ── Semantic splitter ───────────────────────────────────────────────────────

def semantic_split(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    similarity_threshold: float = 0.75,
    embed_fn: Optional[Callable[[list[str]], list]] = None,
) -> list[str]:
    """
    Split text at points where embedding similarity between adjacent sentences
    drops below similarity_threshold. Falls back to recursive when embed_fn is
    unavailable or fails.
    """
    if embed_fn is None:
        logger.debug("[chunking] semantic_split: no embed_fn — falling back to recursive")
        return recursive_split(text, chunk_size, chunk_overlap)

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sentences) <= 2:
        return recursive_split(text, chunk_size, chunk_overlap)

    try:
        embeddings = embed_fn(sentences)
    except Exception as exc:
        logger.warning(f"[chunking] semantic embed failed ({exc}); falling back to recursive")
        return recursive_split(text, chunk_size, chunk_overlap)

    def _cosine(a, b) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na  = math.sqrt(sum(x * x for x in a))
        nb  = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb + 1e-9)

    chunks: list[str] = []
    current: list[str] = [sentences[0]]
    current_len = len(sentences[0])

    for i in range(1, len(sentences)):
        sim = _cosine(embeddings[i - 1], embeddings[i])
        s_len = len(sentences[i])
        if sim < similarity_threshold or current_len + s_len > chunk_size:
            chunk_text = " ".join(current)
            if chunk_text.strip():
                chunks.append(chunk_text)
            if chunk_overlap > 0:
                current = [current[-1], sentences[i]]
                current_len = len(current[-2]) + s_len
            else:
                current = [sentences[i]]
                current_len = s_len
        else:
            current.append(sentences[i])
            current_len += s_len

    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if c.strip()]


# ── Auto-mode selector ──────────────────────────────────────────────────────

def auto_detect_strategy(text: str) -> ChunkingMode:
    """
    Heuristically pick the best chunking strategy for a document.
      - Many section headers  → contextual
      - Dense numerical text / short lines → recursive
      - Flowing prose         → semantic
    """
    sample = text[:5000]
    header_hits = len(_SECTION_RE.findall(sample))
    avg_line_len = len(sample) / max(sample.count("\n"), 1)
    word_count = len(sample.split())
    number_density = len(re.findall(r"\b\d+(?:[.,]\d+)*\b", sample)) / max(word_count, 1)

    if header_hits >= 3:
        return "contextual"
    if number_density > 0.15 or avg_line_len < 55:
        return "recursive"
    return "semantic"


# ── SmartChunker ────────────────────────────────────────────────────────────

class SmartChunker:
    """
    Unified interface for all chunking strategies.
    Returns (chunk_texts, metadatas) ready for vector store ingestion.
    """

    def __init__(
        self,
        mode: ChunkingMode = "auto",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        embed_fn: Optional[Callable] = None,
        semantic_threshold: float = 0.75,
    ):
        self.mode = mode
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embed_fn = embed_fn
        self.semantic_threshold = semantic_threshold

    def split(
        self,
        text: str,
        base_metadata: Optional[dict] = None,
    ) -> tuple[list[str], list[dict]]:
        """Split text and return (chunk_texts, metadatas)."""
        meta = base_metadata or {}
        mode = self.mode if self.mode != "auto" else auto_detect_strategy(text)
        logger.debug(f"[chunking] strategy={mode} text_len={len(text)}")

        if mode == "contextual":
            items = contextual_split(text, self.chunk_size, self.chunk_overlap)
            texts = [i["text"] for i in items]
            metas = [{**meta, "section_header": i["section_header"],
                      "chunk_strategy": "contextual"} for i in items]
        elif mode == "semantic":
            texts = semantic_split(
                text, self.chunk_size, self.chunk_overlap,
                self.semantic_threshold, self.embed_fn,
            )
            metas = [{**meta, "chunk_strategy": "semantic"} for _ in texts]
        else:
            texts = recursive_split(text, self.chunk_size, self.chunk_overlap)
            metas = [{**meta, "chunk_strategy": "recursive"} for _ in texts]

        pairs = [(t, m) for t, m in zip(texts, metas) if t.strip()]
        if not pairs:
            return [], []
        return [p[0] for p in pairs], [p[1] for p in pairs]
