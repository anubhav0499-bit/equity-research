"""
Context Compression — LLM-based extraction + redundancy removal.

After initial retrieval, raw chunks may contain large amounts of irrelevant text
(tables of unrelated figures, boilerplate disclaimers, prior-period comparisons
that don't answer the query). Compression extracts only the directly relevant
passages, reducing context noise and fitting more signal into the LLM's window.

Two stages:
  1. Relevance filtering — drop chunks whose keyword overlap with the query is < threshold
  2. LLM extraction — for each remaining chunk, ask the LLM to extract only the
                       sentences directly relevant to the query

Both stages are optional; the compressor degrades gracefully when the LLM is unavailable.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from loguru import logger


_COMPRESS_SYSTEM = """You are a context extraction assistant for equity research.
Given a retrieved passage and a specific analyst question, extract only the sentences
or phrases from the passage that directly address the question.

Rules:
- Copy text verbatim from the passage — do not paraphrase or add information
- If the passage contains no relevant information, respond with exactly: [IRRELEVANT]
- Preserve specific numbers, dates, and company names exactly
- Output only the extracted text, no commentary"""


@dataclass
class CompressionResult:
    compressed_chunks: list[str]
    original_count: int
    retained_count: int
    chars_before: int
    chars_after: int

    @property
    def compression_ratio(self) -> float:
        return 1.0 - (self.chars_after / max(self.chars_before, 1))


class ContextCompressor:
    """
    Compresses retrieved context for a specific query.

    Usage:
        compressor = ContextCompressor(llm_fn=agent.llm_analyze)
        result = compressor.compress(query, chunks, max_output_chars=6000)
        context = "\\n\\n---\\n\\n".join(result.compressed_chunks)
    """

    def __init__(
        self,
        llm_fn: Optional[Callable[[str, str], str]] = None,
        keyword_threshold: float = 0.10,
    ):
        """
        llm_fn:             (system_prompt, user_prompt) → answer_string
                            If None, only keyword filtering is applied.
        keyword_threshold:  Minimum keyword-overlap ratio to keep a chunk.
                            Set to 0.0 to disable filtering.
        """
        self.llm_fn             = llm_fn
        self.keyword_threshold  = keyword_threshold

    # ── Stage 1: keyword relevance filter ──────────────────────────────────

    @staticmethod
    def _keyword_overlap(query: str, chunk: str) -> float:
        terms = set(re.findall(r"\b\w{3,}\b", query.lower()))
        if not terms:
            return 1.0
        chunk_lower = chunk.lower()
        hits = sum(1 for t in terms if t in chunk_lower)
        return hits / len(terms)

    def _filter_by_keywords(self, query: str, chunks: list[str]) -> list[str]:
        if self.keyword_threshold <= 0.0:
            return chunks
        kept = [c for c in chunks
                if self._keyword_overlap(query, c) >= self.keyword_threshold]
        if not kept:
            return chunks[:3]   # always keep at least some chunks
        return kept

    # ── Stage 2: LLM extraction ─────────────────────────────────────────────

    def _extract_relevant(self, query: str, chunk: str) -> str:
        if self.llm_fn is None:
            return chunk
        user_prompt = (
            f"Analyst question: {query}\n\n"
            f"Retrieved passage:\n{chunk[:2000]}\n\n"
            f"Extract only the relevant sentences:"
        )
        try:
            extracted = self.llm_fn(_COMPRESS_SYSTEM, user_prompt)
            extracted = extracted.strip()
            if extracted == "[IRRELEVANT]" or len(extracted) < 20:
                return ""
            return extracted
        except Exception as exc:
            logger.warning(f"[compression] LLM extraction failed ({exc}); returning raw chunk")
            return chunk

    # ── Redundancy removal ─────────────────────────────────────────────────

    @staticmethod
    def _deduplicate(chunks: list[str], similarity_threshold: float = 0.85) -> list[str]:
        """Remove chunks whose first 200 chars are very similar to an earlier chunk."""
        seen: list[str] = []
        unique: list[str] = []
        for chunk in chunks:
            fingerprint = chunk[:200].lower()
            if any(_jaccard(fingerprint, s) > similarity_threshold for s in seen):
                continue
            seen.append(fingerprint)
            unique.append(chunk)
        return unique

    # ── Public API ─────────────────────────────────────────────────────────

    def compress(
        self,
        query: str,
        chunks: list[str],
        max_output_chars: int = 8000,
    ) -> CompressionResult:
        """
        Compress retrieved chunks for the given query.
        Returns a CompressionResult with compressed_chunks and statistics.
        """
        if not chunks:
            return CompressionResult([], 0, 0, 0, 0)

        original_count = len(chunks)
        chars_before = sum(len(c) for c in chunks)

        # Stage 1: keyword filter
        filtered = self._filter_by_keywords(query, chunks)

        # Stage 2: LLM extraction (per chunk)
        extracted: list[str] = []
        for chunk in filtered:
            ex = self._extract_relevant(query, chunk)
            if ex.strip():
                extracted.append(ex)

        if not extracted:
            extracted = filtered[:3]

        # Stage 3: dedup
        deduped = self._deduplicate(extracted)

        # Stage 4: truncate to max_output_chars (keep highest-priority chunks first)
        final: list[str] = []
        total = 0
        for chunk in deduped:
            if total + len(chunk) > max_output_chars:
                break
            final.append(chunk)
            total += len(chunk)

        chars_after = sum(len(c) for c in final)
        logger.debug(
            f"[compression] {original_count}→{len(final)} chunks, "
            f"{chars_before}→{chars_after} chars "
            f"({100*(1-chars_after/max(chars_before,1)):.0f}% reduction)"
        )
        return CompressionResult(
            compressed_chunks=final,
            original_count=original_count,
            retained_count=len(final),
            chars_before=chars_before,
            chars_after=chars_after,
        )


def _jaccard(a: str, b: str) -> float:
    ta = set(a.split())
    tb = set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)
