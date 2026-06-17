"""
HyDE — Hypothetical Document Embeddings.

Instead of embedding the raw analyst question, the system generates a hypothetical
document that *would answer* the question, then embeds that document. The resulting
embedding is closer in vector-space to real filing passages than a short question.

This materially improves recall for queries like:
  "What caused the gross margin compression in FY2023?"
  → generates a 100-word hypothetical explanation
  → embedding aligns with real MD&A passages describing margin headwinds

Reference: Gao et al. 2022, "Precise Zero-Shot Dense Retrieval without Relevance Labels"
"""

from __future__ import annotations
from typing import Callable, Optional

from loguru import logger


_HYDE_SYSTEM = """You are an expert equity research analyst. Given a question about a company,
write a concise hypothetical passage (80-120 words) that would appear in the company's filings,
earnings transcripts, or analyst reports and directly answers the question.
Write as if you are extracting this passage from the actual document.
Do not mention that this is hypothetical. Write factual-sounding financial prose only."""


class HyDE:
    """
    Generates a hypothetical document for a query and optionally blends its
    embedding with the original query embedding for enhanced dense retrieval.
    """

    def __init__(
        self,
        llm_fn: Callable[[str, str], str],
        embed_fn: Callable[[list[str]], "np.ndarray"],  # type: ignore[name-defined]
    ):
        """
        llm_fn:   (system_prompt, user_prompt) → answer_string
        embed_fn: (list_of_texts) → float32 numpy array, shape (n, d), L2-normalised
        """
        self.llm_fn   = llm_fn
        self.embed_fn = embed_fn

    def generate(
        self,
        question: str,
        company_name: str = "",
        ticker: str = "",
    ) -> str:
        """Return a hypothetical document passage that would answer the question."""
        company_ctx = f"Company: {company_name} ({ticker})\n" if company_name or ticker else ""
        user_prompt = f"{company_ctx}Question: {question}\n\nHypothetical passage:"
        try:
            hypo = self.llm_fn(_HYDE_SYSTEM, user_prompt)
            logger.debug(f"[hyde] generated {len(hypo)} chars for: {question[:60]}")
            return hypo.strip()
        except Exception as exc:
            logger.warning(f"[hyde] generation failed ({exc}); returning empty")
            return ""

    def embed(
        self,
        question: str,
        company_name: str = "",
        ticker: str = "",
        blend_alpha: float = 0.5,
    ) -> Optional["np.ndarray"]:  # type: ignore[name-defined]
        """
        Return a blended query embedding:
          blend_alpha × embed(original_question)
        + (1 - blend_alpha) × embed(hypothetical_document)

        Both vectors are L2-normalised before blending; the result is re-normalised.
        Returns None on any failure (callers fall back to plain query embedding).
        """
        import numpy as np

        hypo = self.generate(question, company_name, ticker)
        if not hypo:
            return None

        try:
            vecs = self.embed_fn([question, hypo])   # shape (2, d)
            q_vec    = vecs[0]
            hypo_vec = vecs[1]
            blended  = blend_alpha * q_vec + (1.0 - blend_alpha) * hypo_vec
            norm = np.linalg.norm(blended)
            if norm < 1e-9:
                return None
            return (blended / norm).astype(np.float32)
        except Exception as exc:
            logger.warning(f"[hyde] embedding failed ({exc})")
            return None
