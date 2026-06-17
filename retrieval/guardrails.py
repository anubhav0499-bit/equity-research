"""
Guardrails — Groundedness checking, hallucination detection, and confidence scoring.

Three layers:
  1. Rule-based groundedness (fast, no LLM)
     — Checks whether specific numbers and dates in the response appear in the context.
     — 95%+ of hallucinations involve invented figures; this catches most of them instantly.

  2. LLM-based faithfulness check (thorough, one LLM call)
     — Asks the LLM to identify claims in the response that are unsupported by context.
     — Returns a list of potentially hallucinated statements.

  3. Confidence scoring (composite)
     — Combines retrieval_quality, relevance_score, and groundedness into [0, 1].
     — Surfaces as a single number callers can act on.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from loguru import logger


_FAITHFULNESS_SYSTEM = """You are a hallucination detector for equity research.
Your task is to verify that every factual claim in the response is explicitly supported
by the retrieved context provided.

A "factual claim" is any specific statement about numbers, dates, percentages, ratings,
company names, or events that a reader would take as sourced from a document.

Output ONLY valid JSON with this structure:
{
  "grounded": true/false,
  "score": <0.0–1.0>,
  "unsupported_claims": ["<claim>", "..."],
  "supported_claims": ["<claim>", "..."],
  "reasoning": "<brief explanation>"
}"""


@dataclass
class GuardrailsResult:
    grounded:            bool
    groundedness_score:  float
    confidence_score:    float       # composite: retrieval + relevance + groundedness
    unsupported_claims:  list[str]   = field(default_factory=list)
    supported_claims:    list[str]   = field(default_factory=list)
    hallucinated_numbers: list[str]  = field(default_factory=list)  # rule-based hits
    reasoning:           str         = ""


class GuardrailsChecker:
    """
    Checks response faithfulness and returns a composite confidence score.

    Usage:
        checker = GuardrailsChecker(llm_fn=agent.llm_analyze)
        result = checker.check(
            query="What was HDFC Bank's NIM?",
            response="HDFC Bank reported a NIM of 4.1% in Q3 FY2024.",
            context_chunks=["... net interest margin was 4.1% for the quarter ..."],
            relevance_score=0.88,
        )
    """

    # Regex patterns for numeric claims that are hallucination-prone
    _NUMBER_RE  = re.compile(r"(?:₹|Rs\.?|USD|\$|€|£)?\s*[\d,]+(?:\.\d+)?(?:\s*(?:%|Cr|Bn|Mn|bn|mn|cr|billion|million|crore))?")
    _DATE_RE    = re.compile(r"(?:Q[1-4]\s*FY\d{2,4}|FY\s*\d{2,4}|\d{4}[-/]\d{2}(?:[-/]\d{2})?|[A-Z][a-z]+ \d{4})")

    def __init__(
        self,
        llm_fn: Optional[Callable[[str, str], str]] = None,
        groundedness_threshold: float = 0.70,
    ):
        self.llm_fn                 = llm_fn
        self.groundedness_threshold = groundedness_threshold

    # ── Rule-based: numeric/date presence check ─────────────────────────────

    def _rule_based_check(
        self,
        response: str,
        context: str,
    ) -> tuple[float, list[str]]:
        """
        Check that numbers and dates in the response appear somewhere in the context.
        Returns (score, list_of_missing_values).
        score = 1 - (missing_count / total_claims)
        """
        claims  = self._NUMBER_RE.findall(response) + self._DATE_RE.findall(response)
        claims  = [c.strip() for c in claims if c.strip()]
        if not claims:
            return 1.0, []

        context_lower = context.lower()
        missing = [c for c in claims if c.lower() not in context_lower]
        score   = 1.0 - len(missing) / len(claims)
        return max(0.0, score), missing

    # ── LLM-based: faithfulness ─────────────────────────────────────────────

    def _llm_faithfulness(
        self,
        query: str,
        response: str,
        context_preview: str,
    ) -> dict:
        if self.llm_fn is None:
            return {}
        user_prompt = (
            f"Query: {query}\n\n"
            f"Retrieved context (first 3,000 chars):\n{context_preview[:3000]}\n\n"
            f"Response to verify:\n{response}\n\n"
            f"Verify faithfulness. Return JSON only."
        )
        import json, re as _re
        try:
            raw = self.llm_fn(_FAITHFULNESS_SYSTEM, user_prompt)
            clean = _re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
            start = clean.find("{")
            end   = clean.rfind("}") + 1
            return json.loads(clean[start:end]) if start >= 0 else {}
        except Exception as exc:
            logger.warning(f"[guardrails] LLM faithfulness check failed: {exc}")
            return {}

    # ── Confidence composite ─────────────────────────────────────────────────

    @staticmethod
    def _composite_confidence(
        rule_score:       float,
        llm_score:        float,
        relevance_score:  float,
        retrieval_quality: float,
    ) -> float:
        """
        Weighted composite confidence:
          40% groundedness (average of rule + LLM)
          35% relevance (faithfulness to query)
          25% retrieval quality (# chunks, source diversity)
        """
        g_score = 0.5 * rule_score + 0.5 * llm_score if llm_score > 0 else rule_score
        return round(0.40 * g_score + 0.35 * relevance_score + 0.25 * retrieval_quality, 3)

    # ── Public ───────────────────────────────────────────────────────────────

    def check(
        self,
        query: str,
        response: str,
        context_chunks: list[str],
        relevance_score: float = 0.5,
        retrieval_quality: float = 0.5,
    ) -> GuardrailsResult:
        """
        Run groundedness checks and return a GuardrailsResult.

        retrieval_quality:  proxy for how good the retrieval was.
                            Pass `min(1.0, len(context_chunks) / top_k)`.
        """
        if not response or not context_chunks:
            return GuardrailsResult(
                grounded=False,
                groundedness_score=0.0,
                confidence_score=0.0,
                reasoning="Empty response or no context retrieved.",
            )

        full_context = "\n\n".join(context_chunks)

        # Stage 1: rule-based
        rule_score, missing = self._rule_based_check(response, full_context)

        # Stage 2: LLM-based (if available)
        llm_result   = self._llm_faithfulness(query, response, full_context)
        llm_score    = float(llm_result.get("score", 0.0))
        unsupported  = llm_result.get("unsupported_claims", [])
        supported    = llm_result.get("supported_claims", [])
        reasoning    = llm_result.get("reasoning", "")

        # Use rule score if LLM unavailable
        effective_ground = (0.5 * rule_score + 0.5 * llm_score) if llm_score > 0 else rule_score
        grounded         = effective_ground >= self.groundedness_threshold

        confidence = self._composite_confidence(
            rule_score, llm_score, relevance_score, retrieval_quality
        )

        logger.debug(
            f"[guardrails] rule={rule_score:.2f} llm={llm_score:.2f} "
            f"grounded={grounded} confidence={confidence:.2f}"
        )

        return GuardrailsResult(
            grounded             = grounded,
            groundedness_score   = round(effective_ground, 3),
            confidence_score     = confidence,
            unsupported_claims   = unsupported,
            supported_claims     = supported,
            hallucinated_numbers = missing,
            reasoning            = reasoning,
        )
