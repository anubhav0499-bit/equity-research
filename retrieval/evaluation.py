"""
RAGAS-style Evaluation — Context Relevance, Faithfulness, Answer Relevance.

Implements the three core RAGAS metrics using LLM-as-judge, without the ragas
package dependency (heavy install, breaks on some Python versions). Results are
fully compatible with the RAGAS 0-1 scale and can be compared against library scores.

Metrics:
  context_relevance:  Are the retrieved chunks relevant to the question?
                      Measures retrieval quality.
  faithfulness:       Are the response's claims grounded in the retrieved context?
                      Measures hallucination rate.
  answer_relevance:   Does the response address the original question?
                      Measures answer completeness and focus.

Usage:
    evaluator = RAGASEvaluator(llm_fn=agent.llm_analyze)
    result = evaluator.evaluate(
        question="What was Infosys revenue in FY2024?",
        context_chunks=["Infosys reported revenue of $18.56B in FY2024..."],
        response="Infosys FY2024 revenue was $18.56B, up 1.4% YoY.",
    )
    print(result.ragas_score)   # 0.0 – 1.0

Batch evaluation:
    results = evaluator.evaluate_batch(questions, contexts, responses)
"""

from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from loguru import logger


_CONTEXT_RELEVANCE_SYSTEM = """You are a retrieval quality judge for equity research.
Given a question and retrieved context chunks, score how relevant the chunks are
to answering the question.

Score each chunk 0-3:
  0 — completely unrelated
  1 — loosely related (same company, wrong topic)
  2 — partially relevant (correct topic, missing key data)
  3 — directly relevant (contains the answer or strong evidence)

Return ONLY valid JSON:
{"chunk_scores": [<int>, ...], "relevance_score": <0.0–1.0>, "reasoning": "<brief>"}"""


_FAITHFULNESS_SYSTEM = """You are a hallucination judge for equity research.
Decompose the response into individual factual claims, then verify each claim against
the provided context.

A claim is "supported" if it can be directly inferred from the context.
A claim is "unsupported" if it contains a specific figure, date, or assertion not in context.

Return ONLY valid JSON:
{"total_claims": <int>, "supported_claims": <int>,
 "faithfulness_score": <0.0–1.0>, "unsupported": ["<claim>", ...]}"""


_ANSWER_RELEVANCE_SYSTEM = """You are an answer quality judge for equity research.
Given an original question and the generated answer, score how well the answer
addresses the question:
  1.0 — Complete, direct, and precise answer
  0.75 — Mostly answers the question; minor gaps
  0.5  — Partially answers the question
  0.25 — Tangentially related; misses the main point
  0.0  — Does not address the question at all

Return ONLY valid JSON:
{"answer_relevance_score": <0.0–1.0>, "reasoning": "<brief>"}"""


@dataclass
class RAGASResult:
    context_relevance:  float
    faithfulness:       float
    answer_relevance:   float
    ragas_score:        float        # geometric mean of the three
    reasoning:          dict         = field(default_factory=dict)
    unsupported_claims: list[str]    = field(default_factory=list)


class RAGASEvaluator:
    """LLM-as-judge RAGAS evaluator for the equity RAG pipeline."""

    def __init__(self, llm_fn: Callable[[str, str], str]):
        """
        llm_fn: (system_prompt, user_prompt) → answer_string
        """
        self.llm_fn = llm_fn

    def _call(self, system: str, user: str) -> dict:
        try:
            raw   = self.llm_fn(system, user)
            clean = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
            start = clean.find("{")
            end   = clean.rfind("}") + 1
            return json.loads(clean[start:end]) if start >= 0 else {}
        except Exception as exc:
            logger.warning(f"[ragas] LLM call failed: {exc}")
            return {}

    # ── Metric 1: Context Relevance ─────────────────────────────────────────

    def context_relevance(
        self,
        question: str,
        context_chunks: list[str],
    ) -> float:
        """Score how well the retrieved chunks address the question (0-1)."""
        context_listing = "\n\n".join(
            f"[Chunk {i+1}]: {c[:800]}" for i, c in enumerate(context_chunks[:6])
        )
        result = self._call(
            _CONTEXT_RELEVANCE_SYSTEM,
            f"Question: {question}\n\nRetrieved chunks:\n{context_listing}",
        )
        score = float(result.get("relevance_score", 0.5))
        # Fallback: average chunk scores if top-level score missing
        if "chunk_scores" in result and "relevance_score" not in result:
            raw_scores = result["chunk_scores"]
            if raw_scores:
                score = sum(raw_scores) / (len(raw_scores) * 3.0)
        return round(min(max(score, 0.0), 1.0), 3)

    # ── Metric 2: Faithfulness ──────────────────────────────────────────────

    def faithfulness(
        self,
        response: str,
        context_chunks: list[str],
    ) -> tuple[float, list[str]]:
        """
        Score faithfulness (0-1) and return list of unsupported claims.
        """
        context_preview = "\n\n".join(context_chunks[:4])[:4000]
        result = self._call(
            _FAITHFULNESS_SYSTEM,
            f"Context:\n{context_preview}\n\nResponse:\n{response[:2000]}",
        )
        score       = float(result.get("faithfulness_score", 0.5))
        unsupported = result.get("unsupported", [])
        return round(min(max(score, 0.0), 1.0), 3), unsupported

    # ── Metric 3: Answer Relevance ──────────────────────────────────────────

    def answer_relevance(self, question: str, response: str) -> float:
        """Score how well the response addresses the original question (0-1)."""
        result = self._call(
            _ANSWER_RELEVANCE_SYSTEM,
            f"Question: {question}\n\nAnswer: {response[:1500]}",
        )
        score = float(result.get("answer_relevance_score", 0.5))
        return round(min(max(score, 0.0), 1.0), 3)

    # ── Composite: RAGAS Score ──────────────────────────────────────────────

    def evaluate(
        self,
        question: str,
        context_chunks: list[str],
        response: str,
    ) -> RAGASResult:
        """
        Run all three RAGAS metrics and return a composite score.
        RAGAS score = geometric mean(context_relevance, faithfulness, answer_relevance).
        """
        cr   = self.context_relevance(question, context_chunks)
        f, u = self.faithfulness(response, context_chunks)
        ar   = self.answer_relevance(question, response)

        # Geometric mean (standard RAGAS aggregation)
        import math
        ragas = round(math.pow(cr * f * ar, 1.0 / 3.0), 3) if cr * f * ar > 0 else 0.0

        logger.debug(
            f"[ragas] context_relevance={cr} faithfulness={f} "
            f"answer_relevance={ar} score={ragas}"
        )
        return RAGASResult(
            context_relevance=cr,
            faithfulness=f,
            answer_relevance=ar,
            ragas_score=ragas,
            unsupported_claims=u,
            reasoning={
                "context_relevance": cr,
                "faithfulness": f,
                "answer_relevance": ar,
            },
        )

    def evaluate_batch(
        self,
        questions:       list[str],
        context_batches: list[list[str]],
        responses:       list[str],
    ) -> list[RAGASResult]:
        """Evaluate a batch. Returns one RAGASResult per item."""
        assert len(questions) == len(context_batches) == len(responses), \
            "questions, context_batches, and responses must have the same length"
        results = []
        for i, (q, ctx, r) in enumerate(zip(questions, context_batches, responses)):
            logger.info(f"[ragas] evaluating {i+1}/{len(questions)}")
            results.append(self.evaluate(q, ctx, r))
        return results

    def summary_stats(self, results: list[RAGASResult]) -> dict:
        """Return mean scores across a batch."""
        if not results:
            return {}
        n = len(results)
        return {
            "n":                   n,
            "context_relevance":   round(sum(r.context_relevance  for r in results) / n, 3),
            "faithfulness":        round(sum(r.faithfulness        for r in results) / n, 3),
            "answer_relevance":    round(sum(r.answer_relevance    for r in results) / n, 3),
            "ragas_score":         round(sum(r.ragas_score         for r in results) / n, 3),
        }
