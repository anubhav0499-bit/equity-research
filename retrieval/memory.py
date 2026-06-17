"""
Conversation Memory — session-based history with context window management.

Each session (identified by session_id) maintains a sliding window of
question-answer exchanges. When the history exceeds memory_max_chars,
older turns are compressed into a rolling summary via the LLM.

Usage:
    store = ConversationStore.get()
    store.add_exchange(session_id, question="What is HDFC Bank's NIM?", answer="3.8%...")
    context = store.get_context(session_id, max_chars=3000)

The context string is injected into the RAG pipeline's query_rewriter node
so the LLM can resolve pronouns ("What about the Q2 figure?") and follow-up
questions without re-retrieving already-answered context.
"""

from __future__ import annotations
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from loguru import logger


@dataclass
class Exchange:
    question:  str
    answer:    str
    timestamp: float = field(default_factory=time.time)
    sources:   list[str] = field(default_factory=list)


@dataclass
class SessionMemory:
    session_id:   str
    ticker:       str         = ""
    company_name: str         = ""
    turns:        list[Exchange] = field(default_factory=list)
    summary:      str         = ""   # rolling LLM summary of older turns
    created_at:   float       = field(default_factory=time.time)
    last_access:  float       = field(default_factory=time.time)

    def add(self, question: str, answer: str, sources: Optional[list[str]] = None) -> None:
        self.turns.append(Exchange(question, answer, sources=sources or []))
        self.last_access = time.time()

    def total_chars(self) -> int:
        return (len(self.summary)
                + sum(len(t.question) + len(t.answer) for t in self.turns))

    def get_context(self, max_chars: int = 4000) -> str:
        """
        Return conversation history as a formatted string within max_chars.
        Older turns are dropped first if the budget is exceeded.
        """
        parts: list[str] = []
        if self.summary:
            parts.append(f"[Earlier conversation summary]\n{self.summary}")

        budget = max_chars - len(self.summary)
        for turn in reversed(self.turns):
            snippet = f"Q: {turn.question}\nA: {turn.answer[:500]}"
            if len(snippet) > budget:
                break
            parts.insert(1 if self.summary else 0, snippet)
            budget -= len(snippet)

        return "\n\n".join(parts)

    def compress(
        self,
        llm_fn: Callable[[str, str], str],
        max_chars: int = 4000,
    ) -> None:
        """
        Compress old turns into a rolling summary when total exceeds max_chars.
        Keeps the most recent 3 turns verbatim for immediate context.
        """
        if self.total_chars() <= max_chars:
            return

        keep = 3
        to_compress = self.turns[:-keep] if len(self.turns) > keep else []
        if not to_compress:
            return

        history_text = "\n".join(
            f"Q: {t.question}\nA: {t.answer[:300]}" for t in to_compress
        )
        prompt = (
            f"Summarise this equity research Q&A conversation history in 2-3 sentences, "
            f"preserving key financial figures, dates, and conclusions:\n\n{history_text}"
        )
        try:
            new_summary = llm_fn("You are a concise research assistant.", prompt)
            self.summary = (
                (self.summary + "\n" + new_summary).strip()
                if self.summary
                else new_summary.strip()
            )
            self.turns = self.turns[-keep:]
            logger.debug(
                f"[memory:{self.session_id}] compressed {len(to_compress)} turns "
                f"into rolling summary"
            )
        except Exception as exc:
            logger.warning(f"[memory:{self.session_id}] compression failed: {exc}")


class ConversationStore:
    """Thread-safe singleton registry of all active sessions."""

    _instance: Optional["ConversationStore"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._sessions: dict[str, SessionMemory] = {}
        self._rw_lock = threading.RLock()

    @classmethod
    def get(cls) -> "ConversationStore":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def get_or_create(
        self,
        session_id: str,
        ticker: str = "",
        company_name: str = "",
    ) -> SessionMemory:
        with self._rw_lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionMemory(
                    session_id=session_id,
                    ticker=ticker,
                    company_name=company_name,
                )
            session = self._sessions[session_id]
            session.last_access = time.time()
            return session

    def add_exchange(
        self,
        session_id: str,
        question: str,
        answer: str,
        sources: Optional[list[str]] = None,
        llm_fn: Optional[Callable] = None,
        max_chars: int = 4000,
        max_turns: int = 10,
    ) -> None:
        session = self.get_or_create(session_id)
        with self._rw_lock:
            session.add(question, answer, sources)
            if llm_fn and session.total_chars() > max_chars:
                session.compress(llm_fn, max_chars)
            # Hard cap on turns (no LLM available)
            if len(session.turns) > max_turns:
                session.turns = session.turns[-max_turns:]

    def get_context(self, session_id: str, max_chars: int = 4000) -> str:
        session = self._sessions.get(session_id)
        if not session:
            return ""
        return session.get_context(max_chars)

    def clear(self, session_id: str) -> None:
        with self._rw_lock:
            self._sessions.pop(session_id, None)

    def evict_stale(self, ttl_seconds: int = 3600) -> int:
        """Remove sessions inactive for longer than ttl_seconds. Returns eviction count."""
        now = time.time()
        with self._rw_lock:
            stale = [sid for sid, s in self._sessions.items()
                     if now - s.last_access > ttl_seconds]
            for sid in stale:
                del self._sessions[sid]
        return len(stale)
