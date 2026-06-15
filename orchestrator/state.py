"""
Re-export ResearchState from models for convenience.
LangGraph workflow uses this as the shared state TypedDict.
"""
from ..models.research import ResearchState, AgentOutput, Finding, ValidationResult

__all__ = ["ResearchState", "AgentOutput", "Finding", "ValidationResult"]
