"""Typed contracts used inside the governed News Agent."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class RoutePlan(BaseModel):
    """Bounded routing decision produced by Nemotron or deterministic fallback."""

    model_config = ConfigDict(extra="ignore")

    company_names: list[str] = Field(default_factory=list)
    need_historical: bool = True
    need_realtime: bool = True
    financial_only: bool = False
    contains_financial_request: bool = False
    risk_level: Literal["low", "medium", "high"] = "low"
    reason: str = "Use both bounded news sources."


class EvidenceItem(BaseModel):
    """Normalized evidence from either OpenSearch or Tavily."""

    model_config = ConfigDict(extra="ignore")

    citation_id: str = ""
    source_kind: Literal["historical", "realtime"]
    source_id: str
    title: str
    text: str
    score: float | None = None
    path: str | None = None
    url: str | None = None
    category: str | None = None
    chunk_index: int | None = None
    published_at: str | None = None
    retrieved_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerificationResult(BaseModel):
    """Verifier decision used to release, retry, or withhold an answer."""

    model_config = ConfigDict(extra="ignore")

    approved: bool = False
    issues: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    missing_citations: list[str] = Field(default_factory=list)
    cross_domain_financial: bool = False
    retry_instruction: str = ""
    verifier: str = "deterministic"


class NewsResult(BaseModel):
    """Host-facing result returned by ``NewsAgent.ainvoke``."""

    content: str
    audit_id: str
    request_id: str
    status: str
    verification_status: str
    evidence_count: int = 0


__all__ = ["EvidenceItem", "NewsResult", "RoutePlan", "VerificationResult"]
