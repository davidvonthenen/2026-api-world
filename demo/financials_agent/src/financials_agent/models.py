"""Internal contracts for planning, evidence, traces, and verification."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class FinancialPlan:
    """A constrained execution plan produced by policy plus Nemotron."""

    in_scope: bool = True
    intent: str = "financial_analysis"
    company: str = ""
    symbol: str = ""
    needs_filings: bool = False
    needs_market_data: bool = False
    tools: list[str] = field(default_factory=list)
    risk: str = "low"
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Evidence:
    """One bounded source item supplied to the answer model."""

    evidence_id: str
    kind: str
    source: str
    title: str
    content: str
    as_of: str = ""
    source_url: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_content: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        if not include_content:
            payload.pop("content", None)
        return payload

    def prompt_block(self) -> str:
        metadata = ", ".join(
            f"{key}={value}"
            for key, value in self.metadata.items()
            if value not in (None, "", [], {})
        )
        lines = [
            f"[{self.evidence_id}]",
            f"kind: {self.kind}",
            f"source: {self.source}",
            f"title: {self.title}",
        ]
        if self.as_of:
            lines.append(f"as_of: {self.as_of}")
        if self.source_url:
            lines.append(f"source_url: {self.source_url}")
        if metadata:
            lines.append(f"metadata: {metadata}")
        lines.append(f"content:\n{self.content}")
        return "\n".join(lines)


@dataclass
class ToolCallTrace:
    tool: str
    arguments: dict[str, Any]
    ok: bool
    data: dict[str, Any] | None = None
    error: str = ""
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalTrace:
    index: str
    symbol_filter: str
    hit_count: int
    latency_ms: float
    hits: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelCallTrace:
    role: str
    model: str
    endpoint: str
    latency_ms: float
    ok: bool
    error: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationResult:
    approved: bool
    reasons: list[str] = field(default_factory=list)
    cited_ids: list[str] = field(default_factory=list)
    invalid_ids: list[str] = field(default_factory=list)
    model_checked: bool = False
    model_approved: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
