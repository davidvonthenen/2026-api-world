"""Typed contracts for routing, specialist calls, and orchestration results."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


RouteName = Literal["news", "financial", "combined", "clarification", "unsupported"]
AgentName = Literal["news", "financial"]


class AgentCallPlan(BaseModel):
    """One allowlisted A2A specialist call proposed by Nemotron."""

    model_config = ConfigDict(extra="forbid")

    agent: AgentName
    request: str = Field(min_length=1, max_length=4000)


class RoutingPlan(BaseModel):
    """Structured routing plan emitted by the orchestration model."""

    model_config = ConfigDict(extra="forbid")

    request_type: RouteName
    company: str = Field(default="", max_length=200)
    company_candidates: list[str] = Field(default_factory=list, max_length=4)
    next_action: Literal[
        "call_agents", "ask_clarification", "respond_unsupported", "finish"
    ]
    agent_calls: list[AgentCallPlan] = Field(default_factory=list, max_length=2)
    done: bool
    clarification_question: str = Field(default="", max_length=500)
    reason: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def validate_route_shape(self) -> "RoutingPlan":
        agents = [call.agent for call in self.agent_calls]
        if len(agents) != len(set(agents)):
            raise ValueError("agent_calls must not contain duplicate specialists")

        expected: dict[str, list[str]] = {
            "news": ["news"],
            "financial": ["financial"],
            "combined": ["news", "financial"],
            "clarification": [],
            "unsupported": [],
        }
        if sorted(agents) != sorted(expected[self.request_type]):
            raise ValueError("agent_calls do not match request_type")

        if self.request_type in {"news", "financial", "combined"}:
            if self.next_action != "call_agents" or self.done:
                raise ValueError("specialist routes must call agents before completion")
            if not self.company.strip():
                raise ValueError("a company is required for specialist routes")
        elif self.request_type == "clarification":
            if self.next_action != "ask_clarification" or not self.done:
                raise ValueError("clarification plans must stop after one question")
            if not self.clarification_question.strip():
                raise ValueError("clarification_question is required")
        elif self.request_type == "unsupported":
            if self.next_action != "respond_unsupported" or not self.done:
                raise ValueError("unsupported plans must return without specialist calls")

        return self


class SpecialistResponse(BaseModel):
    """Normalized response envelope around a specialist's A2A text result."""

    model_config = ConfigDict(extra="forbid")

    agent: AgentName
    status: Literal["completed", "input_required", "failed"]
    summary: str = ""
    facts: list[str] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)
    citations: list[str] = Field(default_factory=list)
    as_of: str = ""
    warnings: list[str] = Field(default_factory=list)
    raw_text: str = ""
    context_id: str = ""
    task_id: str = ""
    latency_ms: float = 0.0


class CompletionDecision(BaseModel):
    """Nemotron's bounded decision about whether orchestration is complete."""

    model_config = ConfigDict(extra="forbid")

    done: bool
    next_action: Literal["finish", "call_news", "call_financial"]
    synthesis_required: bool = False
    synthesis_guidance: str = Field(default="", max_length=1500)
    reason: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def validate_completion_shape(self) -> "CompletionDecision":
        if self.done and self.next_action != "finish":
            raise ValueError("done=true requires next_action=finish")
        if not self.done and self.next_action == "finish":
            raise ValueError("done=false requires a specialist next action")
        if self.synthesis_required and not self.done:
            raise ValueError("synthesis requires a completed specialist workflow")
        if self.synthesis_required and not self.synthesis_guidance.strip():
            raise ValueError("synthesis_guidance is required when synthesis is requested")
        if not self.synthesis_required and self.synthesis_guidance.strip():
            raise ValueError("synthesis_guidance requires synthesis_required=true")
        return self


class ModelCallTrace(BaseModel):
    """Minimal model-call trace retained for the audit record."""

    model_config = ConfigDict(extra="forbid")

    purpose: Literal["routing_plan", "completion_decision", "combined_synthesis"]
    model: str
    endpoint: str
    ok: bool
    latency_ms: float
    attempts: int
    error: str = ""


class PolicyAssessment(BaseModel):
    """Deterministic signals that constrain model-proposed routing."""

    model_config = ConfigDict(extra="forbid")

    route_hint: RouteName
    route_reason: str
    company_candidates: list[str] = Field(default_factory=list)
    history_company_candidates: list[str] = Field(default_factory=list)
    explicit_tickers: list[str] = Field(default_factory=list)
    history_explicit_tickers: list[str] = Field(default_factory=list)
    ticker_company_links: dict[str, list[str]] = Field(default_factory=dict)
    history_ticker_company_links: dict[str, list[str]] = Field(default_factory=dict)
    clarification_count: int = 0
    supported_news: bool = False
    supported_financial: bool = False
    requires_current_financial_data: bool = False
    requires_filing_context: bool = False
    explicit_unsupported: bool = False


class OrchestratorResult(BaseModel):
    """Internal result used to construct the OpenAI-compatible response."""

    model_config = ConfigDict(extra="forbid")

    content: str
    request_id: str
    audit_id: str
    route: RouteName
    company: str = ""
    rounds: int
    specialist_calls: list[SpecialistResponse] = Field(default_factory=list)
    private_company: bool = False
    completion_reason: str = ""
    execution_mode: Literal["none", "sequential", "parallel"] = "none"


__all__ = [
    "AgentCallPlan",
    "AgentName",
    "CompletionDecision",
    "ModelCallTrace",
    "OrchestratorResult",
    "PolicyAssessment",
    "RouteName",
    "RoutingPlan",
    "SpecialistResponse",
]
