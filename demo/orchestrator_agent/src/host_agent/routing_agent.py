"""Bounded News/Financial orchestrator for the API World demonstration."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
import uuid

from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import Protocol

from .audit import AuditError, AuditLog
from .config import Settings, load_settings
from .models import (
    AgentCallPlan,
    CompletionDecision,
    ModelCallTrace,
    OrchestratorResult,
    PolicyAssessment,
    RouteName,
    RoutingPlan,
    SpecialistResponse,
)
from .policy_manager import (
    NewsFinancePolicyManager,
    collapse_linked_identities,
    identity_aliases,
    linked_ticker_security_ambiguity,
    normalize_company,
)


LOGGER = logging.getLogger(__name__)


class PlannerGateway(Protocol):
    async def plan(
        self,
        messages: Sequence[object],
        assessment: PolicyAssessment,
        validator: Callable[[RoutingPlan], None],
    ) -> tuple[RoutingPlan, ModelCallTrace]: ...

    async def decide_completion(
        self,
        *,
        route: str,
        company: str,
        user_query: str,
        calls_made: Sequence[SpecialistResponse],
        unused_agents: Sequence[str],
        private_company: bool,
        validator: Callable[[CompletionDecision], None],
    ) -> tuple[CompletionDecision, ModelCallTrace]: ...

    async def synthesize_combined(
        self,
        *,
        company: str,
        user_query: str,
        specialists: Sequence[SpecialistResponse],
        guidance: str,
        validator: Callable[[str], None],
    ) -> tuple[str, ModelCallTrace]: ...

    async def close(self) -> None: ...


class SpecialistGateway(Protocol):
    agent: str
    agent_name: str
    agent_url: str

    async def send_message(
        self,
        task: str,
        *,
        context_id: str | None = None,
    ) -> SpecialistResponse: ...

    async def close(self) -> None: ...


def _message_fields(message: object) -> tuple[str, str]:
    if isinstance(message, dict):
        return str(message.get("role") or ""), str(message.get("content") or "")
    return (
        str(getattr(message, "role", "")),
        str(getattr(message, "content", "") or ""),
    )


def _latest_user_text(messages: Sequence[object]) -> str:
    for message in reversed(messages):
        role, content = _message_fields(message)
        if role == "user":
            return content.strip()
    return ""


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _exception_detail(exc: Exception) -> str:
    trace = getattr(exc, "trace", None)
    detail = (
        trace.error
        if isinstance(trace, ModelCallTrace) and trace.error
        else f"{type(exc).__name__}: {exc}"
    )
    return " ".join(detail.split())[:1200]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _deduplicate_companies(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = normalize_company(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(value.strip())
    return result


def _assessment_identities(
    assessment: PolicyAssessment,
    *,
    history: bool = False,
) -> list[str]:
    if history:
        return collapse_linked_identities(
            assessment.history_company_candidates,
            assessment.history_explicit_tickers,
            assessment.history_ticker_company_links,
        )
    return collapse_linked_identities(
        assessment.company_candidates,
        assessment.explicit_tickers,
        assessment.ticker_company_links,
    )


def _identity_ambiguity(assessment: PolicyAssessment) -> bool:
    current = _assessment_identities(assessment)
    if current:
        return len(current) > 1 or linked_ticker_security_ambiguity(
            assessment.ticker_company_links
        )

    history = _assessment_identities(assessment, history=True)
    return len(history) > 1 or linked_ticker_security_ambiguity(
        assessment.history_ticker_company_links
    )


def _fallback_company(assessment: PolicyAssessment) -> str:
    current = _assessment_identities(assessment)
    if len(current) == 1 and not linked_ticker_security_ambiguity(
        assessment.ticker_company_links
    ):
        return current[0]
    if not current:
        history = _assessment_identities(assessment, history=True)
        if len(history) == 1 and not linked_ticker_security_ambiguity(
            assessment.history_ticker_company_links
        ):
            return history[0]
    return ""


def _clarification_question(assessment: PolicyAssessment) -> str:
    identities = _assessment_identities(assessment)
    links = assessment.ticker_company_links
    if not identities:
        identities = _assessment_identities(assessment, history=True)
        links = assessment.history_ticker_company_links

    if len(identities) == 1 and linked_ticker_security_ambiguity(links):
        company = identities[0]
        securities = identity_aliases(company, links)[1:]
        return (
            f"Which one security should this demo analyze for {company}: "
            + ", ".join(securities)
            + "?"
        )

    if len(identities) > 1:
        shown = identities[:4]
        suffix = f", and {len(identities) - 4} more" if len(identities) > 4 else ""
        return (
            "Which one company or security should this demo analyze: "
            + ", ".join(shown)
            + suffix
            + "?"
        )

    company = _fallback_company(assessment)
    if company and assessment.route_hint == "clarification":
        return (
            f"For {company}, do you want company news, financial information, or both?"
        )
    if not company and assessment.route_hint in {"news", "financial", "combined"}:
        return "Which one company should this demo analyze?"
    return (
        "Which one company should this demo analyze, and do you want news, "
        "financial information, or both?"
    )


def _specialist_request(
    agent: str,
    company: str,
    user_query: str,
    assessment: PolicyAssessment,
) -> str:
    bounded_query = user_query.strip()[:2600]
    if agent == "news":
        return (
            "You are the News specialist.\n"
            f"Company: {company}\n"
            "Use only the News Agent's approved historical and current-news sources.\n"
            "Preserve application-owned citations and the News Agent audit identifier.\n"
            "User request:\n"
            f"{bounded_query}"
        )

    ticker_candidates = (
        assessment.explicit_tickers or assessment.history_explicit_tickers
    )
    user_ticker = ticker_candidates[0] if len(ticker_candidates) == 1 else ""
    return (
        "You are the Financial specialist.\n"
        f"Company: {company}\n"
        f"User-supplied ticker: {user_ticker}\n"
        "Current market data required: "
        f"{str(assessment.requires_current_financial_data).lower()}\n"
        "Filing context required: "
        f"{str(assessment.requires_filing_context).lower()}\n"
        "Use only the Financial Agent's approved filing and Finnhub MCP sources.\n"
        "Resolve a ticker only when the user did not supply one. Preserve citations and "
        "the Financial Agent audit identifier.\n"
        "User request:\n"
        f"{bounded_query}"
    )


def _canonicalize_specialist_requests(
    plan: RoutingPlan,
    assessment: PolicyAssessment,
    user_query: str,
) -> RoutingPlan:
    """Keep model routing while deterministically preserving specialist intent."""

    if plan.request_type not in {"news", "financial", "combined"}:
        return plan

    agents = {
        "news": ["news"],
        "financial": ["financial"],
        "combined": ["news", "financial"],
    }[plan.request_type]
    company = _fallback_company(assessment) or plan.company
    return RoutingPlan(
        request_type=plan.request_type,
        company=company,
        company_candidates=[company],
        next_action="call_agents",
        agent_calls=[
            AgentCallPlan(
                agent=agent,
                request=_specialist_request(
                    agent,
                    company,
                    user_query,
                    assessment,
                ),
            )
            for agent in agents
        ],
        done=False,
        clarification_question="",
        reason=plan.reason,
    )


def _fallback_plan(
    assessment: PolicyAssessment,
    user_query: str,
    *,
    relaxed: bool = False,
) -> RoutingPlan:
    route = assessment.route_hint
    company = _fallback_company(assessment)

    if relaxed and company and route in {"clarification", "unsupported"}:
        if assessment.supported_news and assessment.supported_financial:
            route = "combined"
        elif assessment.supported_financial:
            route = "financial"
        elif assessment.supported_news:
            route = "news"
        else:
            route = "combined"

    if route == "unsupported":
        return RoutingPlan(
            request_type="unsupported",
            company="",
            company_candidates=assessment.company_candidates[:4],
            next_action="respond_unsupported",
            agent_calls=[],
            done=True,
            clarification_question="",
            reason=assessment.route_reason,
        )

    if route == "clarification" or not company:
        candidates = _assessment_identities(assessment)
        if not candidates:
            candidates = _assessment_identities(assessment, history=True)
        return RoutingPlan(
            request_type="clarification",
            company=company,
            company_candidates=_deduplicate_companies(candidates)[:4],
            next_action="ask_clarification",
            agent_calls=[],
            done=True,
            clarification_question=_clarification_question(assessment),
            reason=assessment.route_reason,
        )

    agents = {
        "news": ["news"],
        "financial": ["financial"],
        "combined": ["news", "financial"],
    }[route]
    return RoutingPlan(
        request_type=route,
        company=company,
        company_candidates=[company],
        next_action="call_agents",
        agent_calls=[
            AgentCallPlan(
                agent=agent,
                request=_specialist_request(
                    agent,
                    company,
                    user_query,
                    assessment,
                ),
            )
            for agent in agents
        ],
        done=False,
        clarification_question="",
        reason=assessment.route_reason,
    )


def _planning_assessment(
    assessment: PolicyAssessment,
    *,
    policy_checks_enabled: bool,
) -> PolicyAssessment:
    """Keep deterministic classification as a hint when policy gates are disabled."""
    if policy_checks_enabled:
        return assessment

    company = _fallback_company(assessment)
    if not company or assessment.route_hint in {"news", "financial", "combined"}:
        return assessment.model_copy(update={"explicit_unsupported": False})

    if assessment.supported_news and assessment.supported_financial:
        route: RouteName = "combined"
    elif assessment.supported_financial:
        route = "financial"
    elif assessment.supported_news:
        route = "news"
    else:
        route = "combined"
    return assessment.model_copy(
        update={
            "route_hint": route,
            "route_reason": (
                "Policy checks are disabled; deterministic classification is advisory "
                "and one grounded company may use the bounded specialists."
            ),
            "explicit_unsupported": False,
        }
    )


def _normalize_relaxed_plan(
    plan: RoutingPlan,
    assessment: PolicyAssessment,
    user_query: str,
) -> RoutingPlan:
    """Repair a model plan instead of rejecting it when policy checks are disabled."""
    company = _fallback_company(assessment)
    if not company or _identity_ambiguity(assessment):
        return _fallback_plan(assessment, user_query, relaxed=True)

    if plan.request_type not in {"news", "financial", "combined"}:
        return _fallback_plan(assessment, user_query, relaxed=True)

    agents = {
        "news": ["news"],
        "financial": ["financial"],
        "combined": ["news", "financial"],
    }[plan.request_type]
    return RoutingPlan(
        request_type=plan.request_type,
        company=company,
        company_candidates=[company],
        next_action="call_agents",
        agent_calls=[
            AgentCallPlan(
                agent=agent,
                request=_specialist_request(
                    agent,
                    company,
                    user_query,
                    assessment,
                ),
            )
            for agent in agents
        ],
        done=False,
        clarification_question="",
        reason=plan.reason,
    )


def _unsupported_content() -> str:
    return (
        "This conference demo supports questions about one company's news, AI-related "
        "developments, public-market data, earnings, and financial filings. The request "
        "falls outside those domains."
    )


def _clarification_exhausted_content() -> str:
    return (
        "The demo stopped after one clarification turn because it still could not resolve "
        "one company and one requested domain. Start a new request that names one company "
        "and asks for news, financial information, or both."
    )


def _private_market_notice(company: str) -> str:
    return (
        f"> Public stock-market data is not available for {company} based on the "
        "Financial Agent's symbol-resolution result. The orchestrator did not infer or "
        "substitute a ticker."
    )


def _audit_failure_content() -> str:
    return (
        "The orchestrator stopped because its demonstration audit chain could not "
        "be validated or extended. No specialist result was released."
    )


def _append_audit_warning(content: str) -> str:
    return (
        content.rstrip()
        + "\n\n> **Audit warning:** The response was released, but the orchestrator "
        "could not persist its audit record."
    )


def _audit_plan(plan: RoutingPlan) -> dict[str, object]:
    """Serialize a validated plan without retaining specialist prompt text."""

    return {
        "request_type": plan.request_type,
        "company": plan.company,
        "company_candidates": list(plan.company_candidates),
        "next_action": plan.next_action,
        "agent_calls": [
            {
                "agent": call.agent,
                "request_sha256": _sha256(call.request),
                "request_length": len(call.request),
            }
            for call in plan.agent_calls
        ],
        "done": plan.done,
        "clarification_question_sha256": (
            _sha256(plan.clarification_question) if plan.clarification_question else ""
        ),
        "clarification_question_length": len(plan.clarification_question),
        "reason_sha256": _sha256(plan.reason),
        "reason_length": len(plan.reason),
    }


def _audit_model_trace(trace: ModelCallTrace) -> dict[str, object]:
    """Keep model outcome metadata without retaining validation payload excerpts."""

    value = trace.model_dump(exclude={"error"})
    value["error_type"] = trace.error.split(":", 1)[0] if trace.error else ""
    return value


def _audit_completion(
    completion: CompletionDecision | None,
) -> dict[str, object] | None:
    """Serialize completion control flow without retaining model prose."""

    if completion is None:
        return None
    return {
        "done": completion.done,
        "next_action": completion.next_action,
        "synthesis_required": completion.synthesis_required,
        "synthesis_guidance_sha256": (
            _sha256(completion.synthesis_guidance)
            if completion.synthesis_guidance
            else ""
        ),
        "synthesis_guidance_length": len(completion.synthesis_guidance),
        "reason_sha256": _sha256(completion.reason),
        "reason_length": len(completion.reason),
    }


def _specialist_body(result: SpecialistResponse, display_name: str) -> str:
    if result.raw_text:
        return result.raw_text
    warning = result.warnings[0] if result.warnings else "No terminal text was returned."
    return f"The {display_name} did not return a governed result. {warning}"


_EVIDENCE_LIMIT_MARKERS = (
    "no evidence",
    "could not retrieve",
    "could not verify",
    "outside the",
    "unavailable",
    "withheld",
    "not verified as a publicly traded company",
)

_CITATION_GROUP_RE = re.compile(
    r"\[((?:[A-Z][A-Z0-9_-]*\d+)(?:\s*,\s*[A-Z][A-Z0-9_-]*\d+)*)\]"
)


def _specialist_has_evidence_or_limit(result: SpecialistResponse) -> bool:
    text = result.raw_text.strip()
    if result.citations:
        return True
    if text and any(marker in text.lower() for marker in _EVIDENCE_LIMIT_MARKERS):
        return True
    if result.data.get("audit_id"):
        return True
    return result.status != "completed" and bool(result.warnings)


def _extract_citations(text: str) -> list[str]:
    references: list[str] = []
    for group in _CITATION_GROUP_RE.findall(text):
        references.extend(
            f"[{part.strip()}]" for part in group.split(",") if part.strip()
        )
    references.extend(
        match.rstrip(".,);]")
        for match in re.findall(r"https?://[^\s<>()]+", text)
    )
    return list(dict.fromkeys(references))


def _validate_synthesized_content(
    content: str,
    specialists: Sequence[SpecialistResponse],
) -> None:
    candidate = content.strip()
    if not candidate:
        raise ValueError("The synthesis model returned no content.")

    output_citations = set(_extract_citations(candidate))
    allowed_citations = {
        citation for result in specialists for citation in result.citations
    }
    invented = sorted(output_citations - allowed_citations)
    if invented:
        raise ValueError(
            "The synthesized answer introduced unsupported citations: "
            + ", ".join(invented)
        )

    missing_agents = [
        result.agent
        for result in specialists
        if result.raw_text.strip()
        and result.citations
        and not output_citations.intersection(result.citations)
    ]
    if missing_agents:
        raise ValueError(
            "The synthesized answer omitted citations from: "
            + ", ".join(missing_agents)
        )

def _default_synthesis_guidance(user_query: str) -> str:
    if re.search(
        r"\b(affect(?:ed|s)?|impact(?:ed|s)?|relationship|connect|because|why)\b",
        user_query,
        flags=re.IGNORECASE,
    ):
        return (
            "Integrate the company developments with the financial or market evidence "
            "requested by the user. Explain supported relationships while distinguishing "
            "documented causation from timing, correlation, and inference."
        )
    return (
        "Answer the user's combined request as one integrated narrative. Cover material "
        "developments and financial or market evidence, preserve time boundaries and exact "
        "citations, and disclose conflicts or evidence limits."
    )


def _specialist_audit_line(specialists: Sequence[SpecialistResponse]) -> str:
    values = [
        f"{result.agent}={result.data.get('audit_id')}"
        for result in specialists
        if result.data.get("audit_id")
    ]
    if not values:
        return ""
    return "**Specialist audit IDs:** `" + " | ".join(values) + "`"


def _trace_line(
    *,
    route: str,
    company: str,
    specialists: Sequence[SpecialistResponse],
    execution_mode: str,
    rounds: int,
    audit_id: str,
    policy_status: str,
    composition_mode: str = "none",
) -> str:
    names = ",".join(result.agent for result in specialists) or "none"
    company_value = company or "unresolved"
    return (
        "**Routing trace:** `"
        f"route={route} | company={company_value} | specialists={names} | "
        f"execution={execution_mode} | composition={composition_mode} | "
        f"rounds={rounds} | policy={policy_status} | "
        f"audit_id={audit_id}`"
    )


def _compose_response(
    *,
    route: RouteName,
    company: str,
    specialists: Sequence[SpecialistResponse],
    private_company: bool,
    execution_mode: str,
    rounds: int,
    audit_id: str,
    policy_status: str,
    synthesized_content: str = "",
) -> str:
    by_agent = {result.agent: result for result in specialists}
    sections: list[str] = []
    combine_outputs = route == "combined" or {"news", "financial"}.issubset(
        by_agent
    )

    if combine_outputs:
        if synthesized_content.strip():
            sections.append(synthesized_content.strip())
            if private_company:
                sections.append(_private_market_notice(company))
        else:
            if "news" in by_agent:
                sections.extend(
                    [
                        "## Company developments",
                        _specialist_body(by_agent["news"], "News Agent"),
                    ]
                )
            if "financial" in by_agent:
                sections.append("## Financial and market information")
                if private_company:
                    sections.append(_private_market_notice(company))
                sections.append(
                    _specialist_body(by_agent["financial"], "Financial Agent")
                )
            sections.append(
                "**Composition note:** The synthesis model did not return a validated "
                "combined narrative, so the governed specialist results are shown in "
                "separate sections."
            )
        audit_line = _specialist_audit_line(specialists)
        if audit_line:
            sections.append(audit_line)
    elif "news" in by_agent:
        sections.extend(
            [
                "## Company developments",
                _specialist_body(by_agent["news"], "News Agent"),
            ]
        )

    if not combine_outputs and "financial" in by_agent:
        sections.append("## Financial and market information")
        if private_company:
            sections.append(_private_market_notice(company))
        sections.append(_specialist_body(by_agent["financial"], "Financial Agent"))

    if not sections:
        sections.append(
            "No specialist result was released because the bounded A2A calls did not "
            "produce terminal output."
        )

    sections.extend(
        [
            "---",
            _trace_line(
                route=route,
                company=company,
                specialists=specialists,
                execution_mode=execution_mode,
                rounds=rounds,
                audit_id=audit_id,
                policy_status=policy_status,
                composition_mode=(
                    "synthesized"
                    if synthesized_content.strip()
                    else "specialist_fallback"
                    if combine_outputs
                    else "specialist"
                    if specialists
                    else "none"
                ),
            ),
        ]
    )
    return "\n\n".join(sections)


class RoutingAgent:
    """Apply policy, plan with Nemotron, call specialists through A2A, and audit."""

    def __init__(
        self,
        *,
        settings: Settings,
        planner: PlannerGateway,
        news_connection: SpecialistGateway,
        financial_connection: SpecialistGateway,
        policy_manager: NewsFinancePolicyManager | None = None,
        audit_log: AuditLog | None = None,
    ) -> None:
        self.settings = settings
        self.planner = planner
        self.policy_manager = policy_manager or NewsFinancePolicyManager()
        self.audit_log = audit_log or AuditLog(settings.audit_log_path)
        self.connections: dict[str, SpecialistGateway] = {
            "news": news_connection,
            "financial": financial_connection,
        }
        self._session_context_ids: dict[tuple[str, str, str], str] = {}

    async def _call_specialist(
        self,
        call: AgentCallPlan,
        *,
        session_id: str,
        request_id: str,
        company: str,
    ) -> SpecialistResponse:
        connection = self.connections[call.agent]
        context_key = (session_id, call.agent, normalize_company(company))
        context_id = self._session_context_ids.get(context_key)
        LOGGER.info(
            "specialist_call_started request_id=%s agent=%s url=%s",
            request_id,
            call.agent,
            connection.agent_url,
        )
        result = await connection.send_message(call.request, context_id=context_id)
        if result.status == "failed":
            self._session_context_ids.pop(context_key, None)
        elif result.context_id:
            self._session_context_ids[context_key] = result.context_id
        LOGGER.info(
            "specialist_call_finished request_id=%s agent=%s status=%s latency_ms=%.3f",
            request_id,
            call.agent,
            result.status,
            result.latency_ms,
        )
        if result.warnings:
            LOGGER.warning(
                "specialist_call_warning request_id=%s agent=%s warnings=%s",
                request_id,
                call.agent,
                " | ".join(result.warnings)[:2400],
            )
        return result

    def _validate_completion(
        self,
        decision: CompletionDecision,
        *,
        route: RouteName,
        private_company: bool,
        called_agents: set[str],
    ) -> None:
        remaining_budget = self.settings.max_specialist_calls - len(called_agents)

        if decision.done:
            if decision.next_action != "finish":
                raise ValueError("A completed workflow must use next_action=finish.")
            if route == "combined" and not decision.synthesis_required:
                raise ValueError(
                    "A completed combined route must request external synthesis."
                )
            if route != "combined" and decision.synthesis_required:
                raise ValueError(
                    "A one-specialist route must not request combined synthesis."
                )
            return

        if remaining_budget <= 0:
            raise ValueError("No specialist-call budget remains.")
        requested_agent = (
            "news" if decision.next_action == "call_news" else "financial"
        )
        if requested_agent in called_agents:
            raise ValueError("The completion plan attempted to repeat a specialist call.")

    def _mechanical_checks(
        self,
        *,
        plan: RoutingPlan,
        specialists: Sequence[SpecialistResponse],
        private_company: bool,
        synthesized_content: str = "",
    ) -> list[dict[str, object]]:
        actual_agents = [result.agent for result in specialists]
        combined_response_expected = plan.request_type == "combined" or {
            "news",
            "financial",
        }.issubset(actual_agents)
        source_citations = {
            citation for result in specialists for citation in result.citations
        }
        synthesized_citations = set(_extract_citations(synthesized_content))
        enabled_by_category = {
            "release": self.settings.release_checks_enabled,
            "policy": self.settings.policy_checks_enabled,
            "evidence": self.settings.evidence_checks_enabled,
        }

        def check(
            name: str,
            category: str,
            passed: bool,
            detail: str,
        ) -> dict[str, object]:
            return {
                "name": name,
                "category": category,
                "enabled": enabled_by_category[category],
                "passed": passed,
                "detail": detail,
            }

        specialist_route = plan.request_type in {"news", "financial", "combined"}
        checks = [
            check(
                "specialist_call_budget",
                "release",
                len(specialists) <= self.settings.max_specialist_calls,
                f"calls={len(specialists)} max={self.settings.max_specialist_calls}",
            ),
            check(
                "allowlisted_specialists_only",
                "release",
                all(agent in self.connections for agent in actual_agents),
                ",".join(actual_agents) or "none",
            ),
            check(
                "one_response_envelope_per_specialist",
                "release",
                len(actual_agents) == len(set(actual_agents)),
                "Independent specialist envelopes are retained before final composition.",
            ),
            check(
                "combined_response_composed",
                "release",
                not combined_response_expected
                or bool(synthesized_content.strip())
                or {"news", "financial"}.issubset(actual_agents),
                (
                    "A combined route may release a synthesized answer or separately labeled "
                    "specialist sections when synthesis is unavailable."
                ),
            ),
            check(
                "synthesized_citations_grounded",
                "evidence",
                not synthesized_content.strip()
                or synthesized_citations.issubset(source_citations),
                "Every citation in the synthesized answer came from a specialist response.",
            ),
            check(
                "no_orchestrator_domain_data_access",
                "policy",
                True,
                "All domain requests were sent through A2A specialist connections.",
            ),
            check(
                "private_company_ticker_policy",
                "policy",
                not private_company
                or any(result.agent == "financial" for result in specialists),
                (
                    "Private/unlisted notice is derived from the Financial Agent result; "
                    "the orchestrator does not infer a ticker."
                ),
            ),
            check(
                "plan_route_preserved",
                "release",
                plan.request_type
                in {"news", "financial", "combined", "clarification", "unsupported"},
                plan.request_type,
            ),
            check(
                "specialist_result_or_limit_available",
                "evidence",
                not specialist_route
                or (
                    bool(specialists)
                    and all(
                        _specialist_has_evidence_or_limit(result)
                        for result in specialists
                    )
                ),
                (
                    "Each called specialist returned cited output, an audited terminal "
                    "result, or an explicit failure/evidence limit."
                ),
            ),
        ]
        return checks

    async def _write_audit(
        self,
        *,
        audit_id: str,
        request_id: str,
        session_id: str,
        user_query: str,
        messages: Sequence[object],
        assessment: PolicyAssessment,
        plan: RoutingPlan,
        model_traces: Sequence[ModelCallTrace],
        specialists: Sequence[SpecialistResponse],
        completion: CompletionDecision | None,
        checks: Sequence[dict[str, object]],
        private_company: bool,
        execution_mode: str,
        rounds: int,
        started: float,
        outcome: str,
        synthesis: dict[str, object] | None = None,
    ) -> None:
        request_data: dict[str, object] = {
            "query_sha256": _sha256(user_query),
            "query_length": len(user_query),
            "conversation_turns": len(messages),
        }
        if self.settings.audit_include_query:
            request_data["query"] = user_query

        record = {
            "schema_version": "1.0",
            "event": "orchestrator_request_completed",
            "timestamp": _utc_now(),
            "audit_id": audit_id,
            "request_id": request_id,
            "identity": {
                "session_sha256": _sha256(session_id),
            },
            "request": request_data,
            "routing": {
                "deterministic_assessment": assessment.model_dump(),
                "validated_plan": _audit_plan(plan),
                "execution_mode": execution_mode,
                "rounds": rounds,
                "limits": {
                    "max_rounds": self.settings.max_orchestration_rounds,
                    "max_specialist_calls": self.settings.max_specialist_calls,
                    "max_clarifications": self.settings.max_clarifications,
                },
                "completion_decision": _audit_completion(completion),
            },
            "models": [_audit_model_trace(trace) for trace in model_traces],
            "synthesis": synthesis
            or {
                "requested": False,
                "attempted": False,
                "released": False,
            },
            "specialists": [
                {
                    "agent": result.agent,
                    "status": result.status,
                    "context_id_sha256": _sha256(result.context_id)
                    if result.context_id
                    else "",
                    "task_id": result.task_id,
                    "latency_ms": result.latency_ms,
                    "citation_count": len(result.citations),
                    "as_of": result.as_of,
                    "warnings": result.warnings,
                    "specialist_audit_id": result.data.get("audit_id", ""),
                    "private_or_unlisted": result.data.get(
                        "private_or_unlisted", False
                    ),
                    "output_sha256": result.data.get("output_sha256", ""),
                    "output_length": result.data.get("output_length", 0),
                }
                for result in specialists
            ],
            "governance": {
                "checks": list(checks),
                "check_configuration": {
                    "release": self.settings.release_checks_enabled,
                    "policy": self.settings.policy_checks_enabled,
                    "evidence": self.settings.evidence_checks_enabled,
                },
                "private_company": private_company,
                "direct_domain_sources_called": [],
                "verification_scope": (
                    "Mechanical route, budget, independent specialist envelopes, synthesis "
                    "citation provenance, release checks, and lightweight specialist "
                    "citation/evidence-limit presence. Claim-level domain evidence verification "
                    "remains inside each specialist."
                ),
            },
            "outcome": {
                "status": outcome,
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            },
        }
        await asyncio.to_thread(self.audit_log.append, record)

    async def handle_messages(
        self,
        messages: Sequence[object],
        *,
        session_id: str,
        request_id: str | None = None,
    ) -> OrchestratorResult:
        started = time.perf_counter()
        request_id = request_id or f"chatcmpl-{uuid.uuid4().hex}"
        audit_id = str(uuid.uuid4())
        user_query = _latest_user_text(messages)
        assessment = self.policy_manager.assess(messages)
        model_traces: list[ModelCallTrace] = []
        specialists: list[SpecialistResponse] = []
        completion: CompletionDecision | None = None
        execution_mode = "none"
        rounds = 1

        if not user_query:
            assessment = assessment.model_copy(
                update={
                    "route_hint": "clarification",
                    "route_reason": "No user message was supplied.",
                }
            )
        elif len(user_query) > self.settings.max_query_chars:
            assessment = assessment.model_copy(
                update={
                    "route_hint": "clarification",
                    "route_reason": (
                        f"The request exceeds the {self.settings.max_query_chars}-character "
                        "demo input limit."
                    ),
                }
            )

        effective_assessment = _planning_assessment(
            assessment,
            policy_checks_enabled=self.settings.policy_checks_enabled,
        )
        policy_status = "passed" if self.settings.policy_checks_enabled else "disabled"

        bypass_model = (
            not user_query
            or len(user_query) > self.settings.max_query_chars
            or not _fallback_company(effective_assessment)
            or _identity_ambiguity(effective_assessment)
            or (
                self.settings.policy_checks_enabled
                and effective_assessment.explicit_unsupported
            )
        )

        if bypass_model:
            plan = _fallback_plan(
                effective_assessment,
                user_query,
                relaxed=not self.settings.policy_checks_enabled,
            )
        else:
            plan_validator: Callable[[RoutingPlan], None]
            if self.settings.policy_checks_enabled:
                plan_validator = lambda candidate: self.policy_manager.validate_plan(
                    candidate,
                    effective_assessment,
                    messages,
                )
            else:
                plan_validator = lambda candidate: None
            try:
                plan, trace = await self.planner.plan(
                    messages,
                    effective_assessment,
                    plan_validator,
                )
                if not self.settings.policy_checks_enabled:
                    plan = _normalize_relaxed_plan(
                        plan,
                        effective_assessment,
                        user_query,
                    )
                model_traces.append(trace)
            except Exception as exc:
                trace = getattr(exc, "trace", None)
                if isinstance(trace, ModelCallTrace):
                    model_traces.append(trace)
                LOGGER.warning(
                    "nemotron_plan_fallback request_id=%s error_type=%s error=%s",
                    request_id,
                    type(exc).__name__,
                    _exception_detail(exc),
                )
                plan = _fallback_plan(
                    effective_assessment,
                    user_query,
                    relaxed=not self.settings.policy_checks_enabled,
                )

        plan = _canonicalize_specialist_requests(
            plan,
            effective_assessment,
            user_query,
        )

        LOGGER.info(
            "route_decision request_id=%s route=%s company=%s reason=%s",
            request_id,
            plan.request_type,
            plan.company or "unresolved",
            effective_assessment.route_reason,
        )

        if (
            self.settings.policy_checks_enabled
            and plan.request_type == "clarification"
            and self.policy_manager.clarification_exhausted(effective_assessment)
        ):
            content = _clarification_exhausted_content()
            checks = self._mechanical_checks(
                plan=plan, specialists=[], private_company=False
            )
            try:
                await self._write_audit(
                    audit_id=audit_id,
                    request_id=request_id,
                    session_id=session_id,
                    user_query=user_query,
                    messages=messages,
                    assessment=effective_assessment,
                    plan=plan,
                    model_traces=model_traces,
                    specialists=[],
                    completion=None,
                    checks=checks,
                    private_company=False,
                    execution_mode="none",
                    rounds=rounds,
                    started=started,
                    outcome="clarification_exhausted",
                )
                content += "\n\n---\n\n" + _trace_line(
                    route="clarification",
                    company=plan.company,
                    specialists=[],
                    execution_mode="none",
                    rounds=rounds,
                    audit_id=audit_id,
                    policy_status=policy_status,
                )
            except (AuditError, OSError) as exc:
                LOGGER.error("audit_write_failed request_id=%s error=%s", request_id, exc)
                content = (
                    _audit_failure_content()
                    if self.settings.release_checks_enabled
                    else _append_audit_warning(content)
                )
            return OrchestratorResult(
                content=content,
                request_id=request_id,
                audit_id=audit_id,
                route="clarification",
                company=plan.company,
                rounds=rounds,
                specialist_calls=[],
                private_company=False,
                completion_reason=plan.reason,
                execution_mode="none",
            )

        if plan.request_type in {"clarification", "unsupported"}:
            content = (
                plan.clarification_question
                if plan.request_type == "clarification"
                else _unsupported_content()
            )
            checks = self._mechanical_checks(
                plan=plan, specialists=[], private_company=False
            )
            outcome = plan.request_type
            try:
                await self._write_audit(
                    audit_id=audit_id,
                    request_id=request_id,
                    session_id=session_id,
                    user_query=user_query,
                    messages=messages,
                    assessment=effective_assessment,
                    plan=plan,
                    model_traces=model_traces,
                    specialists=[],
                    completion=None,
                    checks=checks,
                    private_company=False,
                    execution_mode="none",
                    rounds=rounds,
                    started=started,
                    outcome=outcome,
                )
                content += "\n\n---\n\n" + _trace_line(
                    route=plan.request_type,
                    company=plan.company,
                    specialists=[],
                    execution_mode="none",
                    rounds=rounds,
                    audit_id=audit_id,
                    policy_status=policy_status,
                )
            except (AuditError, OSError) as exc:
                LOGGER.error("audit_write_failed request_id=%s error=%s", request_id, exc)
                content = (
                    _audit_failure_content()
                    if self.settings.release_checks_enabled
                    else _append_audit_warning(content)
                )

            return OrchestratorResult(
                content=content,
                request_id=request_id,
                audit_id=audit_id,
                route=plan.request_type,
                company=plan.company,
                rounds=rounds,
                specialist_calls=[],
                private_company=False,
                completion_reason=plan.reason,
                execution_mode="none",
            )

        rounds = 2
        planned_calls = plan.agent_calls[: self.settings.max_specialist_calls]
        if plan.request_type == "combined":
            execution_mode = "parallel"
            specialists = list(
                await asyncio.gather(
                    *(
                        self._call_specialist(
                            call,
                            session_id=session_id,
                            request_id=request_id,
                            company=plan.company,
                        )
                        for call in planned_calls
                    )
                )
            )
        else:
            execution_mode = "sequential"
            for call in planned_calls:
                specialists.append(
                    await self._call_specialist(
                        call,
                        session_id=session_id,
                        request_id=request_id,
                        company=plan.company,
                    )
                )

        private_company = any(
            result.agent == "financial"
            and bool(result.data.get("private_or_unlisted"))
            for result in specialists
        )

        rounds = 3
        called_agents = {result.agent for result in specialists}
        unused_agents = [
            agent for agent in ("news", "financial") if agent not in called_agents
        ]

        completion_validator: Callable[[CompletionDecision], None]
        if self.settings.release_checks_enabled:
            completion_validator = lambda candidate: self._validate_completion(
                candidate,
                route=plan.request_type,
                private_company=private_company,
                called_agents=called_agents,
            )
        else:
            completion_validator = lambda candidate: None

        try:
            completion, trace = await self.planner.decide_completion(
                route=plan.request_type,
                company=plan.company,
                user_query=user_query,
                calls_made=specialists,
                unused_agents=unused_agents,
                private_company=private_company,
                validator=completion_validator,
            )
            model_traces.append(trace)
        except Exception as exc:
            trace = getattr(exc, "trace", None)
            if isinstance(trace, ModelCallTrace):
                model_traces.append(trace)
            LOGGER.warning(
                "nemotron_completion_fallback request_id=%s error_type=%s error=%s",
                request_id,
                type(exc).__name__,
                _exception_detail(exc),
            )
            completion = CompletionDecision(
                done=True,
                next_action="finish",
                synthesis_required=plan.request_type == "combined",
                synthesis_guidance=(
                    _default_synthesis_guidance(user_query)
                    if plan.request_type == "combined"
                    else ""
                ),
                reason="The planned specialist route reached a terminal result.",
            )

        if not completion.done and len(specialists) < self.settings.max_specialist_calls:
            escalation_agent = (
                "news" if completion.next_action == "call_news" else "financial"
            )
            if escalation_agent not in called_agents:
                escalation = AgentCallPlan(
                    agent=escalation_agent,
                    request=_specialist_request(
                        escalation_agent,
                        plan.company,
                        user_query,
                        effective_assessment,
                    ),
                )
                specialists.append(
                    await self._call_specialist(
                        escalation,
                        session_id=session_id,
                        request_id=request_id,
                        company=plan.company,
                    )
                )
                called_agents.add(escalation_agent)

        private_company = any(
            result.agent == "financial"
            and bool(result.data.get("private_or_unlisted"))
            for result in specialists
        )
        called_agents = {result.agent for result in specialists}
        synthesis_required = plan.request_type == "combined" or {
            "news",
            "financial",
        }.issubset(called_agents)
        synthesized_content = ""
        synthesis_attempted = False
        synthesis_attempts = 0
        synthesis_error_type = ""

        if synthesis_required and any(
            result.raw_text.strip() for result in specialists
        ):
            synthesis_attempted = True
            guidance = (
                completion.synthesis_guidance.strip()
                if completion.synthesis_required
                else _default_synthesis_guidance(user_query)
            )
            try:
                synthesized_content, trace = await self.planner.synthesize_combined(
                    company=plan.company,
                    user_query=user_query,
                    specialists=specialists,
                    guidance=guidance,
                    validator=lambda candidate: _validate_synthesized_content(
                        candidate,
                        specialists,
                    ),
                )
                synthesis_attempts = trace.attempts
                model_traces.append(trace)
            except Exception as exc:
                trace = getattr(exc, "trace", None)
                if isinstance(trace, ModelCallTrace):
                    synthesis_attempts = trace.attempts
                    model_traces.append(trace)
                synthesis_error_type = type(exc).__name__
                LOGGER.warning(
                    "combined_synthesis_failed request_id=%s error_type=%s error=%s",
                    request_id,
                    synthesis_error_type,
                    _exception_detail(exc),
                )

        synthesis_metadata: dict[str, object] = {
            "requested": synthesis_required,
            "attempted": synthesis_attempted,
            "released": bool(synthesized_content.strip()),
            "attempts": synthesis_attempts,
            "model": self.settings.llm_model if synthesis_attempted else "",
            "endpoint": self.settings.llm_url if synthesis_attempted else "",
            "source_citations": sorted(
                {
                    citation
                    for result in specialists
                    for citation in result.citations
                }
            ),
            "output_citations": _extract_citations(synthesized_content),
            "output_sha256": (
                _sha256(synthesized_content) if synthesized_content else ""
            ),
            "output_length": len(synthesized_content),
            "error_type": synthesis_error_type,
        }

        checks = self._mechanical_checks(
            plan=plan,
            specialists=specialists,
            private_company=private_company,
            synthesized_content=synthesized_content,
        )
        checks_passed = all(
            not bool(check.get("enabled")) or bool(check["passed"])
            for check in checks
        )
        content = _compose_response(
            route=plan.request_type,
            company=plan.company,
            specialists=specialists,
            private_company=private_company,
            execution_mode=execution_mode,
            rounds=rounds,
            audit_id=audit_id,
            policy_status=policy_status,
            synthesized_content=synthesized_content,
        )
        if not checks_passed:
            failed_checks = ", ".join(
                str(check["name"])
                for check in checks
                if bool(check.get("enabled")) and not bool(check["passed"])
            )
            content = (
                "The orchestrator withheld the specialist output because enabled "
                f"governance checks failed: {failed_checks}. Review the audit record "
                "for the demonstration trace."
            )

        try:
            await self._write_audit(
                audit_id=audit_id,
                request_id=request_id,
                session_id=session_id,
                user_query=user_query,
                messages=messages,
                assessment=effective_assessment,
                plan=plan,
                model_traces=model_traces,
                specialists=specialists,
                completion=completion,
                checks=checks,
                private_company=private_company,
                execution_mode=execution_mode,
                rounds=rounds,
                started=started,
                outcome=(
                    "withheld"
                    if not checks_passed
                    else "synthesis_unavailable"
                    if synthesis_required and not synthesized_content.strip()
                    else "completed"
                ),
                synthesis=synthesis_metadata,
            )
        except (AuditError, OSError) as exc:
            LOGGER.error("audit_write_failed request_id=%s error=%s", request_id, exc)
            content = (
                _audit_failure_content()
                if self.settings.release_checks_enabled
                else _append_audit_warning(content)
            )

        return OrchestratorResult(
            content=content,
            request_id=request_id,
            audit_id=audit_id,
            route=plan.request_type,
            company=plan.company,
            rounds=rounds,
            specialist_calls=specialists,
            private_company=private_company,
            completion_reason=completion.reason,
            execution_mode=execution_mode,
        )

    async def close(self) -> None:
        await asyncio.gather(
            self.planner.close(),
            *(connection.close() for connection in self.connections.values()),
            return_exceptions=True,
        )


def create_routing_agent(settings: Settings | None = None) -> RoutingAgent:
    """Construct the runtime orchestrator while keeping external agents separate."""

    from .llm_client import NemotronOrchestrator
    from .remote_agent_connection import RemoteAgentConnection

    settings = settings or load_settings()
    return RoutingAgent(
        settings=settings,
        planner=NemotronOrchestrator(settings),
        news_connection=RemoteAgentConnection(
            agent="news",
            agent_name="News Agent",
            agent_url=settings.news_agent_url,
            timeout_seconds=settings.a2a_timeout_seconds,
        ),
        financial_connection=RemoteAgentConnection(
            agent="financial",
            agent_name="Financial Agent",
            agent_url=settings.financial_agent_url,
            timeout_seconds=settings.a2a_timeout_seconds,
        ),
    )


__all__ = ["RoutingAgent", "create_routing_agent"]
