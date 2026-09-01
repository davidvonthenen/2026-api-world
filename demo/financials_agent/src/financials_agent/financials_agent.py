"""Governed Financials Agent combining filing RAG, Finnhub MCP, and local models."""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
import uuid
from collections.abc import AsyncIterable
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from ..common.config import Settings, load_settings
from ..common.logging import get_logger, log_payload

from .audit import AuditLog
from .governance import (
    merge_model_verdict,
    normalize_generated_answer,
    verify_answer,
)
from .llm import ModelGatewayError, OpenAIModelGateway
from .mcp_client import FinancialMCPClient
from .models import (
    Evidence,
    FinancialPlan,
    ModelCallTrace,
    RetrievalTrace,
    ToolCallTrace,
    VerificationResult,
)
from .planner import (
    deterministic_plan,
    extract_company_hint,
    extract_json_object,
    extract_user_question,
    merge_model_plan,
)
from .retrieval import FinancialFilingsRAG

LOGGER = get_logger(__name__)
_FINNHUB_SOURCE_URL = "https://finnhub.io/"
_TOOL_TITLES = {
    "resolve_public_symbol": "Public-company symbol resolution",
    "get_company_profile": "Company profile",
    "get_stock_quote": "Current stock quote",
    "get_company_metrics": "Company valuation and operating metrics",
    "get_recent_quarterly_earnings": "Recent quarterly earnings",
    "get_earnings_calendar": "Earnings calendar",
}
_VERIFIER_WITHHOLDING_REASON = (
    "The candidate response did not satisfy citation, policy, or evidence-release "
    "requirements. Detailed verifier diagnostics are retained in audit metadata."
)
_VALID_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.-]{0,9}$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalise_tool_data(data: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    result = data.get("result")
    if len(data) == 1 and isinstance(result, dict):
        return result
    return data


def _as_of(data: dict[str, Any]) -> str:
    for key in ("market_timestamp", "retrieved_at", "to_date", "from_date"):
        value = data.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _tool_evidence(
    traces: list[ToolCallTrace],
    *,
    start_index: int = 1,
) -> list[Evidence]:
    evidence: list[Evidence] = []
    next_index = start_index
    for trace in traces:
        if not trace.ok or not trace.data:
            continue
        data = _normalise_tool_data(trace.data)
        evidence.append(
            Evidence(
                evidence_id=f"M{next_index}",
                kind="market_data",
                source=f"Finnhub via MCP: {trace.tool}",
                title=_TOOL_TITLES.get(trace.tool, trace.tool),
                content=json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False),
                as_of=_as_of(data),
                source_url=_FINNHUB_SOURCE_URL,
                metadata={
                    "tool": trace.tool,
                    "symbol": data.get("symbol", ""),
                    "latency_ms": round(trace.latency_ms, 3),
                },
            )
        )
        next_index += 1
    return evidence


def _market_calls(plan: FinancialPlan) -> list[tuple[str, dict[str, Any]]]:
    if not plan.symbol:
        return []
    calls: list[tuple[str, dict[str, Any]]] = []
    for tool in plan.tools:
        if tool == "resolve_public_symbol":
            continue
        arguments: dict[str, Any] = {"symbol": plan.symbol}
        if tool == "get_recent_quarterly_earnings":
            arguments["limit"] = 4
        calls.append((tool, arguments))
    return calls


def _out_of_scope_answer(audit_id: str) -> str:
    return (
        "## Financial assessment\n"
        "This request is outside the Financials Agent authority boundary. It should be routed to the News Agent or another matching vertical expert.\n\n"
        "## Evidence limits\n"
        "No financial filing or market-data source was queried.\n\n"
        "## Sources\n"
        "- None. The request was rejected before retrieval.\n\n"
        f"Audit ID: {audit_id}"
    )


def _identity_required_answer(audit_id: str) -> str:
    return (
        "## Financial assessment\n"
        "A company name or stock ticker is required before the agent can select a filing corpus or market-data endpoint.\n\n"
        "## Evidence limits\n"
        "No ticker was guessed, and no similarly named security was substituted.\n\n"
        "## Sources\n"
        "- None. Execution stopped before retrieval.\n\n"
        f"Audit ID: {audit_id}"
    )


def _resolution_failed_answer(audit_id: str) -> str:
    return (
        "## Financial assessment\n"
        "The company-to-ticker lookup did not return approved evidence, so the agent did not query filings or stock data.\n\n"
        "## Evidence limits\n"
        "The company may be private, unlisted, misspelled, or temporarily unavailable through the market-data service. No ticker was inferred.\n\n"
        "## Sources\n"
        "- None. The symbol-resolution tool failed or returned no structured result.\n\n"
        f"Audit ID: {audit_id}"
    )


def _private_company_answer(company: str, evidence: Evidence, audit_id: str) -> str:
    label = company or "The requested company"
    return (
        "## Financial assessment\n"
        f"{label} was not verified as a publicly traded company under a matching symbol, so the agent did not substitute another security or fabricate a stock price. [{evidence.evidence_id}]\n\n"
        "## Evidence limits\n"
        f"The lookup result is bounded to Finnhub's symbol search at the recorded retrieval time; it is not a legal determination of corporate status. [{evidence.evidence_id}]\n\n"
        "## Sources\n"
        f"- [{evidence.evidence_id}] {evidence.source}; as of {evidence.as_of or 'not supplied'}; {evidence.source_url}\n\n"
        f"Audit ID: {audit_id}"
    )


def _withheld_answer(audit_id: str, reason: str) -> str:
    return (
        "## Financial assessment\n"
        "The Financials Agent withheld an answer because the available evidence or release checks were insufficient.\n\n"
        "## Evidence limits\n"
        f"{reason}\n\n"
        "## Sources\n"
        "- No source set was approved for release.\n\n"
        f"Audit ID: {audit_id}"
    )


def _verification_failure_reason(
    verification: VerificationResult | None,
) -> str:
    if verification is None or not verification.reasons:
        return _VERIFIER_WITHHOLDING_REASON

    details: list[str] = []
    for reason in verification.reasons:
        cleaned = " ".join(str(reason).split())
        if not cleaned or cleaned in details:
            continue
        details.append(cleaned[:240])
        if len(details) == 6:
            break
    if not details:
        return _VERIFIER_WITHHOLDING_REASON
    return (
        "The candidate response was rejected after the bounded correction attempt. "
        "Verification reasons: "
        + " | ".join(details)
    )


def _requested_evidence_failure_reason(
    plan: FinancialPlan,
    retrieval_trace: RetrievalTrace | None,
    market_traces: list[ToolCallTrace],
) -> str:
    failures: list[str] = []
    if plan.needs_filings:
        if retrieval_trace and retrieval_trace.error:
            failures.append("the exact-symbol filing retrieval failed")
        else:
            failures.append("the exact-symbol filing retrieval returned no matching chunks")

    requested_market_tools = [
        tool for tool in plan.tools if tool != "resolve_public_symbol"
    ]
    successful_market_tools = {
        trace.tool for trace in market_traces if trace.ok and trace.data
    }
    for tool in requested_market_tools:
        if tool not in successful_market_tools:
            title = _TOOL_TITLES.get(tool, tool)
            failures.append(f"the {title.lower()} tool returned no usable result")

    if not failures:
        failures.append("the requested financial sources returned no usable evidence")
    return (
        "The company identity may have been resolved, but it is not sufficient to "
        "answer the requested financial question because "
        + "; ".join(failures)
        + "."
    )


class FinancialsAgent:
    """Bounded financial expert used directly or through the A2A executor."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        models: Any | None = None,
        rag: Any | None = None,
        mcp: Any | None = None,
        audit_log: AuditLog | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.models = models or OpenAIModelGateway(self.settings)
        self.rag = rag or FinancialFilingsRAG(self.settings)
        self.mcp = mcp or FinancialMCPClient(self.settings)
        self.audit_log = audit_log or AuditLog(self.settings.audit_log_path)

    async def _verify_candidate(
        self,
        question: str,
        candidate: str,
        evidence: list[Evidence],
        audit_id: str,
        model_traces: list[ModelCallTrace],
        warnings: list[str],
    ) -> VerificationResult:
        deterministic = verify_answer(
            candidate,
            evidence,
            audit_id,
            release_checks_enabled=self.settings.release_checks_enabled,
            policy_checks_enabled=self.settings.policy_checks_enabled,
            evidence_checks_enabled=self.settings.evidence_checks_enabled,
        )
        LOGGER.info(
            "Deterministic verification completed audit_id=%s approved=%s reasons=%d",
            audit_id,
            deterministic.approved,
            len(deterministic.reasons),
        )
        if not self.settings.release_checks_enabled:
            LOGGER.info(
                "Advisory release verification disabled audit_id=%s",
                audit_id,
            )
            return deterministic
        try:
            raw_verdict, trace = await self.models.verify(question, candidate, evidence)
            model_traces.append(trace)
            result = merge_model_verdict(deterministic, raw_verdict)
            LOGGER.info(
                "Advisory verification completed audit_id=%s approved=%s model_checked=%s model_approved=%s reasons=%d",
                audit_id,
                result.approved,
                result.model_checked,
                result.model_approved,
                len(result.reasons),
            )
            log_payload(
                LOGGER,
                f"Verification output payload audit_id={audit_id}",
                result.to_dict(),
            )
            return result
        except ModelGatewayError as exc:
            model_traces.append(exc.trace)
            warning = f"Nemotron release verifier unavailable: {exc}"
        except Exception as exc:  # injected gateways and defensive runtime boundary
            warning = f"Nemotron release verifier unavailable: {exc}"
        warnings.append(warning)
        result = VerificationResult(
            approved=deterministic.approved,
            reasons=[*deterministic.reasons, warning],
            cited_ids=deterministic.cited_ids,
            invalid_ids=deterministic.invalid_ids,
            model_checked=False,
            model_approved=None,
        )
        LOGGER.warning(
            "Advisory verification unavailable audit_id=%s deterministic_approved=%s error=%s",
            audit_id,
            deterministic.approved,
            warning,
        )
        log_payload(
            LOGGER,
            f"Verification fallback payload audit_id={audit_id}",
            result.to_dict(),
        )
        return result

    async def _plan(
        self,
        question: str,
        baseline: FinancialPlan,
        model_traces: list[ModelCallTrace],
        warnings: list[str],
    ) -> FinancialPlan:
        model_baseline = baseline
        if not self.settings.policy_checks_enabled:
            model_baseline = replace(
                baseline,
                in_scope=True,
                intent="financial_analysis",
            )
        try:
            raw_plan, trace = await self.models.plan(question, model_baseline)
            model_traces.append(trace)
            plan = merge_model_plan(
                model_baseline,
                extract_json_object(raw_plan),
                enforce_policy=self.settings.policy_checks_enabled,
            )
        except ModelGatewayError as exc:
            model_traces.append(exc.trace)
            warnings.append(f"Nemotron planner unavailable; deterministic plan used: {exc}")
            plan = model_baseline
        except Exception as exc:
            warnings.append(f"Nemotron plan rejected; deterministic plan used: {exc}")
            plan = model_baseline

        if not plan.company and not plan.symbol:
            plan = replace(plan, company=extract_company_hint(question))
        log_payload(LOGGER, "Merged financial execution plan", plan.to_dict())
        return plan

    async def ainvoke(self, query: str, session_id: str) -> dict[str, Any]:
        """Execute a governed route, retrieve, synthesize, verify, and audit cycle."""
        started = time.perf_counter()
        started_at = _utc_now()
        audit_id = str(uuid.uuid4())
        correlation_id = str(uuid.uuid4())
        clean_query = query.strip()
        effective_query = extract_user_question(clean_query)
        baseline = deterministic_plan(clean_query)
        plan = baseline
        evidence: list[Evidence] = []
        tool_traces: list[ToolCallTrace] = []
        model_traces: list[ModelCallTrace] = []
        retrieval_trace: RetrievalTrace | None = None
        verification: VerificationResult | None = None
        warnings: list[str] = []
        require_user_input = False
        outcome = "completed"

        LOGGER.debug(
            "Financials Agent request started audit_id=%s correlation_id=%s",
            audit_id,
            correlation_id,
        )
        log_payload(
            LOGGER,
            f"Financials Agent input payload audit_id={audit_id}",
            {
                "query": query,
                "session_id": session_id,
                "baseline_plan": baseline.to_dict(),
            },
        )

        if not clean_query:
            require_user_input = True
            outcome = "input_required"
            answer = _identity_required_answer(audit_id)
            return self._finalize(
                answer=answer,
                query=query,
                session_id=session_id,
                audit_id=audit_id,
                correlation_id=correlation_id,
                started=started,
                started_at=started_at,
                baseline=baseline,
                plan=plan,
                evidence=evidence,
                tool_traces=tool_traces,
                retrieval_trace=retrieval_trace,
                model_traces=model_traces,
                verification=verification,
                warnings=warnings,
                outcome=outcome,
                require_user_input=require_user_input,
            )

        if self.settings.policy_checks_enabled and not baseline.in_scope:
            outcome = "rejected_out_of_scope"
            answer = _out_of_scope_answer(audit_id)
            return self._finalize(
                answer=answer,
                query=query,
                session_id=session_id,
                audit_id=audit_id,
                correlation_id=correlation_id,
                started=started,
                started_at=started_at,
                baseline=baseline,
                plan=plan,
                evidence=evidence,
                tool_traces=tool_traces,
                retrieval_trace=retrieval_trace,
                model_traces=model_traces,
                verification=verification,
                warnings=warnings,
                outcome=outcome,
                require_user_input=False,
            )

        plan = await self._plan(effective_query, baseline, model_traces, warnings)
        LOGGER.info(
            "Financial execution plan ready audit_id=%s symbol=%s company=%s filings=%s market=%s tools=%s",
            audit_id,
            plan.symbol,
            plan.company,
            plan.needs_filings,
            plan.needs_market_data,
            plan.tools,
        )

        # Resolve an exact public symbol before any corpus or quote is selected.
        if not plan.symbol:
            if not plan.company:
                require_user_input = True
                outcome = "input_required"
                answer = _identity_required_answer(audit_id)
                return self._finalize(
                    answer=answer,
                    query=query,
                    session_id=session_id,
                    audit_id=audit_id,
                    correlation_id=correlation_id,
                    started=started,
                    started_at=started_at,
                    baseline=baseline,
                    plan=plan,
                    evidence=evidence,
                    tool_traces=tool_traces,
                    retrieval_trace=retrieval_trace,
                    model_traces=model_traces,
                    verification=verification,
                    warnings=warnings,
                    outcome=outcome,
                    require_user_input=require_user_input,
                )

            LOGGER.info(
                "Resolving public-company symbol audit_id=%s company=%s",
                audit_id,
                plan.company,
            )
            resolution_traces = await self.mcp.call_tools(
                [("resolve_public_symbol", {"company_or_symbol": plan.company})]
            )
            tool_traces.extend(resolution_traces)
            resolution_evidence = _tool_evidence(resolution_traces)
            evidence.extend(resolution_evidence)
            successful_resolution = next(
                (trace for trace in resolution_traces if trace.ok and trace.data),
                None,
            )
            if successful_resolution is None:
                warnings.extend(trace.error for trace in resolution_traces if trace.error)
                outcome = "symbol_resolution_failed"
                answer = _resolution_failed_answer(audit_id)
                return self._finalize(
                    answer=answer,
                    query=query,
                    session_id=session_id,
                    audit_id=audit_id,
                    correlation_id=correlation_id,
                    started=started,
                    started_at=started_at,
                    baseline=baseline,
                    plan=plan,
                    evidence=evidence,
                    tool_traces=tool_traces,
                    retrieval_trace=retrieval_trace,
                    model_traces=model_traces,
                    verification=verification,
                    warnings=warnings,
                    outcome=outcome,
                    require_user_input=False,
                )

            resolution = _normalise_tool_data(successful_resolution.data)
            if not resolution_evidence:
                warnings.append("Symbol resolution returned no releasable structured evidence.")
                outcome = "symbol_resolution_failed"
                answer = _resolution_failed_answer(audit_id)
                return self._finalize(
                    answer=answer,
                    query=query,
                    session_id=session_id,
                    audit_id=audit_id,
                    correlation_id=correlation_id,
                    started=started,
                    started_at=started_at,
                    baseline=baseline,
                    plan=plan,
                    evidence=evidence,
                    tool_traces=tool_traces,
                    retrieval_trace=retrieval_trace,
                    model_traces=model_traces,
                    verification=verification,
                    warnings=warnings,
                    outcome=outcome,
                    require_user_input=False,
                )
            if resolution.get("publicly_traded") is not True:
                outcome = "not_publicly_traded"
                answer = _private_company_answer(plan.company, resolution_evidence[0], audit_id)
                verification = await self._verify_candidate(
                    effective_query,
                    answer,
                    evidence,
                    audit_id,
                    model_traces,
                    warnings,
                )
                if not verification.approved:
                    outcome = "withheld_by_verifier"
                    answer = _withheld_answer(
                        audit_id,
                        _verification_failure_reason(verification),
                    )
                return self._finalize(
                    answer=answer,
                    query=query,
                    session_id=session_id,
                    audit_id=audit_id,
                    correlation_id=correlation_id,
                    started=started,
                    started_at=started_at,
                    baseline=baseline,
                    plan=plan,
                    evidence=evidence,
                    tool_traces=tool_traces,
                    retrieval_trace=retrieval_trace,
                    model_traces=model_traces,
                    verification=verification,
                    warnings=warnings,
                    outcome=outcome,
                    require_user_input=False,
                )

            resolved_symbol = str(resolution.get("symbol") or "").upper().strip()
            if not _VALID_SYMBOL_RE.fullmatch(resolved_symbol):
                warnings.append("Symbol resolution returned an invalid ticker format.")
                outcome = "symbol_resolution_failed"
                answer = _resolution_failed_answer(audit_id)
                return self._finalize(
                    answer=answer,
                    query=query,
                    session_id=session_id,
                    audit_id=audit_id,
                    correlation_id=correlation_id,
                    started=started,
                    started_at=started_at,
                    baseline=baseline,
                    plan=plan,
                    evidence=evidence,
                    tool_traces=tool_traces,
                    retrieval_trace=retrieval_trace,
                    model_traces=model_traces,
                    verification=verification,
                    warnings=warnings,
                    outcome=outcome,
                    require_user_input=False,
                )
            plan = replace(plan, symbol=resolved_symbol)
            LOGGER.info(
                "Public-company symbol resolved audit_id=%s company=%s symbol=%s",
                audit_id,
                plan.company,
                plan.symbol,
            )

        calls = _market_calls(plan)

        async def retrieve_filings() -> tuple[list[Evidence], RetrievalTrace | None]:
            if not plan.needs_filings:
                return [], None
            return await self.rag.retrieve(effective_query, plan.symbol)

        filing_result, market_result = await asyncio.gather(
            retrieve_filings(),
            self.mcp.call_tools(calls),
        )
        filing_evidence, retrieval_trace = filing_result
        tool_traces.extend(market_result)
        evidence.extend(filing_evidence)
        market_start = len(
            [item for item in evidence if item.evidence_id.startswith("M")]
        ) + 1
        market_evidence = _tool_evidence(market_result, start_index=market_start)
        evidence.extend(market_evidence)

        LOGGER.info(
            "Evidence collection completed audit_id=%s filing_items=%d market_items=%d resolution_items=%d",
            audit_id,
            len(filing_evidence),
            len(market_evidence),
            len([item for item in evidence if item.metadata.get("tool") == "resolve_public_symbol"]),
        )
        log_payload(
            LOGGER,
            f"Approved evidence payload audit_id={audit_id}",
            [item.to_dict() for item in evidence],
        )

        if retrieval_trace and retrieval_trace.error:
            warnings.append(f"Financial filings retrieval failed: {retrieval_trace.error}")
        warnings.extend(
            f"MCP tool {trace.tool} failed: {trace.error}"
            for trace in market_result
            if not trace.ok and trace.error
        )

        # Symbol-resolution output establishes issuer identity. It does not, by
        # itself, answer a quote, earnings, metrics, or filing question. The
        # previous check counted resolution evidence as sufficient and sent an
        # evidence-starved prompt to Qwen, which then reached the generic
        # verifier-withholding path.
        requested_evidence = [*filing_evidence, *market_evidence]
        if self.settings.evidence_checks_enabled and not requested_evidence:
            outcome = "no_approved_evidence"
            answer = _withheld_answer(
                audit_id,
                _requested_evidence_failure_reason(
                    plan,
                    retrieval_trace,
                    market_result,
                ),
            )
            return self._finalize(
                answer=answer,
                query=query,
                session_id=session_id,
                audit_id=audit_id,
                correlation_id=correlation_id,
                started=started,
                started_at=started_at,
                baseline=baseline,
                plan=plan,
                evidence=evidence,
                tool_traces=tool_traces,
                retrieval_trace=retrieval_trace,
                model_traces=model_traces,
                verification=verification,
                warnings=warnings,
                outcome=outcome,
                require_user_input=False,
            )

        try:
            answer, trace = await self.models.synthesize(
                effective_query,
                plan,
                evidence,
                audit_id,
            )
            model_traces.append(trace)
            normalized_answer = normalize_generated_answer(answer, audit_id, evidence)
            if normalized_answer != answer:
                log_payload(
                    LOGGER,
                    f"Normalized answer payload audit_id={audit_id}",
                    {
                        "before": answer,
                        "after": normalized_answer,
                    },
                )
            answer = normalized_answer
        except ModelGatewayError as exc:
            model_traces.append(exc.trace)
            warnings.append(f"Qwen synthesis failed: {exc}")
            outcome = "synthesis_failed"
            answer = _withheld_answer(audit_id, "The answer model did not return a releasable response.")
            return self._finalize(
                answer=answer,
                query=query,
                session_id=session_id,
                audit_id=audit_id,
                correlation_id=correlation_id,
                started=started,
                started_at=started_at,
                baseline=baseline,
                plan=plan,
                evidence=evidence,
                tool_traces=tool_traces,
                retrieval_trace=retrieval_trace,
                model_traces=model_traces,
                verification=verification,
                warnings=warnings,
                outcome=outcome,
                require_user_input=False,
            )
        except Exception as exc:
            warnings.append(f"Qwen synthesis failed: {exc}")
            outcome = "synthesis_failed"
            answer = _withheld_answer(audit_id, "The answer model did not return a releasable response.")
            return self._finalize(
                answer=answer,
                query=query,
                session_id=session_id,
                audit_id=audit_id,
                correlation_id=correlation_id,
                started=started,
                started_at=started_at,
                baseline=baseline,
                plan=plan,
                evidence=evidence,
                tool_traces=tool_traces,
                retrieval_trace=retrieval_trace,
                model_traces=model_traces,
                verification=verification,
                warnings=warnings,
                outcome=outcome,
                require_user_input=False,
            )

        verification = await self._verify_candidate(
            effective_query,
            answer,
            evidence,
            audit_id,
            model_traces,
            warnings,
        )

        if not verification.approved:
            LOGGER.warning(
                "Candidate answer rejected audit_id=%s; starting bounded correction reasons=%s",
                audit_id,
                verification.reasons,
            )
            try:
                retry, retry_trace = await self.models.synthesize(
                    effective_query,
                    plan,
                    evidence,
                    audit_id,
                    retry_reasons=verification.reasons,
                    previous_answer=answer,
                )
                model_traces.append(retry_trace)
                normalized_retry = normalize_generated_answer(retry, audit_id, evidence)
                if normalized_retry != retry:
                    log_payload(
                        LOGGER,
                        f"Normalized retry payload audit_id={audit_id}",
                        {
                            "before": retry,
                            "after": normalized_retry,
                        },
                    )
                retry = normalized_retry
                retry_verification = await self._verify_candidate(
                    effective_query,
                    retry,
                    evidence,
                    audit_id,
                    model_traces,
                    warnings,
                )
                answer = retry
                verification = retry_verification
            except ModelGatewayError as exc:
                model_traces.append(exc.trace)
                warnings.append(f"Qwen governed retry failed: {exc}")
            except Exception as exc:
                warnings.append(f"Qwen governed retry failed: {exc}")

        if verification is None or not verification.approved:
            outcome = "withheld_by_verifier"
            answer = _withheld_answer(
                audit_id,
                _verification_failure_reason(verification),
            )
        else:
            outcome = "released"

        return self._finalize(
            answer=answer,
            query=query,
            session_id=session_id,
            audit_id=audit_id,
            correlation_id=correlation_id,
            started=started,
            started_at=started_at,
            baseline=baseline,
            plan=plan,
            evidence=evidence,
            tool_traces=tool_traces,
            retrieval_trace=retrieval_trace,
            model_traces=model_traces,
            verification=verification,
            warnings=warnings,
            outcome=outcome,
            require_user_input=False,
        )

    def _finalize(
        self,
        *,
        answer: str,
        query: str,
        session_id: str,
        audit_id: str,
        correlation_id: str,
        started: float,
        started_at: str,
        baseline: FinancialPlan,
        plan: FinancialPlan,
        evidence: list[Evidence],
        tool_traces: list[ToolCallTrace],
        retrieval_trace: RetrievalTrace | None,
        model_traces: list[ModelCallTrace],
        verification: VerificationResult | None,
        warnings: list[str],
        outcome: str,
        require_user_input: bool,
    ) -> dict[str, Any]:
        completed_at = _utc_now()
        duration_ms = (time.perf_counter() - started) * 1000
        audit_record = {
            "event": "financials_agent_completion",
            "audit_id": audit_id,
            "correlation_id": correlation_id,
            "started_at": started_at,
            "completed_at": completed_at,
            "duration_ms": round(duration_ms, 3),
            "authority": "financial_filings_and_market_data",
            "outcome": outcome,
            "query_sha256": _sha256(query),
            "query_characters": len(query),
            "session_sha256": _sha256(session_id),
            "baseline_plan": baseline.to_dict(),
            "execution_plan": plan.to_dict(),
            "evidence": [item.to_dict(include_content=False) for item in evidence],
            "evidence_content_sha256": {
                item.evidence_id: _sha256(item.content) for item in evidence
            },
            "tool_calls": [
                {
                    "tool": trace.tool,
                    "argument_keys": sorted(str(key) for key in trace.arguments),
                    "arguments_sha256": _sha256(
                        json.dumps(trace.arguments, sort_keys=True, default=str)
                    ),
                    "ok": trace.ok,
                    "error": trace.error,
                    "latency_ms": round(trace.latency_ms, 3),
                    "data_sha256": _sha256(
                        json.dumps(trace.data, sort_keys=True, default=str)
                    )
                    if trace.data is not None
                    else "",
                }
                for trace in tool_traces
            ],
            "retrieval": retrieval_trace.to_dict() if retrieval_trace else None,
            "model_calls": [trace.to_dict() for trace in model_traces],
            "verification": verification.to_dict() if verification else None,
            "warnings": list(dict.fromkeys(warnings)),
            "governance_checks": {
                "release": self.settings.release_checks_enabled,
                "policy": self.settings.policy_checks_enabled,
                "evidence": self.settings.evidence_checks_enabled,
            },
            "answer_sha256": _sha256(answer),
        }

        record_hash = ""
        log_payload(
            LOGGER,
            f"Audit append input payload audit_id={audit_id}",
            audit_record,
        )
        try:
            record_hash = self.audit_log.append(audit_record)
        except Exception as exc:
            warning = f"Audit append failed: {exc}"
            warnings.append(warning)
            LOGGER.error(warning)

        metadata = {
            "audit_id": audit_id,
            "audit_record_hash": record_hash,
            "correlation_id": correlation_id,
            "authority": "financial_filings_and_market_data",
            "outcome": outcome,
            "duration_ms": round(duration_ms, 3),
            "plan": plan.to_dict(),
            "evidence": [item.to_dict(include_content=False) for item in evidence],
            "tool_calls": [trace.to_dict() for trace in tool_traces],
            "retrieval": retrieval_trace.to_dict() if retrieval_trace else None,
            "model_calls": [trace.to_dict() for trace in model_traces],
            "verification": verification.to_dict() if verification else None,
            "warnings": list(dict.fromkeys(warnings)),
            "governance_checks": {
                "release": self.settings.release_checks_enabled,
                "policy": self.settings.policy_checks_enabled,
                "evidence": self.settings.evidence_checks_enabled,
            },
        }
        LOGGER.info(
            "Financials Agent completed outcome=%s audit_id=%s duration_ms=%.2f",
            outcome,
            audit_id,
            duration_ms,
        )
        result = {
            "is_task_complete": not require_user_input,
            "require_user_input": require_user_input,
            "content": answer,
            "metadata": metadata,
        }
        log_payload(
            LOGGER,
            f"Financials Agent output payload audit_id={audit_id}",
            result,
        )
        return result

    async def stream(self, query: str, session_id: str) -> AsyncIterable[dict[str, Any]]:
        """Retain the scaffold's single-result streaming interface."""
        yield await self.ainvoke(query, session_id)


# Compatibility alias retained for the existing host scaffold's import style.
FinancialAgent = FinancialsAgent


__all__ = ["FinancialsAgent", "FinancialAgent"]
