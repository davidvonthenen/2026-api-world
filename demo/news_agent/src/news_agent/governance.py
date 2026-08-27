"""Domain boundaries, evidence formatting, citation checks, and release formatting."""
from __future__ import annotations

import json
import re
from collections.abc import Iterable

from .models import EvidenceItem, RoutePlan
from .utils import extract_citation_ids, source_domain, strip_model_source_section

_FINANCIAL_QUERY = re.compile(
    r"\b(?:stock(?:\s+price)?|share\s+price|ticker|market\s+cap(?:italization)?|"
    r"p\s*/?\s*e\s+ratio|eps|earnings\s+per\s+share|sec\s+filing|10-[kq]|"
    r"income\s+statement|balance\s+sheet|cash\s+flow|quarterly\s+(?:results|financials)|"
    r"revenue|profit|valuation|price\s+target|buy\s+rating|sell\s+rating|"
    r"investment\s+(?:advice|recommendation)|public(?:ly)?\s+traded|"
    r"private(?:ly)?\s+held|stock\s+exchange\s+listing)\b",
    re.IGNORECASE,
)
_NEWS_DOMAIN_QUERY = re.compile(
    r"\b(?:news|current\s+events?|artificial\s+intelligence|ai|generative\s+ai|"
    r"machine\s+learning|foundation\s+models?|language\s+models?|technology|"
    r"products?|launch(?:ed|es|ing)?|announc(?:e|ed|ement|ements)|partnerships?|"
    r"acquisition|acquired|research|strategy|roadmap|regulation|leadership|"
    r"innovation|data\s+center|cloud|semiconductor|chips?|gpu|software|platform)\b",
    re.IGNORECASE,
)
_CURRENT_QUERY = re.compile(r"\b(?:latest|recent|current|today|news|this\s+week|this\s+month)\b", re.I)
_HISTORICAL_QUERY = re.compile(r"\b(?:history|historical|background|evolution|over\s+time|previously)\b", re.I)
_FORBIDDEN_FINANCIAL_OUTPUT = [
    re.compile(r"\b(?:stock|share)\s+price\b.{0,30}(?:\$|\d)", re.I | re.S),
    re.compile(r"\b(?:eps|earnings\s+per\s+share|p\s*/?\s*e\s+ratio|market\s+cap)\b.{0,25}\d", re.I | re.S),
    re.compile(
        r"\b(?:price\s+target|buy\s+rating|sell\s+rating|"
        r"investment\s+recommendation)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:public(?:ly)?\s+traded|private(?:ly)?\s+held|"
        r"stock\s+exchange\s+listing)\b",
        re.I,
    ),
]


def fallback_route(query: str) -> RoutePlan:
    """Apply non-negotiable domain rules before any model-based routing."""
    contains_financial = bool(_FINANCIAL_QUERY.search(query))
    has_news_signal = bool(_NEWS_DOMAIN_QUERY.search(query))
    financial_only = contains_financial and not has_news_signal

    if financial_only:
        return RoutePlan(
            need_historical=False,
            need_realtime=False,
            financial_only=True,
            contains_financial_request=True,
            risk_level="medium",
            reason="The request asks only for data owned by the Financial Agent.",
        )

    need_historical = bool(_HISTORICAL_QUERY.search(query))
    need_realtime = bool(_CURRENT_QUERY.search(query))
    if not need_historical and not need_realtime:
        # A general "what is this company doing in AI" request benefits from both.
        need_historical = True
        need_realtime = True

    return RoutePlan(
        need_historical=need_historical,
        need_realtime=need_realtime,
        financial_only=False,
        contains_financial_request=contains_financial,
        risk_level="medium" if contains_financial else "low",
        reason="Use bounded historical and current-news evidence selected by temporal intent.",
    )


def enforce_route_boundaries(query: str, model_plan: RoutePlan) -> RoutePlan:
    """Let the model choose news sources, never the News Agent's authority boundary."""
    deterministic = fallback_route(query)
    if deterministic.financial_only:
        return deterministic

    need_historical = model_plan.need_historical
    need_realtime = model_plan.need_realtime

    if deterministic.need_historical and deterministic.need_realtime:
        # General company/AI requests intentionally blend historical and current evidence.
        need_historical = True
        need_realtime = True
    if _HISTORICAL_QUERY.search(query):
        need_historical = True
    if _CURRENT_QUERY.search(query):
        need_realtime = True
    if not need_historical and not need_realtime:
        need_historical = deterministic.need_historical
        need_realtime = deterministic.need_realtime

    return model_plan.model_copy(
        update={
            "need_historical": need_historical,
            "need_realtime": need_realtime,
            "financial_only": False,
            "contains_financial_request": deterministic.contains_financial_request,
            "risk_level": (
                "medium"
                if deterministic.contains_financial_request and model_plan.risk_level == "low"
                else model_plan.risk_level
            ),
        }
    )


def assign_citation_ids(evidence: Iterable[EvidenceItem]) -> list[EvidenceItem]:
    historical_number = 0
    realtime_number = 0
    assigned: list[EvidenceItem] = []
    seen_source_ids: set[str] = set()

    for item in evidence:
        if item.source_id in seen_source_ids:
            continue
        seen_source_ids.add(item.source_id)
        if item.source_kind == "historical":
            historical_number += 1
            citation_id = f"H{historical_number}"
        else:
            realtime_number += 1
            citation_id = f"W{realtime_number}"
        assigned.append(item.model_copy(update={"citation_id": citation_id}))
    return assigned


def render_evidence_for_prompt(evidence: list[EvidenceItem]) -> str:
    """Serialize evidence as JSON Lines so source text cannot masquerade as instructions."""
    rows: list[str] = []
    for item in evidence:
        rows.append(
            json.dumps(
                {
                    "citation_id": item.citation_id,
                    "source_kind": item.source_kind,
                    "title": item.title,
                    "published_at": item.published_at,
                    "retrieved_at": item.retrieved_at,
                    "path": item.path,
                    "url": item.url,
                    "score": item.score,
                    "text": item.text,
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(rows)


def deterministic_verification_issues(
    answer: str,
    evidence: list[EvidenceItem],
    *,
    policy_checks_enabled: bool = True,
    evidence_checks_enabled: bool = True,
) -> list[str]:
    issues: list[str] = []
    known_ids = {item.citation_id for item in evidence}
    used_ids = extract_citation_ids(answer)
    unknown_ids = sorted(used_ids - known_ids)

    if evidence_checks_enabled:
        if unknown_ids:
            issues.append(f"Unknown citation IDs: {', '.join(unknown_ids)}")
        if evidence and not (used_ids & known_ids):
            issues.append("The answer contains no valid inline evidence citation")
    if policy_checks_enabled and any(
        pattern.search(answer) for pattern in _FORBIDDEN_FINANCIAL_OUTPUT
    ):
        issues.append("The answer crosses into stock metrics or investment analysis")
    return issues


def _source_line(item: EvidenceItem) -> str:
    if item.source_kind == "historical":
        details = [f'Historical corpus: "{item.title}"']
        domain = source_domain(item.url) or str(item.metadata.get("domain") or "").strip()
        if domain:
            details.append(domain)
        if item.published_at:
            details.append(f"published={item.published_at}")
        if item.url:
            details.append(item.url)
        if item.category:
            details.append(f"category={item.category}")
        if item.path:
            details.append(f"record={item.path}")
        if item.chunk_index is not None:
            details.append(f"chunk={item.chunk_index}")
        if item.score is not None:
            details.append(f"score={item.score:.4f}")
        return f"- [{item.citation_id}] " + " | ".join(details)

    details = [item.title]
    domain = source_domain(item.url)
    if domain:
        details.append(domain)
    if item.published_at:
        details.append(f"published={item.published_at}")
    if item.url:
        details.append(item.url)
    return f"- [{item.citation_id}] " + " | ".join(details)


def format_released_answer(
    draft: str,
    evidence: list[EvidenceItem],
    plan: RoutePlan,
    audit_id: str,
    *,
    missing_sources: list[str] | None = None,
) -> str:
    body = strip_model_source_section(draft).strip()
    cited = extract_citation_ids(body)
    selected = [item for item in evidence if item.citation_id in cited]

    blocks = [body]
    if plan.contains_financial_request:
        blocks.append(
            "**Scope boundary:** Public-market status, stock prices, SEC filings, valuation, "
            "earnings metrics, and investment analysis must be handled by the "
            "Financial Agent."
        )
    if missing_sources:
        friendly = {
            "historical": "historical OpenSearch corpus",
            "realtime": "current Tavily news",
        }
        missing = ", ".join(friendly.get(name, name) for name in missing_sources)
        blocks.append(
            "**Evidence gap:** No usable evidence was returned from "
            f"{missing}; the response is limited to the available sources."
        )
    if selected:
        blocks.append("**Sources**\n" + "\n".join(_source_line(item) for item in selected))
    blocks.append(f"**Audit ID:** `{audit_id}`")
    return "\n\n".join(block for block in blocks if block)


def format_withheld_answer(evidence: list[EvidenceItem], audit_id: str) -> str:
    blocks = [
        "The News Agent found evidence, but the generated synthesis did not pass "
        "support, citation, and domain-boundary checks. The answer was withheld."
    ]
    if evidence:
        blocks.append(
            "**Evidence reviewed**\n" + "\n".join(_source_line(item) for item in evidence)
        )
    blocks.append(f"**Audit ID:** `{audit_id}`")
    return "\n\n".join(blocks)


def format_no_evidence_answer(audit_id: str) -> str:
    return (
        "The News Agent could not retrieve enough historical or current-news evidence "
        "to support an answer.\n\n"
        f"**Audit ID:** `{audit_id}`"
    )


def format_financial_redirect(audit_id: str) -> str:
    return (
        "This request is outside the News Agent's authority. Public-market status, stock "
        "prices, SEC filings, earnings metrics, valuation, and investment analysis "
        "must be routed to the "
        "Financial Agent. No financial tool was called.\n\n"
        f"**Audit ID:** `{audit_id}`"
    )


def format_input_rejection(reason: str, audit_id: str) -> str:
    return f"The News Agent rejected the request: {reason}\n\n**Audit ID:** `{audit_id}`"


__all__ = [
    "assign_citation_ids",
    "deterministic_verification_issues",
    "enforce_route_boundaries",
    "fallback_route",
    "format_financial_redirect",
    "format_input_rejection",
    "format_no_evidence_answer",
    "format_released_answer",
    "format_withheld_answer",
    "render_evidence_for_prompt",
]
