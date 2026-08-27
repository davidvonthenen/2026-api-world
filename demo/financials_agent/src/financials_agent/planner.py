"""Deterministic policy routing and constrained model-plan parsing."""
from __future__ import annotations

import json
import re
from typing import Any

from .models import FinancialPlan

ALLOWED_TOOLS = {
    "resolve_public_symbol",
    "get_company_profile",
    "get_stock_quote",
    "get_company_metrics",
    "get_recent_quarterly_earnings",
    "get_earnings_calendar",
}

_TICKER_RE = re.compile(r"(?<![A-Za-z0-9])([A-Z]{1,5}(?:\.[A-Z])?)(?![A-Za-z0-9])")
_DOLLAR_TICKER_RE = re.compile(r"(?<![A-Za-z0-9])\$([A-Za-z]{1,5}(?:\.[A-Za-z])?)(?![A-Za-z0-9])")
_NAMED_TICKER_RE = re.compile(
    r"(?i:\b(?:ticker(?:\s+symbol)?|symbol))\s*(?:(?i:is)|=|:)\s*"
    r"\$?([A-Za-z]{1,5}(?:\.[A-Za-z])?)\b"
)
_LOOSE_NAMED_TICKER_RE = re.compile(
    r"(?i:\b(?:ticker(?:\s+symbol)?|symbol))\s+\$?"
    r"([A-Z]{1,5}(?:\.[A-Z])?)\b"
)
_VALID_TICKER_RE = re.compile(r"^[A-Z]{1,5}(?:\.[A-Z])?$")
_USER_REQUEST_MARKER = "User request:"
_TICKER_STOPWORDS = {
    "AI",
    "API",
    "A2A",
    "MCP",
    "SEC",
    "EPS",
    "PE",
    "P",
    "E",
    "RAG",
    "LLM",
    "USD",
    "CEO",
    "CFO",
    "IPO",
    "ETF",
    "Q1",
    "Q2",
    "Q3",
    "Q4",
}

_FINANCIAL_TERMS = {
    "stock",
    "share",
    "ticker",
    "quote",
    "price",
    "valuation",
    "market cap",
    "p/e",
    "financial",
    "filing",
    "10-k",
    "10-q",
    "earnings",
    "revenue",
    "margin",
    "cash flow",
    "balance sheet",
    "eps",
    "quarter",
    "annual report",
    "sec",
    "publicly traded",
    "public company",
    "private company",
    "listed",
    "exchange",
}
_NEWS_TERMS = {
    "news",
    "headline",
    "announced",
    "announcement",
    "current events",
    "press coverage",
    "what is happening",
    "latest ai work",
}
_FILING_TERMS = {
    "filing",
    "10-k",
    "10-q",
    "annual report",
    "quarterly report",
    "revenue",
    "margin",
    "cash flow",
    "balance sheet",
    "risk factor",
    "historical financial",
    "financials",
    "financial performance",
    "quarterly performance",
    "earnings",
    "eps",
}
_MARKET_TERMS = {
    "stock",
    "share price",
    "stock price",
    "quote",
    "market cap",
    "valuation",
    "p/e",
    "pe ratio",
    "52-week",
    "trading",
    "current price",
    "publicly traded",
    "public company",
    "private company",
    "listed",
    "exchange",
    "ticker",
}
_QUOTE_TERMS = {
    "stock price",
    "share price",
    "quote",
    "trading",
    "current price",
    "day high",
    "day low",
    "previous close",
}
_EARNINGS_TERMS = {"earnings", "eps", "quarter", "quarterly", "surprise"}
_UPCOMING_TERMS = {"upcoming earnings", "next earnings", "earnings date", "calendar"}
_PROFILE_TERMS = {
    "exchange",
    "industry",
    "market cap",
    "ipo",
    "publicly traded",
    "public company",
    "private company",
    "listed",
}
_PERSONAL_ADVICE_PATTERNS = (
    "should i buy",
    "should i sell",
    "should i invest",
    "my portfolio",
    "recommend a stock",
    "best stock for me",
)


def _contains_any(text: str, terms: set[str]) -> bool:
    return any(term in text for term in terms)


def extract_user_question(value: str) -> str:
    """Return the user request inside the trusted host envelope, when present."""
    text = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if _USER_REQUEST_MARKER not in text:
        return text
    return text.rsplit(_USER_REQUEST_MARKER, 1)[-1].strip()


def _host_field(value: str, name: str) -> str:
    """Read one explicit field from the host envelope without scanning policy prose."""
    if _USER_REQUEST_MARKER not in value:
        return ""
    envelope = value.rsplit(_USER_REQUEST_MARKER, 1)[0]
    match = re.search(
        rf"(?im)^\s*{re.escape(name)}\s*:\s*([^\n]+?)\s*$",
        envelope,
    )
    return match.group(1).strip() if match else ""


def _host_bool(value: str, name: str) -> bool:
    return _host_field(value, name).lower() in {"1", "true", "yes", "on"}


def extract_ticker(question: str) -> str:
    """Extract only an explicit ticker while rejecting ordinary capitalized words."""
    user_question = extract_user_question(question)
    host_ticker = _host_field(question, "User-supplied ticker")
    search_values = [user_question]
    if host_ticker:
        search_values.append(f"ticker: {host_ticker}")

    for value in search_values:
        for pattern in (
            _DOLLAR_TICKER_RE,
            _NAMED_TICKER_RE,
            _LOOSE_NAMED_TICKER_RE,
            _TICKER_RE,
        ):
            for match in pattern.finditer(value):
                candidate = match.group(1).upper()
                if candidate not in _TICKER_STOPWORDS and _VALID_TICKER_RE.fullmatch(candidate):
                    return candidate
    return ""


def _clean_company(value: str) -> str:
    value = re.sub(r"[\r\n\t]+", " ", value)
    value = " ".join(value.split()).strip(" -.,:;?!'\"")
    return value[:120]


def extract_company_hint(question: str) -> str:
    """Extract a conservative company phrase when no explicit ticker exists."""
    candidate_text = extract_user_question(question)
    candidate_text = re.sub(
        r"(?i)^(?:please\s+)?(?:"
        r"(?:can|could|will|would)\s+you\s+(?:tell|show|give)\s+me(?:\s+about)?|"
        r"tell me(?:\s+about)?|give me|what (?:is|are|was|were)|"
        r"how (?:is|are|was|were)|is|are|was|were|does|did|can|could|will|would"
        r")\s+",
        "",
        candidate_text,
    ).strip()

    quoted = re.search(r'["“]([^"”]{2,120})["”]', candidate_text)
    if quoted:
        return _clean_company(quoted.group(1))

    finance_noun = (
        r"(?:stock|shares?|financials?|filings?|earnings|valuation|price|quote|ticker|symbol|"
        r"revenue|margins?|cash flow|balance sheet|10-[KQ]|publicly traded|public company)"
    )
    modifiers = r"(?:(?i:current|latest|recent|historical|quarterly|annual)\s+){0,3}"
    patterns = (
        rf"\b([A-Z][A-Za-z0-9&. -]{{1,100}}?)'s\s+{modifiers}(?i:{finance_noun})\b",
        rf"(?i:\b(?:about|for|of)\s+)([A-Z][A-Za-z0-9&. -]{{1,100}}?)"
        rf"(?=(?:'s)?\s+{modifiers}(?i:{finance_noun})\b|[?.!,]|$)",
        rf"(?i:^(?:is|was|does|did|can|could|will|would|how is|what is)\s+)"
        rf"([A-Z][A-Za-z0-9&. -]{{1,100}}?)(?=\s+(?i:a\s+)?(?i:{finance_noun})\b)",
        rf"^([A-Z][A-Za-z0-9&. -]{{1,100}}?)(?=\s+{modifiers}(?i:{finance_noun})\b)",
        r"(?i:\b(?:company|firm)\s+)([A-Z][A-Za-z0-9&. -]{1,100}?)(?=[?.!,]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, candidate_text)
        if match:
            value = _clean_company(match.group(1))
            lowered_value = value.lower()
            request_verbs = {
                "analyze",
                "analyse",
                "compare",
                "describe",
                "explain",
                "provide",
                "review",
                "show",
                "summarize",
                "summarise",
            }
            first_token = lowered_value.split(maxsplit=1)[0] if lowered_value else ""
            if (
                value
                and lowered_value not in {"the", "this", "that", "a", "an"}
                and first_token not in request_verbs
                and not lowered_value.endswith(" the")
            ):
                return value
    return ""


def deterministic_plan(question: str) -> FinancialPlan:
    """Apply authority and data-minimization rules before any model is called."""
    user_question = extract_user_question(question)
    lowered = " ".join(user_question.lower().split())
    has_financial = _contains_any(lowered, _FINANCIAL_TERMS)
    has_news = _contains_any(lowered, _NEWS_TERMS)
    symbol = extract_ticker(question)
    host_company = _clean_company(_host_field(question, "Company"))
    company = "" if symbol else (extract_company_hint(user_question) or host_company)

    host_needs_market = _host_bool(question, "Current market data required")
    host_needs_filings = _host_bool(question, "Filing context required")

    in_scope = has_financial or bool(symbol) or host_needs_market or host_needs_filings
    if (
        has_news
        and not has_financial
        and not symbol
        and not host_needs_market
        and not host_needs_filings
    ):
        in_scope = False

    needs_filings = host_needs_filings or _contains_any(lowered, _FILING_TERMS)
    needs_market = (
        _contains_any(lowered, _MARKET_TERMS)
        or _contains_any(lowered, _EARNINGS_TERMS | _UPCOMING_TERMS | _PROFILE_TERMS)
        or host_needs_market
    )

    # A vague request routed to this specialist gets both bounded sources rather
    # than an ungrounded general-model answer.
    vague_financial_request = in_scope and not needs_filings and not needs_market
    if vague_financial_request:
        needs_filings = True
        needs_market = True

    tools: list[str] = []
    if needs_market or needs_filings:
        if not symbol:
            tools.append("resolve_public_symbol")
        if vague_financial_request or _contains_any(lowered, _QUOTE_TERMS):
            tools.append("get_stock_quote")
        if _contains_any(lowered, _PROFILE_TERMS):
            tools.append("get_company_profile")
        if "valuation" in lowered or "market cap" in lowered or "p/e" in lowered or "pe ratio" in lowered:
            tools.append("get_company_metrics")
        if _contains_any(lowered, _EARNINGS_TERMS):
            tools.append("get_recent_quarterly_earnings")
        if _contains_any(lowered, _UPCOMING_TERMS):
            tools.append("get_earnings_calendar")
        if needs_market and not any(
            tool
            in {
                "get_company_profile",
                "get_stock_quote",
                "get_company_metrics",
                "get_recent_quarterly_earnings",
                "get_earnings_calendar",
            }
            for tool in tools
        ):
            tools.append("get_stock_quote")

    risk = "high" if any(pattern in lowered for pattern in _PERSONAL_ADVICE_PATTERNS) else "low"
    return FinancialPlan(
        in_scope=in_scope,
        intent="financial_analysis" if in_scope else "handoff_to_news",
        company=company,
        symbol=symbol,
        needs_filings=needs_filings,
        needs_market_data=needs_market,
        tools=list(dict.fromkeys(tools)),
        risk=risk,
        rationale="Deterministic authority and evidence policy.",
    )


def extract_json_object(text: str) -> dict[str, Any]:
    """Read one JSON object from a model response, tolerating code fences."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Model response did not contain a JSON object")
    payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Model plan must be a JSON object")
    return payload


def merge_model_plan(
    baseline: FinancialPlan,
    payload: dict[str, Any],
    *,
    enforce_policy: bool = True,
) -> FinancialPlan:
    """Constrain Nemotron output so it cannot widen the authority boundary."""
    if enforce_policy and not baseline.in_scope:
        return baseline

    model_tools = payload.get("tools")
    constrained_tools: list[str] = []
    if isinstance(model_tools, list):
        for item in model_tools:
            name = str(item)
            if name not in ALLOWED_TOOLS:
                continue
            if name == "resolve_public_symbol" and not baseline.symbol:
                constrained_tools.append(name)
            elif name != "resolve_public_symbol" and baseline.needs_market_data:
                constrained_tools.append(name)

    if not enforce_policy:
        model_tools = payload.get("tools")
        relaxed_tools = (
            [str(item) for item in model_tools if str(item) in ALLOWED_TOOLS]
            if isinstance(model_tools, list)
            else []
        )
        company = baseline.company or _clean_company(str(payload.get("company") or ""))
        # Relaxed policy mode may recover a company name from the model, but a
        # ticker still must come from deterministic user-input parsing or the
        # approved Finnhub resolver.
        symbol = baseline.symbol
        needs_filings = baseline.needs_filings or payload.get("needs_filings") is True
        needs_market = (
            baseline.needs_market_data or payload.get("needs_market_data") is True
        )
        if not needs_filings and not needs_market:
            needs_filings = True
            needs_market = True

        merged_tools = list(dict.fromkeys([*baseline.tools, *relaxed_tools]))
        if (needs_filings or needs_market) and not symbol and company:
            if "resolve_public_symbol" not in merged_tools:
                merged_tools.insert(0, "resolve_public_symbol")
        if needs_market and not any(
            name in merged_tools
            for name in {
                "get_company_profile",
                "get_stock_quote",
                "get_company_metrics",
                "get_recent_quarterly_earnings",
                "get_earnings_calendar",
            }
        ):
            merged_tools.append("get_stock_quote")

        return FinancialPlan(
            in_scope=True,
            intent=str(payload.get("intent") or "financial_analysis")[:80],
            company=company,
            symbol=symbol,
            needs_filings=needs_filings,
            needs_market_data=needs_market,
            tools=merged_tools,
            risk=baseline.risk,
            rationale=str(payload.get("rationale") or baseline.rationale)[:500],
        )

    merged_tools = list(dict.fromkeys([*baseline.tools, *constrained_tools]))
    # A model is not an approved entity-resolution or security-master service.
    # The company and ticker must originate in deterministic user-input parsing;
    # the MCP resolver may then verify the company-to-symbol mapping.
    company = baseline.company
    symbol = baseline.symbol

    # The model may refine calls inside the approved domain but cannot request
    # additional source categories beyond the deterministic policy baseline.
    needs_filings = baseline.needs_filings
    needs_market = baseline.needs_market_data
    if (needs_filings or needs_market) and not symbol and "resolve_public_symbol" not in merged_tools:
        merged_tools.insert(0, "resolve_public_symbol")

    if needs_market and not any(
        name in merged_tools
        for name in {
            "get_company_profile",
            "get_stock_quote",
            "get_company_metrics",
            "get_recent_quarterly_earnings",
            "get_earnings_calendar",
        }
    ):
        merged_tools.append("get_stock_quote")

    return FinancialPlan(
        in_scope=True,
        intent=str(payload.get("intent") or baseline.intent)[:80],
        company=company,
        symbol=symbol,
        needs_filings=needs_filings,
        needs_market_data=needs_market,
        tools=merged_tools,
        risk=baseline.risk,
        rationale=str(payload.get("rationale") or baseline.rationale)[:500],
    )
