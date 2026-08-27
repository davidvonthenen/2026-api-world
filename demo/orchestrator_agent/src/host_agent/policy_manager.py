"""Deterministic routing policy that constrains Nemotron control-plane output."""

from __future__ import annotations

import re

from collections.abc import Sequence

from .models import PolicyAssessment, RouteName, RoutingPlan


NEWS_PATTERNS = (
    r"\bnews\b",
    r"\bannouncement(?:s)?\b",
    r"\bannounc(?:e|ed|es|ing)\b",
    r"\bproduct(?:s)?\b",
    r"\bpartnership(?:s)?\b",
    r"\bacquisition(?:s)?\b",
    r"\bmerger(?:s)?\b",
    r"\bleadership\b",
    r"\bmanagement\b",
    r"\bexecutive(?:s)?\b",
    r"\bceo\b",
    r"\bcfo\b",
    r"\bfounder(?:s)?\b",
    r"\bstrategy\b",
    r"\bcurrent events?\b",
    r"\bartificial intelligence\b",
    r"\bgenerative ai\b",
    r"\bai\b",
    r"\blaunch(?:ed|es|ing)?\b",
    r"\breleas(?:e|ed|es|ing)\b",
    r"\bdevelopments?\b",
    r"\bworking on\b",
    r"\bdoing in\b",
)


NEWS_CONTEXT_PATTERNS = (
    r"\blatest\b",
    r"\brecent(?:ly)?\b",
    r"\bhistory\b",
    r"\bhistorical\b",
)

FINANCIAL_PATTERNS = (
    r"\bstock(?:s)?\b",
    r"\bshare price\b",
    r"\bmarket price\b",
    r"\bprice performance\b",
    r"\bticker\b",
    r"\bpublicly traded\b",
    r"\bpublic company\b",
    r"\bmarket cap(?:italization)?\b",
    r"\bearnings\b",
    r"\bquarterly results?\b",
    r"\bquarter(?:ly)?\b",
    r"\bfinancial(?:s| performance| results?)?\b",
    r"\bfilings?\b",
    r"\b10[- ]?[kq]\b",
    r"\bsec\b",
    r"\brevenue\b",
    r"\bnet income\b",
    r"\boperating income\b",
    r"\bearnings per share\b",
    r"\beps\b",
    r"\bcash flow\b",
    r"\bbalance sheet\b",
    r"\bincome statement\b",
    r"\bgross margin\b",
    r"\boperating margin\b",
    r"\bguidance\b",
    r"\bprofit(?:ability|s)?\b",
    r"\bvaluation\b",
    r"\bdividend(?:s)?\b",
    r"\bipo\b",
    r"\binvestor(?:s)?\b",
    r"\bshares?\b",
)

MARKET_DATA_PATTERNS = (
    r"\bstock(?:s)?\b",
    r"\bshare price\b",
    r"\bmarket price\b",
    r"\bprice performance\b",
    r"\bmarket cap(?:italization)?\b",
    r"\bquote\b",
    r"\btrading at\b",
    r"\bshares?\b",
)

CURRENT_FINANCIAL_CONTEXT_PATTERNS = (
    r"\bcurrent(?:ly)?\b",
    r"\blatest\b",
    r"\bmost recent\b",
    r"\brecent(?:ly)?\b",
    r"\btoday\b",
    r"\bnow\b",
    r"\breal[- ]?time\b",
    r"\bas of\b",
)

FINANCIAL_RESULTS_PATTERNS = (
    r"\bearnings\b",
    r"\bquarterly results?\b",
    r"\bfinancial results?\b",
    r"\bfinancial performance\b",
    r"\brevenue\b",
    r"\bnet income\b",
    r"\boperating income\b",
    r"\bearnings per share\b",
    r"\beps\b",
    r"\bcash flow\b",
    r"\bgross margin\b",
    r"\boperating margin\b",
    r"\bguidance\b",
)

FILING_CONTEXT_PATTERNS = (
    r"\bfilings?\b",
    r"\b10[- ]?[kq]\b",
    r"\bsec\b",
    r"\bhistor(?:y|ical|ically)\b",
    r"\bover time\b",
    r"\btrend(?:s|ed|ing)?\b",
    r"\bchange(?:d|s| over time)?\b",
    r"\bquarter[- ]over[- ]quarter\b",
    r"\byear[- ]over[- ]year\b",
    r"\bcompare(?:d|s|ing|ison)?\b",
    r"\bpast \d+ (?:quarters?|years?)\b",
)

UNSUPPORTED_PATTERNS = (
    r"\bweather\b",
    r"\bforecast\b",
    r"\bsports?\b",
    r"\brecipe\b",
    r"\bmedical\b",
    r"\bdiagnos(?:e|is|tic)\b",
    r"\bprescription\b",
    r"\blegal advice\b",
    r"\btravel itinerary\b",
    r"\bflight(?:s)?\b",
    r"\brestaurant(?:s)?\b",
)

GENERIC_PROPER_NOUNS = {
    # Capitalized sentence openers are grammar, not additional companies.
    "are",
    "can",
    "could",
    "did",
    "do",
    "does",
    "had",
    "has",
    "have",
    "is",
    "please",
    "should",
    "was",
    "were",
    "would",
    "user",
    "resolve",
    "preserve",
    "get",
    "agent",
    "ai",
    "a2a",
    "api",
    "artificial intelligence",
    "analyze",
    "ceo",
    "cfo",
    "compare",
    "cover",
    "create",
    "current",
    "describe",
    "draft",
    "evaluate",
    "executive",
    "executives",
    "explain",
    "generate",
    "give",
    "financial",
    "founder",
    "founders",
    "generative ai",
    "how",
    "ipo",
    "latest",
    "leadership",
    "management",
    "mcp",
    "news",
    "recent",
    "sec",
    "stock",
    "show",
    "summarize",
    "tell",
    "the",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "write",
}

TICKER_EXCLUSIONS = {
    "A2A",
    "AI",
    "API",
    "CEO",
    "CFO",
    "IPO",
    "LLM",
    "MCP",
    "RAG",
    "SEC",
    "UK",
    "US",
    "USA",
    "USD",
}

COMPANY_SUFFIXES = {
    "co",
    "company",
    "corp",
    "corporation",
    "group",
    "holdings",
    "inc",
    "incorporated",
    "limited",
    "ltd",
    "plc",
}


class PolicyError(ValueError):
    """Raised when a model plan attempts to cross a deterministic boundary."""


def _matches_any(text: str, patterns: Sequence[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _domain_flags(text: str) -> tuple[bool, bool]:
    """Return news and financial intent while treating recency as a modifier."""

    has_financial = _matches_any(text, FINANCIAL_PATTERNS)
    has_news = _matches_any(text, NEWS_PATTERNS) or (
        _matches_any(text, NEWS_CONTEXT_PATTERNS) and not has_financial
    )
    return has_news, has_financial


def _financial_evidence_flags(text: str) -> tuple[bool, bool]:
    """Identify when the Financial Agent needs current data, filings, or both."""

    has_market_data = _matches_any(text, MARKET_DATA_PATTERNS)
    has_current_context = _matches_any(text, CURRENT_FINANCIAL_CONTEXT_PATTERNS)
    has_results = _matches_any(text, FINANCIAL_RESULTS_PATTERNS)
    has_filing_context = _matches_any(text, FILING_CONTEXT_PATTERNS)

    # Price, quote, trading, and market-cap questions belong to the specialist's
    # market-data path even when the requested date is historical. Filing vectors
    # are not authoritative market-price evidence.
    requires_current_data = has_market_data or (
        has_results and (has_current_context or not has_filing_context)
    )
    requires_filings = has_filing_context or has_results
    return requires_current_data, requires_filings


def normalize_company(value: str) -> str:
    """Normalize a company label for comparison without guessing its identity."""

    value = value.replace("’", "'")
    value = re.sub(r"['’]s\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"[^a-zA-Z0-9]+", " ", value).strip().lower()
    tokens = value.split()
    if tokens and tokens[0] == "the":
        tokens = tokens[1:]
    tokens = [token for token in tokens if token not in COMPANY_SUFFIXES]
    return " ".join(tokens)


def _deduplicate(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = re.sub(r"\s+", " ", value).strip(" \t\r\n.,:;!?()[]{}\"'")
        cleaned = re.sub(r"['’]s$", "", cleaned)
        normalized = normalize_company(cleaned)
        if not cleaned or not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(cleaned)
    return result


def extract_explicit_tickers(text: str) -> list[str]:
    """Extract ticker symbols only when the user supplies a ticker-shaped token."""

    matches: list[str] = []
    patterns = (
        r"\$([A-Z][A-Z0-9.]{0,5})\b",
        r"\b(?:NASDAQ|NYSE|AMEX)\s*:\s*([A-Z][A-Z0-9.]{0,5})\b",
        r"\bticker(?:\s+symbol)?\s+(?:is\s+)?([A-Z][A-Z0-9.]{0,5})\b",
    )
    for pattern in patterns:
        matches.extend(re.findall(pattern, text))

    # Common shorthand such as "NVDA stock" is explicit enough to preserve,
    # while exclusions prevent every acronym from becoming a security symbol.
    for symbol in re.findall(r"\b([A-Z]{1,5}(?:\.[A-Z])?)\b", text):
        if symbol not in TICKER_EXCLUSIONS and _matches_any(text, FINANCIAL_PATTERNS):
            matches.append(symbol)

    return _deduplicate(matches)


def extract_company_candidates(text: str) -> list[str]:
    """Find visible proper-noun company hints without resolving them to securities."""

    candidates: list[str] = []

    # Handles names such as "Bank of America" while deliberately treating
    # "Apple and Microsoft" as two candidates.
    token = r"[A-Z][A-Za-z0-9&'-]*(?:\.[A-Za-z0-9&'-]+)*"
    phrase = rf"{token}(?:\s+(?:(?:of|the|&)\s+)?{token}){{0,3}}"
    for match in re.findall(phrase, text):
        words = match.split()
        while words and words[0].lower().strip(".'") in GENERIC_PROPER_NOUNS:
            words.pop(0)
        while words and words[-1].lower().strip(".'") in GENERIC_PROPER_NOUNS:
            words.pop()
        if words:
            candidate = " ".join(words)
            if normalize_company(candidate) not in GENERIC_PROPER_NOUNS:
                candidates.append(candidate)

    ticker_norms = {normalize_company(value) for value in extract_explicit_tickers(text)}
    return [
        value
        for value in _deduplicate(candidates)
        if normalize_company(value) not in ticker_norms
    ]


def _literal_pattern(value: str) -> str:
    escaped = re.escape(value.strip()).replace(r"\ ", r"\s+")
    return rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])"


def _ticker_is_linked_to_company(text: str, company: str, ticker: str) -> bool:
    """Return whether the user explicitly stated a company/ticker relationship."""

    company_pattern = _literal_pattern(company)
    ticker_pattern = _literal_pattern(ticker)
    patterns = (
        rf"\bticker(?:\s+symbol)?\s+(?:for|of)\s+(?:the\s+)?"
        rf"{company_pattern}\s+(?:is|=|:)\s*{ticker_pattern}",
        rf"{company_pattern}(?:['’]s)?\s+ticker(?:\s+symbol)?\s*"
        rf"(?:is|=|:)\s*{ticker_pattern}",
        rf"{ticker_pattern}\s+(?:is|=)\s+(?:the\s+)?ticker"
        rf"(?:\s+symbol)?\s+(?:for|of)\s+(?:the\s+)?{company_pattern}",
        rf"{company_pattern}\s*(?:\(|\[)\s*"
        rf"(?:(?:NASDAQ|NYSE|AMEX)\s*:\s*)?{ticker_pattern}\s*(?:\)|\])",
        rf"{company_pattern}\s+(?:trades?|is\s+traded)\s+(?:under|as)\s+"
        rf"(?:the\s+)?(?:ticker(?:\s+symbol)?\s+)?{ticker_pattern}",
        rf"{company_pattern}\s*,?\s+(?:with\s+)?ticker(?:\s+symbol)?\s*"
        rf"(?:is|=|:)?\s*{ticker_pattern}",
    )
    return _matches_any(text, patterns)


def extract_ticker_company_links(
    text: str,
    *,
    company_candidates: Sequence[str] | None = None,
    explicit_tickers: Sequence[str] | None = None,
) -> dict[str, list[str]]:
    """Associate user-supplied company and ticker identities without resolving symbols."""

    companies = list(company_candidates or extract_company_candidates(text))
    tickers = list(explicit_tickers or extract_explicit_tickers(text))
    links: dict[str, list[str]] = {}

    for company in companies:
        linked = [
            ticker
            for ticker in tickers
            if _ticker_is_linked_to_company(text, company, ticker)
        ]
        if linked:
            links[company] = _deduplicate(linked)

    # One company and one ticker in the same financial request are treated as
    # one routing identity. The Financial Agent still owns symbol verification,
    # so this avoids a needless clarification without making the orchestrator a
    # security master.
    looks_like_comparison = bool(
        re.search(r"\b(?:compare|comparison|versus|vs\.?)\b", text, re.IGNORECASE)
    )
    if not looks_like_comparison and len(companies) == 1 and len(tickers) == 1:
        company_pattern = _literal_pattern(companies[0])
        ticker_pattern = _literal_pattern(tickers[0])
        looks_like_comparison = bool(
            re.search(
                rf"(?:{company_pattern}\s+(?:and|or|versus|vs\.?)\s+{ticker_pattern}|"
                rf"{ticker_pattern}\s+(?:and|or|versus|vs\.?)\s+{company_pattern})",
                text,
                re.IGNORECASE,
            )
        )

    if (
        not links
        and len(companies) == 1
        and len(tickers) == 1
        and _matches_any(text, FINANCIAL_PATTERNS)
        and not looks_like_comparison
    ):
        links[companies[0]] = [tickers[0]]

    return links


def collapse_linked_identities(
    company_candidates: Sequence[str],
    explicit_tickers: Sequence[str],
    ticker_company_links: dict[str, list[str]],
) -> list[str]:
    """Count an explicitly linked company/ticker pair as one user identity."""

    linked_tickers = {
        normalize_company(ticker)
        for tickers in ticker_company_links.values()
        for ticker in tickers
    }
    unlinked_tickers = [
        ticker
        for ticker in explicit_tickers
        if normalize_company(ticker) not in linked_tickers
    ]
    return _deduplicate([*company_candidates, *unlinked_tickers])


def linked_ticker_security_ambiguity(
    ticker_company_links: dict[str, list[str]],
) -> bool:
    """Require clarification when one company is linked to several securities."""

    return any(
        len(_deduplicate(tickers)) > 1
        for tickers in ticker_company_links.values()
    )


def identity_aliases(
    identity: str,
    ticker_company_links: dict[str, list[str]],
) -> list[str]:
    """Return user-supplied aliases for one active company identity."""

    aliases = [identity]
    identity_norm = normalize_company(identity)
    for company, tickers in ticker_company_links.items():
        if normalize_company(company) == identity_norm:
            aliases.extend(tickers)
    return _deduplicate(aliases)


def _message_text(message: object) -> tuple[str, str]:
    if isinstance(message, dict):
        return str(message.get("role") or ""), str(message.get("content") or "")
    role = str(getattr(message, "role", ""))
    content = getattr(message, "content", "")
    return role, str(content or "")


def _latest_user_text(messages: Sequence[object]) -> str:
    for message in reversed(messages):
        role, content = _message_text(message)
        if role == "user":
            return content.strip()
    return ""


def _prior_user_text(messages: Sequence[object]) -> str:
    user_messages = [
        content.strip()
        for role, content in (_message_text(message) for message in messages)
        if role == "user" and content.strip()
    ]
    return user_messages[-2] if len(user_messages) >= 2 else ""


def _prior_identity_text(messages: Sequence[object]) -> str:
    """Return the most recent earlier user turn containing a company or ticker."""

    user_messages = [
        content.strip()
        for role, content in (_message_text(message) for message in messages)
        if role == "user" and content.strip()
    ]
    for content in reversed(user_messages[:-1]):
        if extract_company_candidates(content) or extract_explicit_tickers(content):
            return content
    return ""


def _clarification_count(messages: Sequence[object]) -> int:
    """Count clarification only for the currently unresolved request."""

    for message in reversed(messages):
        role, content = _message_text(message)
        if role != "assistant":
            continue
        match = re.search(r"\broute=([a-z_]+)\b", content, flags=re.IGNORECASE)
        if match is None:
            return 0
        return 1 if match.group(1).lower() == "clarification" else 0
    return 0


def _domain_reply(text: str) -> RouteName | None:
    normalized = re.sub(r"[^a-z]+", " ", text.lower()).strip()
    if normalized in {"both", "both please", "news and financial", "financial and news"}:
        return "combined"
    if normalized in {"news", "news please", "company news", "developments"}:
        return "news"
    if normalized in {
        "financial",
        "financial please",
        "financials",
        "stock",
        "stock information",
    }:
        return "financial"
    return None


class NewsFinancePolicyManager:
    """Classify requests and validate model-proposed control flow."""

    def assess(self, messages: Sequence[object]) -> PolicyAssessment:
        current = _latest_user_text(messages)
        prior = _prior_user_text(messages)
        prior_identity = _prior_identity_text(messages)
        company_candidates = extract_company_candidates(current)
        history_candidates = extract_company_candidates(prior_identity)
        explicit_tickers = extract_explicit_tickers(current)
        history_explicit_tickers = extract_explicit_tickers(prior_identity)
        ticker_company_links = extract_ticker_company_links(
            current,
            company_candidates=company_candidates,
            explicit_tickers=explicit_tickers,
        )
        history_ticker_company_links = extract_ticker_company_links(
            prior_identity,
            company_candidates=history_candidates,
            explicit_tickers=history_explicit_tickers,
        )

        has_news, has_financial = _domain_flags(current)
        prior_news, prior_financial = _domain_flags(prior)
        requires_current_financial_data, requires_filing_context = (
            _financial_evidence_flags(current)
        )
        prior_requires_current_data, prior_requires_filing_context = (
            _financial_evidence_flags(prior)
        )
        has_unsupported = _matches_any(current, UNSUPPORTED_PATTERNS)
        clarification_count = _clarification_count(messages)

        current_identities = collapse_linked_identities(
            company_candidates,
            explicit_tickers,
            ticker_company_links,
        )
        security_ambiguity = linked_ticker_security_ambiguity(
            ticker_company_links,
        )

        direct_reply = _domain_reply(current)
        if direct_reply:
            route: RouteName = direct_reply
            reason = "The user selected a domain after a prior clarification."
            has_news = direct_reply in {"news", "combined"}
            has_financial = direct_reply in {"financial", "combined"}
        elif len(current_identities) > 1 or security_ambiguity:
            route = "clarification"
            reason = (
                "The request identifies more than one company or security."
                if len(current_identities) > 1
                else "The request identifies more than one security for one company."
            )
        elif (
            clarification_count >= 1
            and len(current_identities) == 1
            and not has_news
            and not has_financial
            and not has_unsupported
            and (prior_news or prior_financial)
        ):
            if prior_news and prior_financial:
                route = "combined"
            elif prior_financial:
                route = "financial"
            else:
                route = "news"
            has_news = route in {"news", "combined"}
            has_financial = route in {"financial", "combined"}
            reason = (
                "The user supplied one company after clarification; the prior "
                "supported domain was preserved."
            )
        elif has_news and has_financial:
            route = "combined"
            reason = "The request contains both company-development and financial intent."
        elif has_financial:
            route = "financial"
            reason = "The request contains financial or public-market intent."
        elif has_news:
            route = "news"
            reason = "The request contains company-news or strategy intent."
        elif has_unsupported:
            route = "unsupported"
            reason = "The request is outside company news and financial analysis."
        else:
            route = "clarification"
            reason = (
                "The request does not identify whether news, financial information, "
                "or both is needed."
            )

        if route in {"financial", "combined"} and not (
            requires_current_financial_data or requires_filing_context
        ):
            requires_current_financial_data = prior_requires_current_data
            requires_filing_context = prior_requires_filing_context
        if route not in {"financial", "combined"}:
            requires_current_financial_data = False
            requires_filing_context = False

        return PolicyAssessment(
            route_hint=route,
            route_reason=reason,
            company_candidates=company_candidates,
            history_company_candidates=history_candidates,
            explicit_tickers=explicit_tickers,
            history_explicit_tickers=history_explicit_tickers,
            ticker_company_links=ticker_company_links,
            history_ticker_company_links=history_ticker_company_links,
            clarification_count=clarification_count,
            supported_news=has_news,
            supported_financial=has_financial,
            requires_current_financial_data=requires_current_financial_data,
            requires_filing_context=requires_filing_context,
            explicit_unsupported=has_unsupported and not (has_news or has_financial),
        )

    def company_is_grounded(
        self,
        company: str,
        messages: Sequence[object],
    ) -> bool:
        """Require the planned company identity to appear in user-provided text."""

        company_norm = normalize_company(company)
        if not company_norm:
            return False

        user_text = " ".join(
            content
            for role, content in (_message_text(message) for message in messages)
            if role == "user"
        )
        user_norm = normalize_company(user_text)
        if company_norm in user_norm:
            return True

        company_tokens = company_norm.split()
        if len(company_tokens) > 1:
            return all(token in user_norm.split() for token in company_tokens)
        return company_norm in user_norm.split()

    def validate_plan(
        self,
        plan: RoutingPlan,
        assessment: PolicyAssessment,
        messages: Sequence[object],
    ) -> None:
        """Reject model plans that broaden scope, identity, or specialist access."""

        current_identities = collapse_linked_identities(
            assessment.company_candidates,
            assessment.explicit_tickers,
            assessment.ticker_company_links,
        )
        history_identities = collapse_linked_identities(
            assessment.history_company_candidates,
            assessment.history_explicit_tickers,
            assessment.history_ticker_company_links,
        )
        current_security_ambiguity = linked_ticker_security_ambiguity(
            assessment.ticker_company_links,
        )
        history_security_ambiguity = linked_ticker_security_ambiguity(
            assessment.history_ticker_company_links,
        )
        active_identities = current_identities or history_identities
        active_links = (
            assessment.ticker_company_links
            if current_identities
            else assessment.history_ticker_company_links
        )
        active_identity = (
            active_identities[0]
            if len(active_identities) == 1
            and not (
                current_security_ambiguity
                if current_identities
                else history_security_ambiguity
            )
            else ""
        )
        allowed_identities = (
            identity_aliases(active_identity, active_links)
            if active_identity
            else []
        )

        allowed_routes: set[RouteName]
        if assessment.route_hint in {"news", "financial", "combined"}:
            current_identity_ambiguity = (
                len(current_identities) > 1 or current_security_ambiguity
            )
            history_identity_ambiguity = (
                len(history_identities) > 1 or history_security_ambiguity
            )
            if current_identity_ambiguity or (
                not current_identities and history_identity_ambiguity
            ):
                allowed_routes = {"clarification"}
            else:
                allowed_routes = {assessment.route_hint, "clarification"}
        elif assessment.route_hint == "clarification":
            allowed_routes = {"clarification", "unsupported"}
            if active_identity:
                allowed_routes.update({"news", "financial", "combined"})
        else:
            allowed_routes = {"unsupported", "clarification"}
            if active_identity and (
                assessment.supported_news or assessment.supported_financial
            ):
                allowed_routes.update({"news", "financial", "combined"})

        if plan.request_type not in allowed_routes:
            raise PolicyError(
                f"Model route {plan.request_type!r} conflicts with deterministic route "
                f"{assessment.route_hint!r}."
            )

        candidates = _deduplicate(plan.company_candidates)

        if plan.request_type in {"news", "financial", "combined"}:
            planned_company = normalize_company(plan.company)
            allowed_company_values = {
                normalize_company(value) for value in allowed_identities
            }
            candidate_values = {normalize_company(value) for value in candidates}
            if candidate_values and not candidate_values.issubset(
                allowed_company_values
            ):
                raise PolicyError(
                    "The model identified a company or security outside the active identity."
                )

            if allowed_company_values:
                if planned_company not in allowed_company_values:
                    raise PolicyError(
                        "The planned company does not match the active user-provided identity."
                    )
            elif not self.company_is_grounded(plan.company, messages):
                raise PolicyError("The planned company is not grounded in the conversation.")

            # The model-proposed request text is not authoritative. The host replaces
            # it with an application-owned specialist request after this validation.
            # Route shape, specialist allowlisting, and company identity are validated
            # above, so rejecting harmless wording here only creates planner fallbacks.
            # allowed_tickers = {
            #     value.upper()
            #     for value in [
            #         *assessment.explicit_tickers,
            #         *assessment.history_explicit_tickers,
            #     ]
            # }
            # all_user_identities = _deduplicate(
            #     [
            #         *assessment.company_candidates,
            #         *assessment.explicit_tickers,
            #         *assessment.history_company_candidates,
            #         *assessment.history_explicit_tickers,
            #     ]
            # )
            # for call in plan.agent_calls:
            #     call_messages = [{"role": "user", "content": call.request}]
            #     if allowed_identities:
            #         preserves_identity = any(
            #             self.company_is_grounded(identity, call_messages)
            #             for identity in allowed_identities
            #         )
            #     else:
            #         preserves_identity = self.company_is_grounded(
            #             plan.company, call_messages
            #         )
            #     if not preserves_identity:
            #         raise PolicyError(
            #             f"The {call.agent} specialist request does not preserve the "
            #             "company identity."
            #         )

            #     normalized_request = normalize_company(call.request)
            #     for identity in all_user_identities:
            #         identity_norm = normalize_company(identity)
            #         if not identity_norm or identity_norm in allowed_company_values:
            #             continue
            #         pattern = (
            #             r"\b"
            #             + re.escape(identity_norm).replace(r"\ ", r"\s+")
            #             + r"\b"
            #         )
            #         if re.search(pattern, normalized_request):
            #             raise PolicyError(
            #                 f"The {call.agent} request includes another user-provided "
            #                 "company or security."
            #             )

            #     introduced_tickers = {
            #         value.upper() for value in extract_explicit_tickers(call.request)
            #     } - allowed_tickers
            #     if introduced_tickers:
            #         raise PolicyError(
            #             "The specialist request introduced a ticker that the user did not supply."
            #         )

    def clarification_exhausted(self, assessment: PolicyAssessment) -> bool:
        return assessment.clarification_count >= 1


__all__ = [
    "NewsFinancePolicyManager",
    "PolicyError",
    "collapse_linked_identities",
    "extract_company_candidates",
    "extract_explicit_tickers",
    "extract_ticker_company_links",
    "identity_aliases",
    "linked_ticker_security_ambiguity",
    "normalize_company",
]
