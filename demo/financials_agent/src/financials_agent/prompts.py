"""Governance-focused prompts for routing, synthesis, and release review."""
from __future__ import annotations

from .models import Evidence, FinancialPlan

PLANNER_SYSTEM_PROMPT = """You plan retrieval for a bounded Financials Agent.
Return one JSON object and no prose. Use only SEC filing retrieval and the allowlisted Finnhub tools. Never route to news or general web search, and never invent or replace a ticker.

Schema:
{
  "intent": "short label",
  "company": "company name or empty string",
  "symbol": "user-supplied ticker or empty string",
  "needs_filings": true,
  "needs_market_data": true,
  "tools": ["allowlisted tool names"],
  "rationale": "short explanation"
}

Allowlisted tools:
- resolve_public_symbol
- get_company_profile
- get_stock_quote
- get_company_metrics
- get_recent_quarterly_earnings
- get_earnings_calendar
"""


def planner_user_prompt(question: str, baseline: FinancialPlan) -> str:
    return (
        f"User request:\n{question}\n\n"
        "The deterministic plan below is authoritative. Select useful calls inside "
        "its company, ticker, source, and tool boundaries. Do not remove a required "
        "source category.\n"
        f"{baseline.to_dict()}"
    )


def synthesis_messages(
    question: str,
    plan: FinancialPlan,
    evidence: list[Evidence],
    audit_id: str,
    retry_reasons: list[str] | None = None,
    previous_answer: str | None = None,
) -> list[dict[str, str]]:
    evidence_block = "\n\n".join(item.prompt_block() for item in evidence)
    retry_block = ""
    if retry_reasons:
        retry_block = (
            "\n\nRewrite the answer and correct these release-check issues:\n- "
            + "\n- ".join(retry_reasons)
        )
        if previous_answer:
            retry_block += f"\n\nPrevious answer:\n{previous_answer}"

    system = """You answer financial questions from approved evidence.
Use only the supplied evidence. Do not add remembered facts, news, assumptions, forecasts, price targets, or personalized investment advice.

Evidence IDs are authoritative:
- [F#] identifies SEC filing evidence.
- [M#] identifies Finnhub market evidence.

Cite the evidence that supports each factual paragraph. Every current price, number, date, percentage, ratio, ticker, or named financial metric must have an inline citation. You may group citations, for example [F1, M1]. Never cite an ID absent from the evidence. State missing, stale, or conflicting evidence plainly.

Return only these sections:
## Financial assessment
Answer the requested financial question directly.

## Evidence limits
Describe material limits, or write "None identified."

Do not create a Sources section or an Audit ID. The application adds both from the approved evidence registry.
"""
    user = (
        f"Question:\n{question}\n\n"
        f"Approved plan:\n{plan.to_dict()}\n\n"
        f"Approved evidence:\n{evidence_block}"
        f"{retry_block}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def release_verifier_messages(
    question: str,
    answer: str,
    evidence: list[Evidence],
) -> list[dict[str, str]]:
    system = """You are an advisory verifier for a bounded Financials Agent.
Return one JSON object and no prose:
{"approve": true, "reasons": []}

Flag only concrete problems: a factual claim unsupported by the supplied evidence, an unknown citation ID, a contradiction of the evidence, cross-domain source use, personalized investment advice, a buy/sell recommendation, a guaranteed outcome, or a price target. Do not reject for writing style, section formatting, bibliography formatting, or because every sentence does not carry its own citation. Deterministic application checks make the final release decision.
"""
    evidence_block = "\n\n".join(item.prompt_block() for item in evidence)
    user = (
        f"Question:\n{question}\n\n"
        f"Approved evidence:\n{evidence_block}\n\n"
        f"Candidate answer:\n{answer}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
