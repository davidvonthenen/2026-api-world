"""Deterministic release checks that remain authoritative over model opinions."""
from __future__ import annotations

from collections import Counter
import re
from typing import Any

from ..common.logging import get_logger, log_payload
from .models import (
    Evidence,
    VerificationResult,
)
from .planner import extract_json_object

LOGGER = get_logger(__name__)

_CITATION_GROUP_RE = re.compile(
    r"\[((?:[FM]\d+)(?:\s*,\s*[FM]\d+)*)\]"
)
_EVIDENCE_PREFIX_BY_KIND = {
    "financial_filing": "F",
    "market_data": "M",
}
_AUDIT_ID_LINE_RE = re.compile(r"^\s*Audit ID\s*:", re.IGNORECASE)
_ADVICE_PATTERNS = (
    r"\byou should buy\b",
    r"\byou should sell\b",
    r"\bi recommend (?:buying|selling)\b",
    r"\bguaranteed return\b",
    r"\brisk[- ]free return\b",
    r"\bwill definitely (?:rise|fall|increase|decrease)\b",
    r"\bprice target is\b",
)
_CROSS_DOMAIN_PATTERNS = (
    r"\btavily\b",
    r"\bnews agent corpus\b",
    r"\bclinical record\b",
    r"\bemployee record\b",
    r"\bhr database\b",
)


def _section(text: str, heading: str, next_heading: str | None = None) -> str:
    start = text.find(heading)
    if start < 0:
        return ""
    start += len(heading)
    if next_heading:
        end = text.find(next_heading, start)
        if end >= 0:
            return text[start:end].strip()
    return text[start:].strip()


def _append_citations(line: str, citation_text: str) -> str:
    """Attach a model-supplied citation to the preceding statement."""
    stripped = line.rstrip()
    terminal = re.search(r"([.!?])$", stripped)
    if terminal:
        return f"{stripped[:-1].rstrip()} {citation_text}{terminal.group(1)}"
    return f"{stripped} {citation_text}"


def _inline(value: str) -> str:
    return " ".join(str(value).split())


def _citation_ids(text: str) -> list[str]:
    """Return individual IDs from single or grouped financial citations."""
    values: list[str] = []
    for group in _CITATION_GROUP_RE.findall(text):
        values.extend(part.strip() for part in group.split(",") if part.strip())
    return list(dict.fromkeys(values))


def _source_line(item: Evidence) -> str:
    source = _inline(item.source) or "Approved evidence"
    title = _inline(item.title)
    details = [source]
    if title and title.lower() != source.lower():
        details.append(title)
    details.append(f"as of {_inline(item.as_of) or 'not supplied'}")
    source_url = _inline(item.source_url)
    if source_url:
        details.append(source_url)
    return f"- [{item.evidence_id}] " + "; ".join(details)


def _canonicalize_sources(answer: str, evidence: list[Evidence]) -> str:
    """Render the Sources section from claim citations and approved evidence.

    The answer model decides which evidence supports each claim. The agent owns
    source metadata and can therefore render the bibliography deterministically
    without inventing a citation or changing claim-to-evidence associations.
    """
    heading = "## Sources"
    start = answer.find(heading)
    body = answer[:start].rstrip() if start >= 0 else answer.rstrip()
    cited_ids = _citation_ids(body)
    evidence_by_id = {item.evidence_id: item for item in evidence}
    source_lines = [
        _source_line(evidence_by_id[evidence_id])
        for evidence_id in cited_ids
        if evidence_id in evidence_by_id
    ]
    if not source_lines:
        source_lines = ["- None. No approved evidence was cited in the answer body."]

    return f"{body}\n\n{heading}\n" + "\n".join(source_lines)


def normalize_generated_answer(
    answer: str,
    audit_id: str,
    evidence: list[Evidence] | None = None,
) -> str:
    """Normalize harmless model formatting before deterministic verification.

    The function never invents or substitutes an evidence identifier. It only:

    * moves a citation-only line such as ``[M1]`` onto the immediately
      preceding substantive line;
    * renders the Sources section from citations already present in the answer
      body and the corresponding approved evidence metadata; and
    * replaces any model-written audit line with the server-assigned audit ID.

    Claims that contain no model-supplied citation remain uncited. When evidence
    checks are enabled, the answer must contain at least one valid approved
    citation, while claim-by-claim completeness remains advisory.
    """
    normalized_lines: list[str] = []
    for raw_line in answer.strip().splitlines():
        stripped = raw_line.strip()
        if _AUDIT_ID_LINE_RE.match(stripped):
            continue

        citation_ids = _citation_ids(stripped)
        citation_remainder = _CITATION_GROUP_RE.sub("", stripped).strip(" \t,;:.")
        if citation_ids and not citation_remainder:
            previous_index = len(normalized_lines) - 1
            while previous_index >= 0 and not normalized_lines[previous_index].strip():
                previous_index -= 1
            if (
                previous_index >= 0
                and not normalized_lines[previous_index].lstrip().startswith("#")
            ):
                citation_text = " ".join(f"[{evidence_id}]" for evidence_id in citation_ids)
                normalized_lines[previous_index] = _append_citations(
                    normalized_lines[previous_index],
                    citation_text,
                )
                continue

        normalized_lines.append(raw_line.rstrip())

    while normalized_lines and not normalized_lines[-1].strip():
        normalized_lines.pop()

    normalized = "\n".join(normalized_lines)
    if evidence is not None:
        normalized = _canonicalize_sources(normalized, evidence)

    normalized = normalized.rstrip()
    if normalized:
        normalized += "\n\n"
    return normalized + f"Audit ID: {audit_id}"


def verify_answer(
    answer: str,
    evidence: list[Evidence],
    audit_id: str,
    *,
    release_checks_enabled: bool = True,
    policy_checks_enabled: bool = True,
    evidence_checks_enabled: bool = True,
) -> VerificationResult:
    """Run independently configurable release, policy, and evidence checks."""
    reasons: list[str] = []
    evidence_ids = [item.evidence_id for item in evidence]
    allowed_ids = set(evidence_ids)
    cited_ids = _citation_ids(answer)
    invalid_ids = sorted(set(cited_ids) - allowed_ids)

    if release_checks_enabled:
        if not answer.strip():
            reasons.append("The answer is empty.")
        if "## Financial assessment" not in answer:
            reasons.append("Missing required section: ## Financial assessment.")
        if f"Audit ID: {audit_id}" not in answer:
            reasons.append("The answer does not contain the assigned audit identifier.")

    if evidence_checks_enabled:
        duplicate_ids = sorted(
            evidence_id
            for evidence_id, count in Counter(evidence_ids).items()
            if count > 1
        )
        if duplicate_ids:
            reasons.append(f"Duplicate evidence identifiers: {', '.join(duplicate_ids)}.")

        for item in evidence:
            expected_prefix = _EVIDENCE_PREFIX_BY_KIND.get(item.kind)
            if expected_prefix and not re.fullmatch(
                rf"{expected_prefix}\d+",
                item.evidence_id,
            ):
                reasons.append(
                    f"Evidence {item.evidence_id} uses the wrong namespace for "
                    f"{item.kind}; expected {expected_prefix}# identifiers."
                )

        if invalid_ids:
            reasons.append(f"Unknown evidence citations: {', '.join(invalid_ids)}.")
        if evidence and not (set(cited_ids) & allowed_ids):
            reasons.append("The answer contains no valid evidence citation.")

        source_section = _section(answer, "## Sources")
        for evidence_id in cited_ids:
            if evidence_id in allowed_ids and f"[{evidence_id}]" not in source_section:
                reasons.append(
                    f"Cited evidence {evidence_id} is missing from the Sources section."
                )

    if policy_checks_enabled:
        lowered = answer.lower()
        for pattern in _ADVICE_PATTERNS:
            if re.search(pattern, lowered):
                reasons.append(
                    f"Prohibited investment-advice language matched: {pattern}."
                )
        for pattern in _CROSS_DOMAIN_PATTERNS:
            if re.search(pattern, lowered):
                reasons.append(f"Cross-domain source leakage matched: {pattern}.")

    result = VerificationResult(
        approved=not reasons,
        reasons=reasons,
        cited_ids=cited_ids,
        invalid_ids=invalid_ids,
    )
    log_payload(
        LOGGER,
        "Deterministic release-verifier output",
        result.to_dict(),
    )
    return result


def _approval_value(payload: dict[str, Any]) -> bool | None:
    """Parse a semantically clear model approval without trusting ambiguity."""
    raw = payload.get("approve", payload.get("approved"))
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    return None


def _model_reasons(payload: dict[str, Any]) -> list[str]:
    raw = payload.get("reasons", [])
    if isinstance(raw, list):
        return [str(reason).strip() for reason in raw if str(reason).strip()]
    if isinstance(raw, str) and raw.strip():
        return [raw.strip()]
    return []


def merge_model_verdict(
    deterministic: VerificationResult,
    raw_model_verdict: str,
) -> VerificationResult:
    """Record advisory Nemotron review without allowing it to veto deterministic checks."""
    try:
        payload: dict[str, Any] = extract_json_object(raw_model_verdict)
        model_approved = _approval_value(payload)
        model_reasons = _model_reasons(payload)

        # A malformed advisory verdict is observable, but it cannot veto a
        # deterministic pass. The previous implementation treated every value
        # other than the literal Python boolean True as a rejection, including
        # common model outputs such as "true" and missing approval fields.
        if model_approved is None:
            result = VerificationResult(
                approved=deterministic.approved,
                reasons=[
                    *deterministic.reasons,
                    "Model verifier returned an invalid approval value.",
                ],
                cited_ids=deterministic.cited_ids,
                invalid_ids=deterministic.invalid_ids,
                model_checked=False,
                model_approved=None,
            )
            log_payload(LOGGER, "Merged release-verifier output", result.to_dict())
            return result

        reasons = list(deterministic.reasons)
        if not model_approved:
            reasons.extend(
                f"Advisory verifier concern: {reason}" for reason in model_reasons
            )
            if not model_reasons:
                reasons.append(
                    "Advisory verifier returned approve=false without a reason."
                )
        result = VerificationResult(
            approved=deterministic.approved,
            reasons=list(dict.fromkeys(reasons)),
            cited_ids=deterministic.cited_ids,
            invalid_ids=deterministic.invalid_ids,
            model_checked=True,
            model_approved=model_approved,
        )
        log_payload(LOGGER, "Merged release-verifier output", result.to_dict())
        return result
    except Exception as exc:
        # Endpoint or parsing failures are observable but do not override a
        # deterministic pass. The answer still carries a verifier warning.
        result = VerificationResult(
            approved=deterministic.approved,
            reasons=[*deterministic.reasons, f"Model verifier unavailable: {exc}"],
            cited_ids=deterministic.cited_ids,
            invalid_ids=deterministic.invalid_ids,
            model_checked=False,
            model_approved=None,
        )
        log_payload(LOGGER, "Merged release-verifier output", result.to_dict())
        return result
