"""Small parsing and normalization helpers."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_CITATION = re.compile(r"\[((?:H|W)\d+(?:\s*,\s*(?:H|W)\d+)*)\]")
_SOURCE_HEADING = re.compile(
    r"^\s{0,3}#{0,3}\s*(?:sources|references)\s*:?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)

_HOST_NEWS_PREFIX = "You are the News specialist."
_HOST_USER_REQUEST_MARKER = "\nUser request:\n"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def clean_text(value: object, max_chars: int | None = None) -> str:
    text = _CONTROL_CHARS.sub(" ", str(value or "")).strip()
    if max_chars is None or len(text) <= max_chars:
        return text
    shortened = text[:max_chars]
    last_space = shortened.rfind(" ")
    if last_space > max_chars // 2:
        shortened = shortened[:last_space]
    return shortened.rstrip() + "..."


def parse_json_object(value: str) -> dict[str, Any]:
    """Extract the first valid JSON object from model output."""
    cleaned = value.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    decoder = json.JSONDecoder()
    for index, character in enumerate(cleaned):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Model response did not contain a valid JSON object")



def extract_user_question(value: str) -> str:
    """Remove the known host instruction envelope while leaving direct questions untouched."""
    text = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text.startswith(_HOST_NEWS_PREFIX):
        return text
    marker_index = text.find(_HOST_USER_REQUEST_MARKER)
    if marker_index < 0:
        return text
    return text[marker_index + len(_HOST_USER_REQUEST_MARKER) :].strip()


def extract_citation_ids(answer: str) -> set[str]:
    citations: set[str] = set()
    for group in _CITATION.findall(answer):
        citations.update(part.strip() for part in group.split(",") if part.strip())
    return citations


def strip_model_source_section(answer: str) -> str:
    """Remove a model-invented source appendix; the application adds one itself."""
    match = _SOURCE_HEADING.search(answer)
    if match:
        answer = answer[: match.start()]
    return answer.strip()


def source_domain(url: str | None) -> str:
    if not url:
        return ""
    try:
        return urlparse(url).netloc.lower()
    except ValueError:
        return ""


def json_safe(value: Any) -> Any:
    """Convert arbitrary values to JSON-safe primitives for audit records."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return json_safe(model_dump())
    return str(value)


__all__ = [
    "clean_text",
    "extract_citation_ids",
    "extract_user_question",
    "json_safe",
    "parse_json_object",
    "sha256_text",
    "source_domain",
    "strip_model_source_section",
    "utc_now_iso",
]
