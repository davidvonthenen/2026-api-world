"""Append-only audit persistence for governed news requests."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from ..common.config import Settings
from ..common.logging import get_logger
from .models import EvidenceItem
from .utils import json_safe, sha256_text, utc_now_iso

LOGGER = get_logger(__name__)

_ALLOWED_REQUEST_METADATA = {
    "user_id",
    "tenant_id",
    "role",
    "jurisdiction",
    "trace_id",
    "request_id",
    "client_id",
}


class AuditLogger:
    """Write one compact JSON Lines record per completed request."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.path = Path(settings.audit_log_path).expanduser()
        self._lock = threading.Lock()

    @staticmethod
    def select_request_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
        if not metadata:
            return {}
        return {
            key: json_safe(value)
            for key, value in metadata.items()
            if key in _ALLOWED_REQUEST_METADATA
        }

    def request_fields(self, query: str) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "query_sha256": sha256_text(query),
            "query_length": len(query),
        }
        if self.settings.audit_include_query:
            fields["query"] = query
        return fields

    @staticmethod
    def evidence_fields(evidence: list[EvidenceItem]) -> list[dict[str, Any]]:
        return [
            {
                "citation_id": item.citation_id,
                "source_kind": item.source_kind,
                "source_id": item.source_id,
                "title": item.title,
                "path": item.path,
                "url": item.url,
                "category": item.category,
                "chunk_index": item.chunk_index,
                "score": item.score,
                "published_at": item.published_at,
                "retrieved_at": item.retrieved_at,
                "content_sha256": sha256_text(item.text),
            }
            for item in evidence
        ]

    def write(self, record: dict[str, Any]) -> None:
        """Append a durable JSON record; failures are allowed to fail the request closed."""
        payload = {
            "schema_version": "1.0",
            "recorded_at": utc_now_iso(),
            **json_safe(record),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._lock:
            fd = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
        LOGGER.info(
            "event=audit_written audit_id=%s path=%s",
            record.get("audit_id", "unknown"),
            self.path,
        )


__all__ = ["AuditLogger"]
