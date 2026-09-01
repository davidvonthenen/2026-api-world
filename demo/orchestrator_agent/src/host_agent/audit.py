"""Lightweight, tamper-evident JSONL audit trail for the conference demo."""

from __future__ import annotations

import argparse
import hashlib
import json
import threading

from pathlib import Path
from typing import Any


GENESIS_HASH = "0" * 64
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


class AuditError(RuntimeError):
    """Raised when the audit chain cannot be read, verified, or extended."""


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _entry_hash(entry_without_hash: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(entry_without_hash).encode("utf-8")).hexdigest()


def _path_lock(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _LOCKS_GUARD:
        if key not in _LOCKS:
            _LOCKS[key] = threading.Lock()
        return _LOCKS[key]


def _load_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AuditError(
                    f"Invalid JSON in audit log at line {line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise AuditError(
                    f"Audit log line {line_number} is not a JSON object."
                )
            entries.append(value)
    return entries


def verify_chain(path: str | Path) -> dict[str, Any]:
    """Verify every audit record and return the current chain head."""

    audit_path = Path(path)
    entries = _load_entries(audit_path)
    previous_hash = GENESIS_HASH

    for index, entry in enumerate(entries, start=1):
        recorded_previous = str(entry.get("previous_hash") or "")
        recorded_hash = str(entry.get("entry_hash") or "")
        if recorded_previous != previous_hash:
            raise AuditError(
                f"Audit chain previous_hash mismatch at record {index}."
            )

        payload = dict(entry)
        payload.pop("entry_hash", None)
        expected_hash = _entry_hash(payload)
        if recorded_hash != expected_hash:
            raise AuditError(f"Audit chain entry_hash mismatch at record {index}.")
        previous_hash = recorded_hash

    return {
        "path": str(audit_path),
        "records": len(entries),
        "head_hash": previous_hash,
        "valid": True,
    }


class AuditLog:
    """Append audited orchestration records after validating the existing chain."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, record: dict[str, Any]) -> dict[str, Any]:
        lock = _path_lock(self.path)
        with lock:
            state = verify_chain(self.path)
            entry = dict(record)
            entry["previous_hash"] = state["head_hash"]
            entry.pop("entry_hash", None)
            entry["entry_hash"] = _entry_hash(entry)

            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(_canonical_json(entry))
                handle.write("\n")
                handle.flush()
            return entry


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the Orchestrator Agent demonstration audit chain."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="./logs/orchestrator-audit.jsonl",
        help="Audit JSONL path",
    )
    args = parser.parse_args()
    print(json.dumps(verify_chain(args.path), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["AuditError", "AuditLog", "verify_chain"]
