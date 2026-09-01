"""Append-only, hash-chained JSONL audit records for governed executions."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Sequence

_LOCK = threading.Lock()
_GENESIS = "0" * 64


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class AuditLog:
    """Write one data-minimized completion record per Financials Agent request."""

    def __init__(self, path: str):
        self.path = Path(path)

    def append(self, record: dict[str, Any]) -> str:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK:
            previous_hash = _last_hash(self.path)
            payload = {**record, "previous_record_hash": previous_hash}
            record_hash = hashlib.sha256(
                (previous_hash + _canonical(payload)).encode("utf-8")
            ).hexdigest()
            payload["record_hash"] = record_hash
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(_canonical(payload) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            return record_hash


def _last_hash(path: Path) -> str:
    if not path.exists() or path.stat().st_size == 0:
        return _GENESIS

    valid, message, _ = verify_chain(path)
    if not valid:
        raise ValueError(f"Cannot append to an invalid audit chain: {message}")

    last_line = ""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last_line = line
    if not last_line:
        return _GENESIS

    payload = json.loads(last_line)
    record_hash = payload.get("record_hash")
    if not isinstance(record_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", record_hash):
        raise ValueError("Cannot append to an audit chain with an invalid final record hash")
    return record_hash


def verify_chain(path: str | Path) -> tuple[bool, str, int]:
    audit_path = Path(path)
    if not audit_path.exists():
        return False, f"Audit log not found: {audit_path}", 0

    previous_hash = _GENESIS
    count = 0
    with audit_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            count += 1
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                return False, f"Line {line_number} is invalid JSON: {exc}", count
            recorded_hash = str(payload.pop("record_hash", ""))
            if payload.get("previous_record_hash") != previous_hash:
                return False, f"Line {line_number} has a broken previous hash", count
            expected = hashlib.sha256(
                (previous_hash + _canonical(payload)).encode("utf-8")
            ).hexdigest()
            if recorded_hash != expected:
                return False, f"Line {line_number} has an invalid record hash", count
            previous_hash = recorded_hash
    return True, "Audit chain is valid", count


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Verify the Financials Agent audit hash chain")
    parser.add_argument("path", nargs="?", default="./var/financials-audit.jsonl")
    args = parser.parse_args(argv)
    valid, message, count = verify_chain(args.path)
    print(f"{message}; records={count}")
    raise SystemExit(0 if valid else 1)


if __name__ == "__main__":
    main()
