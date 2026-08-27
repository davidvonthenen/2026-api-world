"""Logging utilities shared by the Financials Agent services."""
from __future__ import annotations

import dataclasses
import json
import logging
import os
from typing import Optional

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"
_CONFIGURED_LOGGERS: set[logging.Logger] = set()
_SENSITIVE_TRANSPORT_LOGGERS = ("httpx", "httpcore")


def _configured_level() -> int:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    return getattr(logging, level_name, logging.INFO)


def _suppress_sensitive_transport_logs() -> None:
    """Prevent dependency loggers from rendering credential-bearing URLs."""
    for logger_name in _SENSITIVE_TRANSPORT_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a configured logger without installing duplicate handlers."""
    _suppress_sensitive_transport_logs()
    logger = logging.getLogger(name if name else __name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(_configured_level())
    logger.propagate = False
    _CONFIGURED_LOGGERS.add(logger)
    return logger


def refresh_log_levels() -> None:
    """Apply LOG_LEVEL again after a .env file has been loaded."""
    _suppress_sensitive_transport_logs()
    level = _configured_level()
    for logger in tuple(_CONFIGURED_LOGGERS):
        logger.setLevel(level)


def log_payload(
    logger: logging.Logger,
    message: str,
    payload: object,
) -> None:
    """Write an untruncated component-boundary payload at DEBUG level.

    Boundary payloads can contain prompts, retrieved filing text, generated
    answers, and public market data. They remain disabled unless
    ``LOG_LEVEL=DEBUG`` is configured. Callers must remove secrets, such as API
    keys, before passing a payload to this function.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return

    value = payload
    try:
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            value = dataclasses.asdict(value)
        elif hasattr(value, "model_dump"):
            value = value.model_dump()
        elif hasattr(value, "to_dict"):
            value = value.to_dict()
    except Exception:
        value = payload

    try:
        rendered = json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
    except (TypeError, ValueError):
        rendered = repr(value)
    logger.debug("%s\n%s", message, rendered)


__all__ = ["get_logger", "log_payload", "refresh_log_levels"]
