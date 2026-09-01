"""Configuration for the API World Orchestrator Agent demo."""

from __future__ import annotations

import os

from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Runtime settings with intentionally narrow configuration surface."""

    api_host: str = "0.0.0.0"
    api_port: int = 10000
    public_model_name: str = "vertical-api-orchestrator"

    orch_url: str = "http://127.0.0.1:8002/v1"
    orch_api_key: str = "not-needed"
    orch_model: str = "nvidia/Nemotron-Orchestrator-8B"
    orch_max_tokens: int = 131072
    orch_request_timeout: float = 600.0

    llm_url: str = "http://127.0.0.1:8001/v1"
    llm_api_key: str = "not-needed"
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    llm_temperature: float = 0.2
    llm_top_p: float = 0.9
    llm_max_tokens: int = 131072
    llm_request_timeout: float = 600.0

    news_agent_url: str = "http://127.0.0.1:9001"
    financial_agent_url: str = "http://127.0.0.1:9002"
    a2a_timeout_seconds: float = 1200.0

    audit_log_path: str = "./logs/orchestrator-audit.jsonl"
    audit_include_query: bool = False
    max_query_chars: int = 32768
    release_checks_enabled: bool = True
    policy_checks_enabled: bool = True
    evidence_checks_enabled: bool = True

    # These are contract limits rather than tuning knobs.
    max_orchestration_rounds: int = 3
    max_specialist_calls: int = 2
    max_clarifications: int = 1
    planner_retries: int = 1


def _load_dotenv(env_file: str | None) -> None:
    if env_file:
        load_dotenv(env_file, override=False)
        return

    project_root = Path(__file__).resolve().parents[2]
    for candidate in (Path.cwd() / ".env", project_root / ".env"):
        if candidate.exists():
            load_dotenv(candidate, override=False)
            return


def _get_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip()


def _get_int(name: str, default_val: int) -> int:
    """Read an integer environment variable, falling back on invalid input."""
    value = os.getenv(name)
    if value is None or value == "":
        return default_val
    try:
        return int(value)
    except ValueError:
        return default_val


def _get_float(name: str, default_val: float) -> float:
    """Read a floating-point environment variable, falling back on invalid input."""
    value = os.getenv(name)
    if value is None or value == "":
        return default_val
    try:
        return float(value)
    except ValueError:
        return default_val


def _get_bool(name: str, default_val: bool) -> bool:
    """Read an explicit boolean value, falling back on invalid input."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default_val
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default_val


def load_settings(env_file: str | None = None) -> Settings:
    """Load supported environment values without adding CLI flags."""

    _load_dotenv(env_file)

    external_ai = os.getenv("USE_EXTERNAL_AI", "false").lower() in ("1", "true", "yes", "on")
    external_orch_ai = os.getenv("USE_EXTERNAL_ORCH_AI", "false").lower() in ("1", "true", "yes", "on")
    external_llm_ai = os.getenv("USE_EXTERNAL_LLM_AI", "false").lower() in ("1", "true", "yes", "on")
    if external_ai:
        external_orch_ai = True
        external_llm_ai = True
    use_openai = os.getenv("USE_EXTERNAL_OPENAI", "false").lower() in ("1", "true", "yes", "on")
    if use_openai:
        external_ai = True
        external_orch_ai = True
        external_llm_ai = True

    _orch_url = os.getenv("ORCH_URL", Settings.orch_url)
    if external_ai or external_orch_ai:
        if use_openai:
            _orch_url = "https://api.openai.com/v1"
        else:
            _orch_url = _get_str("EXTERNAL_ORCH_URL", "https://api.openai.com/v1")

    _orch_api_key = os.getenv("ORCH_API_KEY", Settings.orch_api_key)
    if use_openai:
        _orch_api_key = _get_str("OPENAI_API_KEY", "")
    elif external_ai or external_orch_ai:
        _orch_api_key = _get_str("EXTERNAL_ORCH_API_KEY", Settings.orch_api_key)

    _orch_model = os.getenv("ORCH_MODEL", Settings.orch_model)
    if use_openai:
        _orch_model = os.getenv("OPENAI_MODEL", "gpt-5.4")
    elif external_ai or external_orch_ai:
        _orch_model = os.getenv("EXTERNAL_ORCH_MODEL", "")

    _orch_max_tokens = _get_int("ORCH_MAX_TOKENS", Settings.orch_max_tokens)
    if external_orch_ai or use_openai:
        _orch_max_tokens = _get_int("EXTERNAL_ORCH_MAX_TOKENS", 131072)

    # LLM
    _llm_url = os.getenv("LLM_URL", Settings.llm_url)
    if external_llm_ai:
        if use_openai:
            _llm_url = "https://api.openai.com/v1"
        else:
            _llm_url = _get_str("EXTERNAL_LLM_URL", "https://api.openai.com/v1")

    _llm_api_key = os.getenv("LLM_API_KEY", Settings.llm_api_key)
    if use_openai:
        _llm_api_key = _get_str("OPENAI_API_KEY", "")
    elif external_llm_ai:
        _llm_api_key = _get_str("EXTERNAL_LLM_API_KEY", Settings.llm_api_key)

    _llm_model = os.getenv("LLM_MODEL", Settings.llm_model)
    if use_openai:
        _llm_model = os.getenv("OPENAI_MODEL", "gpt-5.4")
    elif external_llm_ai:
        _llm_model = os.getenv("EXTERNAL_LLM_MODEL", "")

    _llm_max_tokens = _get_int("LLM_MAX_TOKENS", Settings.llm_max_tokens)
    if external_llm_ai or use_openai:
        _llm_max_tokens = _get_int("EXTERNAL_LLM_MAX_TOKENS", 131072)

    defaults = Settings()
    return Settings(
        api_host=_get_str("ORCHESTRATOR_HOST", defaults.api_host),
        api_port=_get_int("ORCHESTRATOR_PORT", defaults.api_port),
        public_model_name=_get_str("ORCHESTRATOR_PUBLIC_MODEL", defaults.public_model_name),
        orch_url=_orch_url.rstrip("/"),
        orch_api_key=_orch_api_key,
        orch_model=_orch_model,
        orch_max_tokens=_orch_max_tokens,
        orch_request_timeout=_get_float("ORCH_REQUEST_TIMEOUT", Settings.orch_request_timeout),
        llm_url=_llm_url.rstrip("/"),
        llm_api_key=_llm_api_key,
        llm_model=_llm_model,
        llm_temperature=_get_float("LLM_TEMPERATURE", Settings.llm_temperature),
        llm_top_p=_get_float("LLM_TOP_P", Settings.llm_top_p),
        llm_max_tokens=_llm_max_tokens,
        llm_request_timeout=_get_float("LLM_REQUEST_TIMEOUT", Settings.llm_request_timeout),
        news_agent_url=_get_str("NEWS_AGENT_URL", defaults.news_agent_url).rstrip("/"),
        financial_agent_url=_get_str("FINANCIAL_AGENT_URL", defaults.financial_agent_url).rstrip("/"),
        a2a_timeout_seconds=_get_float("A2A_TIMEOUT_SECONDS", defaults.a2a_timeout_seconds),
        audit_log_path=_get_str("AUDIT_LOG_PATH", defaults.audit_log_path),
        audit_include_query=_get_bool("AUDIT_INCLUDE_QUERY", defaults.audit_include_query),
        max_query_chars=_get_int("MAX_QUERY_CHARS", defaults.max_query_chars),
        release_checks_enabled=_get_bool(
            "RELEASE_CHECKS_ENABLED",
            defaults.release_checks_enabled,
        ),
        policy_checks_enabled=_get_bool(
            "POLICY_CHECKS_ENABLED",
            defaults.policy_checks_enabled,
        ),
        evidence_checks_enabled=_get_bool(
            "EVIDENCE_CHECKS_ENABLED",
            defaults.evidence_checks_enabled,
        ),
        # Preserve the implementation contract regardless of environment values.
        max_orchestration_rounds=defaults.max_orchestration_rounds,
        max_specialist_calls=defaults.max_specialist_calls,
        max_clarifications=defaults.max_clarifications,
        planner_retries=defaults.planner_retries,
    )


__all__ = ["Settings", "load_settings"]
