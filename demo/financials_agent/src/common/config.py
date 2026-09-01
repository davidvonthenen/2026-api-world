"""Configuration loading for the Financials Agent and local model servers."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from .logging import refresh_log_levels


@dataclass
class Settings:
    """Runtime configuration for A2A, MCP, RAG, and local model endpoints."""

    # OpenSearch and embeddings
    opensearch_host: str = "127.0.0.1"
    opensearch_port: int = 9200
    opensearch_index: str = "financial-filings-vector-chunks"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    rag_top_k: int = 3
    rag_num_candidates: int = 3

    # OpenAI-compatible Nemotron routing/verifier service
    orch_url: str = "http://127.0.0.1:8002/v1"
    orch_api_key: str = "not-needed"
    orch_model: str = "nvidia/Nemotron-Orchestrator-8B"
    orch_temperature: float = 0.0
    orch_top_p: float = 0.9
    orch_max_tokens: int = 131072
    orch_request_timeout: float = 600.0

    # OpenAI-compatible Qwen synthesis service
    llm_url: str = "http://127.0.0.1:8001/v1"
    llm_api_key: str = "not-needed"
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    llm_temperature: float = 0.2
    llm_top_p: float = 0.9
    llm_max_tokens: int = 131072
    llm_request_timeout: float = 600.0

    # Official MCP server/client
    financials_mcp_host: str = "0.0.0.0"
    financials_mcp_port: int = 8766
    financials_mcp_url: str = "http://127.0.0.1:8766/mcp"
    finnhub_api_key: str = ""
    finnhub_api_base: str = "https://finnhub.io/api/v1"
    mcp_call_timeout: float = 30.0

    # A2A service
    financials_agent_host: str = "0.0.0.0"
    financials_agent_port: int = 9002
    financials_agent_url: str = "http://127.0.0.1:9002"

    # Governance
    audit_log_path: str = "./var/financials-audit.jsonl"
    release_checks_enabled: bool = True
    policy_checks_enabled: bool = True
    evidence_checks_enabled: bool = True


def _get_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else value.strip()


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _load_dotenv(env_file: str | None) -> None:
    if env_file:
        load_dotenv(env_file)
        return

    candidates = (
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    )
    for candidate in candidates:
        if candidate.exists():
            load_dotenv(candidate)
            return


def load_settings(env_file: str | None = None) -> Settings:
    """Load environment-backed settings while retaining safe local defaults."""
    _load_dotenv(env_file)
    refresh_log_levels()

    host = _get_str("FINANCIALS_AGENT_HOST", Settings.financials_agent_host)
    port = _get_int("FINANCIALS_AGENT_PORT", Settings.financials_agent_port)
    agent_url = _get_str(
        "FINANCIALS_AGENT_URL",
        _get_str("APP_URL", f"http://127.0.0.1:{port}"),
    ).rstrip("/")


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

    # ORCH
    _orch_url = os.getenv("ORCH_URL", Settings.orch_url)
    if external_orch_ai:
        if use_openai:
            _orch_url = "https://api.openai.com/v1"
        else:
            _orch_url = _get_str("EXTERNAL_ORCH_URL", "https://api.openai.com/v1")

    _orch_api_key = os.getenv("ORCH_API_KEY", Settings.orch_api_key)
    if use_openai:
        _orch_api_key = _get_str("OPENAI_API_KEY", "")
    elif external_orch_ai:
        _orch_api_key = _get_str("EXTERNAL_ORCH_API_KEY", Settings.orch_api_key)

    _orch_model = os.getenv("ORCH_MODEL", Settings.orch_model)
    if use_openai:
        _orch_model = os.getenv("OPENAI_MODEL", "gpt-5.4")
    elif external_orch_ai:
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
    if  use_openai:
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

    return Settings(
        opensearch_host=_get_str("OPENSEARCH_HOST", Settings.opensearch_host),
        opensearch_port=_get_int("OPENSEARCH_PORT", Settings.opensearch_port),
        opensearch_index=_get_str("OPENSEARCH_INDEX", Settings.opensearch_index),
        embedding_model=_get_str("EMBEDDING_MODEL", Settings.embedding_model),
        rag_top_k=max(1, _get_int("RAG_TOP_K", Settings.rag_top_k)),
        rag_num_candidates=max(
            1,
            _get_int("RAG_NUM_CANDIDATES", Settings.rag_num_candidates),
        ),
        orch_url=_orch_url.rstrip("/"),
        orch_api_key=_orch_api_key,
        orch_model=_orch_model,
        orch_temperature=_get_float("ORCH_TEMPERATURE", Settings.orch_temperature),
        orch_top_p=_get_float("ORCH_TOP_P", Settings.orch_top_p),
        orch_max_tokens=_orch_max_tokens,
        orch_request_timeout=_get_float("ORCH_REQUEST_TIMEOUT", Settings.orch_request_timeout),
        llm_url=_llm_url.rstrip("/"),
        llm_api_key=_llm_api_key,
        llm_model=_llm_model,
        llm_temperature=_get_float("LLM_TEMPERATURE", Settings.llm_temperature),
        llm_top_p=_get_float("LLM_TOP_P", Settings.llm_top_p),
        llm_max_tokens=_llm_max_tokens,
        llm_request_timeout=_get_float("LLM_REQUEST_TIMEOUT", Settings.llm_request_timeout),
        financials_mcp_host=_get_str(
            "FINANCIALS_MCP_HOST",
            Settings.financials_mcp_host,
        ),
        financials_mcp_port=_get_int(
            "FINANCIALS_MCP_PORT",
            Settings.financials_mcp_port,
        ),
        financials_mcp_url=_get_str(
            "FINANCIALS_MCP_URL",
            Settings.financials_mcp_url,
        ).rstrip("/"),
        finnhub_api_key=_get_str("FINNHUB_API_KEY", Settings.finnhub_api_key),
        finnhub_api_base=_get_str(
            "FINNHUB_API_BASE",
            Settings.finnhub_api_base,
        ).rstrip("/"),
        financials_agent_host=host,
        financials_agent_port=port,
        financials_agent_url=agent_url,
        mcp_call_timeout=_get_float("MCP_CALL_TIMEOUT", Settings.mcp_call_timeout),
        audit_log_path=_get_str("AUDIT_LOG_PATH", Settings.audit_log_path),
        release_checks_enabled=_get_bool(
            "RELEASE_CHECKS_ENABLED",
            Settings.release_checks_enabled,
        ),
        policy_checks_enabled=_get_bool(
            "POLICY_CHECKS_ENABLED",
            Settings.policy_checks_enabled,
        ),
        evidence_checks_enabled=_get_bool(
            "EVIDENCE_CHECKS_ENABLED",
            Settings.evidence_checks_enabled,
        ),
    )


__all__ = ["Settings", "load_settings"]
