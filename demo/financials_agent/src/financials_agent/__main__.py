"""A2A 1.0 entry point for the governed Financials Agent."""
from __future__ import annotations

import os

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import uvicorn

from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from ..common.config import Settings, load_settings
from ..common.logging import get_logger
from .financials_executor import FinancialsExecutor


DEFAULT_LOG_LEVEL = "info"
LOGGER = get_logger(__name__)


def build_agent_card(settings: Settings | None = None) -> AgentCard:
    """Describe the agent through its sole A2A 1.0 JSON-RPC interface."""
    settings = settings or load_settings()
    skill = AgentSkill(
        id="financial_search",
        name="Financial filings and market analysis",
        description=(
            "Answers bounded questions about public-company SEC filing context, "
            "quotes, valuation metrics, and earnings with citations and an audit ID."
        ),
        input_modes=["text/plain"],
        output_modes=["text/plain"],
        tags=["financial", "stock", "filings", "earnings", "governance"],
        examples=[
            "What is NVDA's current stock price and what do recent filings say about revenue growth?",
            "Is Anthropic publicly traded?",
        ],
    )
    return AgentCard(
        name="Financial Agent",
        description=(
            "Financial-domain specialist for public-company filings, stock data, "
            "earnings, citations, verification, and auditability."
        ),
        version="2.1.0",
        default_input_modes=["text/plain", "text"],
        default_output_modes=["text/plain", "text"],
        capabilities=AgentCapabilities(streaming=True),
        supported_interfaces=[
            AgentInterface(
                protocol_binding="JSONRPC",
                protocol_version="1.0",
                url=settings.financials_agent_url.rstrip("/") + "/",
            ),
        ],
        skills=[skill],
    )


def _health_payload(settings: Settings, card: AgentCard) -> dict[str, Any]:
    """Return service metadata without invoking downstream dependencies."""
    return {
        "status": "ok",
        "agent": card.name,
        "agent_version": card.version,
        "a2a_protocols": ["1.0"],
        "historical_index": settings.opensearch_index,
        "financials_mcp_url": settings.financials_mcp_url,
        "router_verifier_model": settings.orch_model,
        "generator_model": settings.llm_model,
    }


def create_app(settings: Settings | None = None) -> Starlette:
    """Create one native Starlette application for health and A2A routes."""
    settings = settings or load_settings()
    card = build_agent_card(settings)
    handler = DefaultRequestHandler(
        agent_executor=FinancialsExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )

    async def health(_: Request) -> JSONResponse:
        return JSONResponse(_health_payload(settings, card))

    routes = [Route("/health", endpoint=health, methods=["GET"])]
    routes.extend(create_agent_card_routes(card))
    routes.extend(create_jsonrpc_routes(handler, rpc_url="/"))

    @asynccontextmanager
    async def lifespan(_: Starlette) -> AsyncIterator[None]:
        try:
            yield
        finally:
            # Drain active A2A task work before the process exits.
            await handler.aclose()

    return Starlette(routes=routes, lifespan=lifespan)


def main(
    host: str | None = None,
    port: int | None = None,
    log_level: str = DEFAULT_LOG_LEVEL,
) -> None:
    """Start the native Starlette ASGI service with Uvicorn."""
    settings = load_settings()

    if host is not None:
        settings.financials_agent_host = host
    if port is not None:
        settings.financials_agent_port = port

    explicit_public_url = (
        "FINANCIALS_AGENT_URL" in os.environ or "APP_URL" in os.environ
    )
    if (host is not None or port is not None) and not explicit_public_url:
        public_host = (
            "127.0.0.1"
            if settings.financials_agent_host in {"0.0.0.0", "::"}
            else settings.financials_agent_host
        )
        settings.financials_agent_url = (
            f"http://{public_host}:{settings.financials_agent_port}"
        )

    uvicorn.run(
        create_app(settings),
        host=settings.financials_agent_host,
        port=settings.financials_agent_port,
        log_level=log_level.lower(),
    )


if __name__ == "__main__":
    main()
