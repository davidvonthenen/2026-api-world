"""Run the governed News Agent as an A2A protocol 1.0 JSON-RPC server."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

import click
import uvicorn
from starlette.applications import Starlette
from starlette.routing import Mount

from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from ..common.config import Settings, load_settings
from .agent_executor import NewsAgentExecutor
from .news_agent import NewsAgent


DEFAULT_LOG_LEVEL = "info"


def get_agent_card(settings: Settings) -> AgentCard:
    """Build the public card for the single supported A2A interface."""
    settings = settings or load_settings()
    skill = AgentSkill(
        id="technology_news_analysis",
        name="Technology company news analysis",
        description=(
            "Combines a bounded historical technology-news corpus with current Tavily "
            "news, then verifies citations and domain scope before release."
        ),
        input_modes=["text/plain"],
        output_modes=["text/plain"],
        tags=["news", "technology", "AI", "RAG", "governance", "citations"],
        examples=[
            "What does NVIDIA have going on in AI, and how did its strategy evolve?",
            "Summarize recent AI announcements from Microsoft with historical context.",
        ],
    )
    return AgentCard(
        name="News Agent",
        description=(
            "Authority for technology-company news, history, and current events. "
            "It does not provide stock prices, SEC filing analysis, earnings metrics, "
            "valuation, or investment recommendations."
        ),
        version="2.1.0",
        default_input_modes=["text/plain", "text"],
        default_output_modes=["text/plain", "text"],
        capabilities=AgentCapabilities(streaming=True),
        supported_interfaces=[
            AgentInterface(
                protocol_binding="JSONRPC",
                protocol_version="1.0",
                url=settings.a2a_app_url.rstrip("/") + "/",
            ),
        ],
        skills=[skill],
    )


def create_app(settings: Settings | None = None) -> Starlette:
    """Create the Starlette application and its A2A protocol routes."""
    settings = settings or load_settings()
    card = get_agent_card(settings)
    handler = DefaultRequestHandler(
        agent_executor=NewsAgentExecutor(agent=NewsAgent(settings=settings)),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )

    async def health(_: Request) -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "agent": card.name,
                "agent_version": card.version,
                "a2a_protocols": ["1.0"],
                "historical_index": settings.opensearch_index,
                "tavily_mcp_url": settings.tavily_mcp_url,
                "router_verifier_model": settings.orch_model,
                "generator_model": settings.llm_model,
            }
        )

    routes = [Route("/health", endpoint=health, methods=["GET"])]
    routes.extend(create_agent_card_routes(card))
    routes.extend(create_jsonrpc_routes(handler, rpc_url="/"))

    @asynccontextmanager
    async def lifespan(_: Starlette):
        try:
            yield
        finally:
            # Drain active task work before the process exits.
            await handler.aclose()

    return Starlette(routes=routes, lifespan=lifespan)


def main(
    host: str | None = None,
    port: int | None = None,
    log_level: str = DEFAULT_LOG_LEVEL,
) -> None:
    settings = load_settings()
    if host is not None:
        settings.a2a_host = host
    if port is not None:
        settings.a2a_port = port
        if "APP_URL" not in os.environ:
            public_host = (
                "127.0.0.1"
                if settings.a2a_host == "0.0.0.0"
                else settings.a2a_host
            )
            settings.a2a_app_url = f"http://{public_host}:{port}"

    uvicorn.run(
        create_app(settings),
        host=settings.a2a_host,
        port=settings.a2a_port,
        log_level=log_level.lower(),
    )


@click.command()
@click.option("--host", default=None)
@click.option("--port", default=None, type=int)
@click.option("--log-level", default=DEFAULT_LOG_LEVEL)
def cli(host: str | None, port: int | None, log_level: str) -> None:
    main(host, port, log_level)


if __name__ == "__main__":
    cli()
