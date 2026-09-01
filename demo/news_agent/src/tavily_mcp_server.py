"""MCP 2.0 server exposing Tavily technology-news search over Streamable HTTP."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from mcp.server import MCPServer

from .common.config import Settings, load_settings
from .common.tavily_client import TavilySearchClient

mcp = MCPServer(
    "Tavily Technology News",
    instructions=(
        "Provides current technology-company news. The only tool accepts the exact "
        "user question and returns source snippets with URLs for downstream citation."
    ),
)


@lru_cache(maxsize=1)
def _settings() -> Settings:
    return load_settings()


@lru_cache(maxsize=1)
def _client() -> TavilySearchClient:
    return TavilySearchClient(_settings())


@mcp.tool()
async def search_technology_news(question: str) -> dict[str, Any]:
    """Search current technology news using the exact incoming question."""
    client = _client()
    results = await asyncio.to_thread(client.search, question)
    return {
        "provider": "tavily",
        "query": question,
        "topic": "news",
        "time_range": client.infer_time_range(question),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "result_count": len(results),
        "results": results,
    }


def main() -> None:
    settings = _settings()
    mcp.run(
        transport="streamable-http",
        host=settings.tavily_mcp_host,
        port=settings.tavily_mcp_port,
        streamable_http_path=settings.tavily_mcp_path,
        stateless_http=True,
        json_response=True,
    )


if __name__ == "__main__":
    main()
