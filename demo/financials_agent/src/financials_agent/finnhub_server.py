"""Official MCP 2.0 server exposing bounded Finnhub market-data tools."""
from __future__ import annotations

from typing import Any

from .finnhub import (
    CompanyMetrics,
    CompanyProfile,
    EarningsCalendar,
    FinnhubService,
    QuarterlyEarningsHistory,
    StockQuote,
    SymbolResolution,
)
from mcp.server import MCPServer

from ..common.config import Settings, load_settings
from ..common.logging import get_logger, log_payload

LOGGER = get_logger(__name__)


def build_mcp_server(
    settings: Settings | None = None,
    service: FinnhubService | None = None,
) -> Any:
    """Build the MCP server without starting a transport, which keeps tests local."""

    resolved_settings = settings or load_settings()
    finnhub = service or FinnhubService(resolved_settings)
    mcp = MCPServer(
        "Financial Market Data",
        version="2.0.0",
        instructions=(
            "Use these tools only for public-company identity, current market data, "
            "company metrics, and earnings. Never infer a public ticker for a private "
            "company and never substitute a similarly named issuer."
        ),
    )

    async def run_tool(tool: str, arguments: dict[str, Any], call: Any) -> Any:
        LOGGER.info("MCP server tool started tool=%s", tool)
        log_payload(
            LOGGER,
            f"MCP server input payload tool={tool}",
            arguments,
        )
        try:
            result = await call
        except Exception:
            LOGGER.exception("MCP server tool failed tool=%s", tool)
            raise
        LOGGER.info("MCP server tool completed tool=%s", tool)
        log_payload(
            LOGGER,
            f"MCP server output payload tool={tool}",
            result,
        )
        return result

    @mcp.tool()
    async def resolve_public_symbol(company_or_symbol: str) -> SymbolResolution:
        """Resolve an exact public-company symbol or report that no safe match exists."""
        return await run_tool(
            "resolve_public_symbol",
            {"company_or_symbol": company_or_symbol},
            finnhub.resolve_public_symbol(company_or_symbol),
        )

    @mcp.tool()
    async def get_company_profile(symbol: str) -> CompanyProfile:
        """Return exchange, industry, currency, IPO date, and company identity data."""
        return await run_tool(
            "get_company_profile",
            {"symbol": symbol},
            finnhub.get_company_profile(symbol),
        )

    @mcp.tool()
    async def get_stock_quote(symbol: str) -> StockQuote:
        """Return the latest Finnhub quote fields and their market timestamp."""
        return await run_tool(
            "get_stock_quote",
            {"symbol": symbol},
            finnhub.get_stock_quote(symbol),
        )

    @mcp.tool()
    async def get_company_metrics(symbol: str) -> CompanyMetrics:
        """Return bounded valuation, profitability, growth, leverage, and range metrics."""
        return await run_tool(
            "get_company_metrics",
            {"symbol": symbol},
            finnhub.get_company_metrics(symbol),
        )

    @mcp.tool()
    async def get_recent_quarterly_earnings(
        symbol: str,
        limit: int = 4,
    ) -> QuarterlyEarningsHistory:
        """Return one to eight recent reported and estimated quarterly EPS records."""
        return await run_tool(
            "get_recent_quarterly_earnings",
            {"symbol": symbol, "limit": limit},
            finnhub.get_recent_quarterly_earnings(symbol, limit=limit),
        )

    @mcp.tool()
    async def get_earnings_calendar(
        symbol: str,
        from_date: str = "",
        to_date: str = "",
    ) -> EarningsCalendar:
        """Return earnings-calendar entries for an ISO-date range, defaulting to 30 days."""
        return await run_tool(
            "get_earnings_calendar",
            {
                "symbol": symbol,
                "from_date": from_date,
                "to_date": to_date,
            },
            finnhub.get_earnings_calendar(symbol, from_date, to_date),
        )

    return mcp


def main() -> None:
    """Run the official MCP 2.0 Streamable HTTP transport at /mcp."""
    settings = load_settings()
    LOGGER.info(
        "Starting Financial Market Data MCP server on http://%s:%d/mcp",
        settings.financials_mcp_host,
        settings.financials_mcp_port,
    )
    mcp = build_mcp_server(settings)
    mcp.run(
        transport="streamable-http",
        host=settings.financials_mcp_host,
        port=settings.financials_mcp_port,
        streamable_http_path="/mcp",
        json_response=True,
        stateless_http=True,
    )


if __name__ == "__main__":
    main()
