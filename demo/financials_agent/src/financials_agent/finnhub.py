"""Typed Finnhub REST adapter used only by the MCP server."""
from __future__ import annotations

import datetime as dt
import re
import time
from typing import Any

import httpx
from pydantic import BaseModel, Field

from ..common.config import Settings
from ..common.logging import get_logger, log_payload

LOGGER = get_logger(__name__)


class FinnhubError(RuntimeError):
    pass


class SymbolCandidate(BaseModel):
    symbol: str
    display_symbol: str = ""
    description: str = ""
    security_type: str = ""


class SymbolResolution(BaseModel):
    query: str
    publicly_traded: bool
    symbol: str = ""
    company_name: str = ""
    exchange: str = ""
    currency: str = ""
    reason: str = ""
    candidates: list[SymbolCandidate] = Field(default_factory=list)
    retrieved_at: str


class CompanyProfile(BaseModel):
    symbol: str
    publicly_traded: bool
    company_name: str = ""
    exchange: str = ""
    industry: str = ""
    country: str = ""
    currency: str = ""
    market_cap_millions: float | None = None
    ipo_date: str = ""
    website: str = ""
    logo_url: str = ""
    retrieved_at: str


class StockQuote(BaseModel):
    symbol: str
    publicly_traded: bool
    current_price: float | None = None
    change: float | None = None
    percent_change: float | None = None
    day_high: float | None = None
    day_low: float | None = None
    open_price: float | None = None
    previous_close: float | None = None
    market_timestamp: str = ""
    retrieved_at: str


class CompanyMetrics(BaseModel):
    symbol: str
    metric_period: str = ""
    market_cap_millions: float | None = None
    pe_ttm: float | None = None
    price_to_sales_ttm: float | None = None
    price_to_book_quarterly: float | None = None
    return_on_equity_ttm: float | None = None
    net_profit_margin_ttm: float | None = None
    revenue_growth_ttm_yoy: float | None = None
    debt_to_equity_quarterly: float | None = None
    fifty_two_week_high: float | None = None
    fifty_two_week_low: float | None = None
    retrieved_at: str


class QuarterlyEarning(BaseModel):
    period: str
    actual_eps: float | None = None
    estimated_eps: float | None = None
    surprise: float | None = None
    surprise_percent: float | None = None


class QuarterlyEarningsHistory(BaseModel):
    symbol: str
    quarters: list[QuarterlyEarning]
    retrieved_at: str


class EarningsCalendarEntry(BaseModel):
    symbol: str
    date: str
    hour: str = ""
    eps_actual: float | None = None
    eps_estimate: float | None = None
    revenue_actual: float | None = None
    revenue_estimate: float | None = None


class EarningsCalendar(BaseModel):
    symbol: str
    from_date: str
    to_date: str
    entries: list[EarningsCalendarEntry]
    retrieved_at: str


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _float(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _market_time(value: Any) -> str:
    timestamp = _float(value)
    if not timestamp:
        return ""
    return dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc).isoformat()



def _require_symbol(value: str) -> str:
    symbol = value.upper().strip()
    if not symbol:
        raise FinnhubError("symbol is required")
    if not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,9}", symbol):
        raise FinnhubError(f"Invalid symbol format: {value!r}")
    return symbol


def _latest_metric_period(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    annual = payload.get("series", {}).get("annual", {})
    if not isinstance(annual, dict):
        return ""
    for values in annual.values():
        if not isinstance(values, list):
            continue
        periods = [
            str(item.get("period") or "")
            for item in values
            if isinstance(item, dict) and item.get("period")
        ]
        if periods:
            return max(periods)
    return ""


_COMPANY_SUFFIXES = {
    "ag",
    "co",
    "company",
    "corp",
    "corporation",
    "group",
    "holding",
    "holdings",
    "inc",
    "incorporated",
    "limited",
    "llc",
    "ltd",
    "nv",
    "plc",
    "sa",
}
_SHARE_CLASS_TOKENS = {
    "a",
    "b",
    "c",
    "class",
    "common",
    "ordinary",
    "share",
    "shares",
    "stock",
}


def _normalise_company(value: str) -> str:
    """Normalize an issuer name without collapsing it into a fuzzy near-match."""
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", value.lower())
    tokens = cleaned.split()
    while tokens and tokens[-1] in _SHARE_CLASS_TOKENS:
        tokens.pop()
    while tokens and tokens[-1] in _COMPANY_SUFFIXES:
        tokens.pop()
    while tokens and tokens[-1] in _SHARE_CLASS_TOKENS:
        tokens.pop()
    return " ".join(tokens)


class FinnhubService:
    """Minimal asynchronous adapter for the market data required by the demo."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = settings.finnhub_api_key
        self.api_base = settings.finnhub_api_base
        self._client = client

    async def _request(self, endpoint: str, params: dict[str, Any]) -> Any:
        if not self.api_key:
            raise FinnhubError("FINNHUB_API_KEY must be set before market-data tools can run")
        started = time.perf_counter()
        LOGGER.debug("Finnhub request started endpoint=%s", endpoint)
        log_payload(
            LOGGER,
            f"Finnhub request payload endpoint={endpoint}",
            {
                "method": "GET",
                "url": f"{self.api_base}{endpoint}",
                "params": params,
            },
        )
        own_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=15.0)
        try:
            response = await client.get(
                f"{self.api_base}{endpoint}",
                params={**params, "token": self.api_key},
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            LOGGER.warning(
                "Finnhub request failed endpoint=%s status=%s latency_ms=%.2f",
                endpoint,
                exc.response.status_code,
                (time.perf_counter() - started) * 1000,
            )
            log_payload(
                LOGGER,
                f"Finnhub error response endpoint={endpoint}",
                {
                    "status_code": exc.response.status_code,
                    "body": exc.response.text,
                },
            )
            raise FinnhubError(
                f"Finnhub request failed for {endpoint} with HTTP status "
                f"{exc.response.status_code}"
            ) from exc
        except httpx.RequestError as exc:
            LOGGER.warning(
                "Finnhub request failed endpoint=%s error=%s latency_ms=%.2f",
                endpoint,
                type(exc).__name__,
                (time.perf_counter() - started) * 1000,
            )
            raise FinnhubError(
                f"Finnhub request failed for {endpoint}: {type(exc).__name__}"
            ) from exc
        except ValueError as exc:
            LOGGER.warning(
                "Finnhub returned invalid JSON endpoint=%s latency_ms=%.2f",
                endpoint,
                (time.perf_counter() - started) * 1000,
            )
            raise FinnhubError(
                f"Finnhub returned invalid JSON for {endpoint}"
            ) from exc
        finally:
            if own_client:
                await client.aclose()

        if isinstance(payload, dict) and payload.get("error"):
            log_payload(
                LOGGER,
                f"Finnhub API error payload endpoint={endpoint}",
                payload,
            )
            raise FinnhubError(f"Finnhub error: {payload['error']}")
        LOGGER.info(
            "Finnhub request completed endpoint=%s status=%s latency_ms=%.2f",
            endpoint,
            response.status_code,
            (time.perf_counter() - started) * 1000,
        )
        log_payload(
            LOGGER,
            f"Finnhub response payload endpoint={endpoint}",
            payload,
        )
        return payload

    async def resolve_public_symbol(self, company_or_symbol: str) -> SymbolResolution:
        query = company_or_symbol.strip()
        if not query:
            raise FinnhubError("company_or_symbol is required")
        payload = await self._request("/search", {"q": query})
        raw_results = payload.get("result", []) if isinstance(payload, dict) else []
        candidates = [
            SymbolCandidate(
                symbol=str(item.get("symbol") or ""),
                display_symbol=str(item.get("displaySymbol") or ""),
                description=str(item.get("description") or ""),
                security_type=str(item.get("type") or ""),
            )
            for item in raw_results[:10]
            if isinstance(item, dict) and item.get("symbol")
        ]

        ticker_like = (
            bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9.-]{0,9}", query))
            and query.upper() == query
        )
        selected: SymbolCandidate | None = None
        selected_as_exact_ticker = False
        if ticker_like:
            selected = next(
                (item for item in candidates if item.symbol.upper() == query.upper()),
                None,
            )
            selected_as_exact_ticker = selected is not None

        # An all-uppercase company name such as NVIDIA resembles a ticker but may
        # not be one. A company-name request is accepted only on exact normalized
        # issuer identity. Near matches are not interchangeable securities.
        target = _normalise_company(query)
        if selected is None and target:
            for item in candidates:
                security = item.security_type.lower()
                if security and not any(
                    term in security for term in ("stock", "adr", "reit", "equity")
                ):
                    continue
                if _normalise_company(item.description) == target:
                    selected = item
                    break

        if selected is None:
            return SymbolResolution(
                query=query,
                publicly_traded=False,
                reason="No sufficiently close public-company symbol was returned by Finnhub.",
                candidates=candidates[:5],
                retrieved_at=_now(),
            )

        profile_payload = await self._request("/stock/profile2", {"symbol": selected.symbol})
        profile = profile_payload if isinstance(profile_payload, dict) else {}
        if not profile:
            return SymbolResolution(
                query=query,
                publicly_traded=False,
                reason="A symbol candidate existed, but Finnhub did not confirm a company profile.",
                candidates=candidates[:5],
                retrieved_at=_now(),
            )

        if not selected_as_exact_ticker:
            profile_name = _normalise_company(str(profile.get("name") or ""))
            if not profile_name or profile_name != target:
                return SymbolResolution(
                    query=query,
                    publicly_traded=False,
                    reason=(
                        "Finnhub returned a candidate symbol, but the confirmed company profile "
                        "did not exactly match the requested issuer."
                    ),
                    candidates=candidates[:5],
                    retrieved_at=_now(),
                )

        return SymbolResolution(
            query=query,
            publicly_traded=True,
            symbol=selected.symbol.upper(),
            company_name=str(profile.get("name") or selected.description),
            exchange=str(profile.get("exchange") or ""),
            currency=str(profile.get("currency") or ""),
            reason="Finnhub symbol search and company profile matched the request.",
            candidates=candidates[:5],
            retrieved_at=_now(),
        )

    async def get_company_profile(self, symbol: str) -> CompanyProfile:
        symbol = _require_symbol(symbol)
        payload = await self._request("/stock/profile2", {"symbol": symbol})
        profile = payload if isinstance(payload, dict) else {}
        return CompanyProfile(
            symbol=symbol,
            publicly_traded=bool(profile),
            company_name=str(profile.get("name") or ""),
            exchange=str(profile.get("exchange") or ""),
            industry=str(profile.get("finnhubIndustry") or ""),
            country=str(profile.get("country") or ""),
            currency=str(profile.get("currency") or ""),
            market_cap_millions=_float(profile.get("marketCapitalization")),
            ipo_date=str(profile.get("ipo") or ""),
            website=str(profile.get("weburl") or ""),
            logo_url=str(profile.get("logo") or ""),
            retrieved_at=_now(),
        )

    async def get_stock_quote(self, symbol: str) -> StockQuote:
        symbol = _require_symbol(symbol)
        payload = await self._request("/quote", {"symbol": symbol})
        quote = payload if isinstance(payload, dict) else {}
        values = [_float(quote.get(key)) for key in ("c", "pc", "h", "l", "o")]
        publicly_traded = any(value not in (None, 0.0) for value in values) or bool(_float(quote.get("t")))
        return StockQuote(
            symbol=symbol,
            publicly_traded=publicly_traded,
            current_price=_float(quote.get("c")),
            change=_float(quote.get("d")),
            percent_change=_float(quote.get("dp")),
            day_high=_float(quote.get("h")),
            day_low=_float(quote.get("l")),
            open_price=_float(quote.get("o")),
            previous_close=_float(quote.get("pc")),
            market_timestamp=_market_time(quote.get("t")),
            retrieved_at=_now(),
        )

    async def get_company_metrics(self, symbol: str) -> CompanyMetrics:
        symbol = _require_symbol(symbol)
        payload = await self._request("/stock/metric", {"symbol": symbol, "metric": "all"})
        metric = payload.get("metric", {}) if isinstance(payload, dict) else {}
        return CompanyMetrics(
            symbol=symbol,
            metric_period=_latest_metric_period(payload),
            market_cap_millions=_float(metric.get("marketCapitalization")),
            pe_ttm=_float(metric.get("peTTM")),
            price_to_sales_ttm=_float(metric.get("psTTM")),
            price_to_book_quarterly=_float(metric.get("pbQuarterly")),
            return_on_equity_ttm=_float(metric.get("roeTTM")),
            net_profit_margin_ttm=_float(metric.get("netProfitMarginTTM")),
            revenue_growth_ttm_yoy=_float(metric.get("revenueGrowthTTMYoy")),
            debt_to_equity_quarterly=_float(metric.get("totalDebt/totalEquityQuarterly")),
            fifty_two_week_high=_float(metric.get("52WeekHigh")),
            fifty_two_week_low=_float(metric.get("52WeekLow")),
            retrieved_at=_now(),
        )

    async def get_recent_quarterly_earnings(self, symbol: str, limit: int = 4) -> QuarterlyEarningsHistory:
        symbol = _require_symbol(symbol)
        limit = max(1, min(int(limit), 8))
        payload = await self._request("/stock/earnings", {"symbol": symbol, "limit": limit})
        if not isinstance(payload, list):
            raise FinnhubError("Unexpected Finnhub company-earnings payload")
        sorted_rows = sorted(payload, key=lambda row: str(row.get("period") or ""), reverse=True)
        quarters: list[QuarterlyEarning] = []
        for row in sorted_rows[:limit]:
            actual = _float(row.get("actual"))
            estimate = _float(row.get("estimate"))
            surprise = actual - estimate if actual is not None and estimate is not None else None
            surprise_percent = None
            if surprise is not None and estimate not in (None, 0.0):
                surprise_percent = surprise / abs(estimate) * 100
            quarters.append(
                QuarterlyEarning(
                    period=str(row.get("period") or ""),
                    actual_eps=actual,
                    estimated_eps=estimate,
                    surprise=surprise,
                    surprise_percent=surprise_percent,
                )
            )
        return QuarterlyEarningsHistory(symbol=symbol, quarters=quarters, retrieved_at=_now())

    async def get_earnings_calendar(
        self,
        symbol: str,
        from_date: str = "",
        to_date: str = "",
    ) -> EarningsCalendar:
        symbol = _require_symbol(symbol)
        try:
            start = dt.date.fromisoformat(from_date) if from_date else dt.date.today()
            end = dt.date.fromisoformat(to_date) if to_date else start + dt.timedelta(days=30)
        except ValueError as exc:
            raise FinnhubError("from_date and to_date must use YYYY-MM-DD") from exc
        if end < start:
            raise FinnhubError("to_date must be on or after from_date")
        payload = await self._request(
            "/calendar/earnings",
            {"symbol": symbol, "from": start.isoformat(), "to": end.isoformat()},
        )
        rows = payload.get("earningsCalendar", []) if isinstance(payload, dict) else []
        entries = [
            EarningsCalendarEntry(
                symbol=str(row.get("symbol") or symbol),
                date=str(row.get("date") or ""),
                hour=str(row.get("hour") or ""),
                eps_actual=_float(row.get("epsActual")),
                eps_estimate=_float(row.get("epsEstimate")),
                revenue_actual=_float(row.get("revenueActual")),
                revenue_estimate=_float(row.get("revenueEstimate")),
            )
            for row in rows
            if isinstance(row, dict)
        ]
        return EarningsCalendar(
            symbol=symbol,
            from_date=start.isoformat(),
            to_date=end.isoformat(),
            entries=entries,
            retrieved_at=_now(),
        )
