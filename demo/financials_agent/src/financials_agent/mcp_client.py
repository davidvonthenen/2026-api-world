"""MCP 2.0 client for the Financial Market Data streamable-HTTP server."""
from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from typing import Any

from mcp import Client

from ..common.config import Settings
from ..common.logging import get_logger, log_payload
from .models import ToolCallTrace

LOGGER = get_logger(__name__)


class FinancialMCPClient:
    """Call only allowlisted market-data tools and consume structured outputs."""

    def __init__(self, settings: Settings):
        self.url = settings.financials_mcp_url
        self.timeout_seconds = settings.mcp_call_timeout

    @staticmethod
    def _error_text(result: Any) -> str:
        parts: list[str] = []
        for block in getattr(result, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                parts.append(str(text))
        return "\n".join(parts) or "MCP tool returned an unspecified error"

    @staticmethod
    def _structured_data(result: Any) -> dict[str, Any] | None:
        """Read structured MCP output across compatible result representations."""
        structured = getattr(result, "structured_content", None)
        if structured is None:
            structured = getattr(result, "structuredContent", None)

        if structured is None:
            model_dump = getattr(result, "model_dump", None)
            if callable(model_dump):
                for by_alias in (False, True):
                    try:
                        payload = model_dump(by_alias=by_alias)
                    except TypeError:
                        payload = model_dump()
                    if not isinstance(payload, dict):
                        continue
                    structured = payload.get("structured_content")
                    if structured is None:
                        structured = payload.get("structuredContent")
                    if structured is not None:
                        break

        if structured is None:
            for block in getattr(result, "content", []) or []:
                text = getattr(block, "text", None)
                if not isinstance(text, str) or not text.strip():
                    continue
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    structured = parsed
                    break

        if structured is None:
            return None
        if isinstance(structured, dict):
            return structured
        return {"result": structured}

    async def call_tools(
        self,
        calls: Sequence[tuple[str, dict[str, Any]]],
    ) -> list[ToolCallTrace]:
        if not calls:
            return []

        LOGGER.info("MCP request batch started url=%s tools=%s", self.url, [tool for tool, _ in calls])
        log_payload(
            LOGGER,
            "MCP client request batch payload",
            [{"tool": tool, "arguments": arguments} for tool, arguments in calls],
        )
        try:
            async with Client(self.url) as client:
                traces: list[ToolCallTrace] = []
                for tool, arguments in calls:
                    started = time.perf_counter()
                    LOGGER.debug("MCP tool request started tool=%s", tool)
                    log_payload(
                        LOGGER,
                        f"MCP client request payload tool={tool}",
                        arguments,
                    )
                    try:
                        result = await asyncio.wait_for(
                            client.call_tool(tool, arguments),
                            timeout=self.timeout_seconds,
                        )
                        is_error = bool(getattr(result, "is_error", False))
                        structured = self._structured_data(result)
                        if is_error:
                            trace = ToolCallTrace(
                                tool=tool,
                                arguments=arguments,
                                ok=False,
                                error=self._error_text(result),
                                latency_ms=(time.perf_counter() - started) * 1000,
                            )
                            traces.append(trace)
                            LOGGER.warning(
                                "MCP tool request completed tool=%s ok=false latency_ms=%.2f error=%s",
                                tool,
                                trace.latency_ms,
                                trace.error,
                            )
                            log_payload(
                                LOGGER,
                                f"MCP client response payload tool={tool}",
                                trace.to_dict(),
                            )
                            continue

                        if structured is None:
                            trace = ToolCallTrace(
                                tool=tool,
                                arguments=arguments,
                                ok=False,
                                error="MCP tool returned no structured content",
                                latency_ms=(time.perf_counter() - started) * 1000,
                            )
                            traces.append(trace)
                            LOGGER.warning(
                                "MCP tool request completed tool=%s ok=false latency_ms=%.2f error=%s",
                                tool,
                                trace.latency_ms,
                                trace.error,
                            )
                            log_payload(
                                LOGGER,
                                f"MCP client response payload tool={tool}",
                                trace.to_dict(),
                            )
                            continue

                        trace = ToolCallTrace(
                            tool=tool,
                            arguments=arguments,
                            ok=True,
                            data=structured,
                            latency_ms=(time.perf_counter() - started) * 1000,
                        )
                        traces.append(trace)
                        LOGGER.info(
                            "MCP tool request completed tool=%s ok=true latency_ms=%.2f",
                            tool,
                            trace.latency_ms,
                        )
                        log_payload(
                            LOGGER,
                            f"MCP client response payload tool={tool}",
                            trace.to_dict(),
                        )
                    except Exception as exc:
                        trace = ToolCallTrace(
                            tool=tool,
                            arguments=arguments,
                            ok=False,
                            error=str(exc),
                            latency_ms=(time.perf_counter() - started) * 1000,
                        )
                        traces.append(trace)
                        LOGGER.exception(
                            "MCP tool request failed tool=%s latency_ms=%.2f",
                            tool,
                            trace.latency_ms,
                        )
                        log_payload(
                            LOGGER,
                            f"MCP client error payload tool={tool}",
                            trace.to_dict(),
                        )
                log_payload(
                    LOGGER,
                    "MCP client response batch payload",
                    [trace.to_dict() for trace in traces],
                )
                return traces
        except Exception as exc:
            LOGGER.exception("MCP connection failed url=%s", self.url)
            traces = [
                ToolCallTrace(
                    tool=tool,
                    arguments=arguments,
                    ok=False,
                    error=f"MCP connection failed: {exc}",
                )
                for tool, arguments in calls
            ]
            log_payload(
                LOGGER,
                "MCP client connection error payload",
                [trace.to_dict() for trace in traces],
            )
            return traces
