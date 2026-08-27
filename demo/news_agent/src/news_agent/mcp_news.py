"""MCP client adapter for current Tavily news evidence."""
from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

from ..common.config import Settings
from ..common.logging import get_logger
from .models import EvidenceItem
from .utils import clean_text, utc_now_iso

LOGGER = get_logger(__name__)


class TavilyNewsMCPClient:
    """Call only the bounded ``search_technology_news`` MCP tool."""

    TOOL_NAME = "search_technology_news"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @staticmethod
    def _structured_payload(result: Any) -> dict[str, Any]:
        payload = getattr(result, "structured_content", None)
        if isinstance(payload, dict):
            wrapped = payload.get("result")
            if isinstance(wrapped, dict):
                return wrapped
            return payload

        for block in getattr(result, "content", []) or []:
            text = getattr(block, "text", None)
            if not isinstance(text, str):
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        raise RuntimeError("MCP tool returned no structured news payload")

    async def search(self, question: str) -> list[EvidenceItem]:
        """Send the exact user question to the remote Streamable HTTP MCP server."""
        from mcp import Client

        async def call_tool() -> Any:
            async with Client(self.settings.tavily_mcp_url) as client:
                return await client.call_tool(self.TOOL_NAME, {"question": question})

        result = await asyncio.wait_for(
            call_tool(),
            timeout=self.settings.mcp_call_timeout,
        )

        if getattr(result, "is_error", False):
            raise RuntimeError("Tavily MCP tool reported an execution error")

        payload = self._structured_payload(result)
        retrieved_at = str(payload.get("retrieved_at") or utc_now_iso())
        evidence: list[EvidenceItem] = []
        seen_urls: set[str] = set()

        for item in payload.get("results", []) or []:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            text = clean_text(item.get("content", ""), self.settings.web_result_max_chars)
            if not url or not text or url in seen_urls:
                continue
            seen_urls.add(url)
            score_value = item.get("score")
            try:
                score = float(score_value) if score_value is not None else None
            except (TypeError, ValueError):
                score = None
            evidence.append(
                EvidenceItem(
                    source_kind="realtime",
                    source_id=hashlib.sha256(url.encode("utf-8")).hexdigest(),
                    title=str(item.get("title") or "Current news source"),
                    text=text,
                    score=score,
                    url=url,
                    published_at=(
                        str(item.get("published_date"))
                        if item.get("published_date")
                        else None
                    ),
                    retrieved_at=retrieved_at,
                    metadata={
                        "provider": payload.get("provider", "tavily"),
                        "topic": payload.get("topic", "news"),
                        "time_range": payload.get("time_range"),
                        "tool": self.TOOL_NAME,
                    },
                )
            )

        LOGGER.info(
            "event=mcp_news_retrieval tool=%s evidence_count=%d",
            self.TOOL_NAME,
            len(evidence),
        )
        return evidence


__all__ = ["TavilyNewsMCPClient"]
