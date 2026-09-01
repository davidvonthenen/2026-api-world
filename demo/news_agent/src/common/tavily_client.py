"""Tavily SDK wrapper used by the technology-news MCP server."""
from __future__ import annotations

import re
from typing import Any, Dict, List
from urllib.parse import urlparse

from .config import Settings

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class TavilySearchClient:
    """Run the exact incoming question through the official Tavily Python SDK."""

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self.settings = settings
        if client is None:
            if not settings.tavily_api_key:
                raise RuntimeError("TAVILY_API_KEY must be set for real-time news search")
            from tavily import TavilyClient

            client = TavilyClient(api_key=settings.tavily_api_key)
        self.client = client

    def infer_time_range(self, question: str) -> str:
        lowered = question.lower()
        if any(term in lowered for term in ("today", "last 24 hours", "past 24 hours")):
            return "day"
        if any(term in lowered for term in ("this week", "last week", "past week")):
            return "week"
        if any(
            term in lowered
            for term in ("this year", "last year", "past year", "last 12 months")
        ):
            return "year"
        return self.settings.web_search_default_time_range

    @staticmethod
    def _safe_url(value: object) -> str:
        url = str(value or "").strip()
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return ""
        return url

    def _clean_content(self, value: object) -> str:
        content = _CONTROL_CHARS.sub(" ", str(value or "")).strip()
        if len(content) <= self.settings.web_result_max_chars:
            return content
        return content[: self.settings.web_result_max_chars].rsplit(" ", 1)[0] + "..."

    def search(self, question: str) -> List[Dict[str, Any]]:
        """Return normalized real-time news results for the exact question."""
        question = question.strip()
        if not question:
            raise ValueError("Search question must not be empty")

        response = self.client.search(
            query=question,
            topic="news",
            time_range=self.infer_time_range(question),
            search_depth="basic",
            include_answer=False,
            include_raw_content=False,
            include_images=False,
            max_results=self.settings.web_search_max_results,
            timeout=self.settings.web_search_timeout,
        )
        raw_results = response.get("results", []) if isinstance(response, dict) else []

        results: List[Dict[str, Any]] = []
        seen_urls: set[str] = set()
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            url = self._safe_url(item.get("url"))
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            results.append(
                {
                    "provider": "tavily",
                    "title": str(item.get("title") or "Untitled source").strip(),
                    "url": url,
                    "content": self._clean_content(item.get("content")),
                    "score": item.get("score"),
                    "published_date": item.get("published_date"),
                }
            )
        return results


__all__ = ["TavilySearchClient"]
