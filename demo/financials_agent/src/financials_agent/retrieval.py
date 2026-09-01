"""Filtered vector retrieval over the financial-filings OpenSearch index."""
from __future__ import annotations

import asyncio
import time
from typing import Any

from ..common.config import Settings
from ..common.embeddings import EmbeddingModel
from ..common.logging import get_logger, log_payload
from ..common.opensearch_client import create_client, ensure_index
from ..common.embeddings import to_list
from ..common.opensearch_client import knn_search
from .models import Evidence, RetrievalTrace

LOGGER = get_logger(__name__)


def _trim(text: str, limit: int = 1800) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    candidate = text[:limit]
    boundary = candidate.rfind(" ")
    return (candidate[:boundary] if boundary > 0 else candidate) + "..."


class FinancialFilingsRAG:
    """Retrieve only filing chunks belonging to the resolved public symbol."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client: Any | None = None
        self._embedder: Any | None = None

    def _dependencies(self) -> tuple[Any, Any]:
        if self._client is None or self._embedder is None:

            self._client = create_client(self.settings)
            self._embedder = EmbeddingModel(self.settings)
            ensure_index(self.settings, self._embedder.dimension)
        return self._client, self._embedder

    def _retrieve_sync(self, question: str, symbol: str) -> tuple[list[Evidence], RetrievalTrace]:
        started = time.perf_counter()
        LOGGER.info(
            "Filing retrieval started index=%s symbol=%s",
            self.settings.opensearch_index,
            symbol or "<missing>",
        )
        log_payload(
            LOGGER,
            "Filing retrieval input payload",
            {
                "question": question,
                "symbol": symbol,
                "index": self.settings.opensearch_index,
                "top_k": self.settings.rag_top_k,
                "num_candidates": self.settings.rag_num_candidates,
            },
        )
        if not symbol:
            trace = RetrievalTrace(
                index=self.settings.opensearch_index,
                symbol_filter="",
                hit_count=0,
                latency_ms=0.0,
                error="A resolved ticker is required before filing retrieval.",
            )
            LOGGER.warning("Filing retrieval skipped: %s", trace.error)
            log_payload(LOGGER, "Filing retrieval output payload", trace.to_dict())
            return [], trace

        try:
            client, embedder = self._dependencies()
            query_vector = to_list(embedder.encode([question])[0])
            response = knn_search(
                client,
                self.settings.opensearch_index,
                query_vector,
                k=self.settings.rag_top_k,
                num_candidates=self.settings.rag_num_candidates,
                filters=[{"term": {"symbol": symbol.upper()}}],
            )
            raw_hits = response.get("hits", {}).get("hits", [])
            evidence: list[Evidence] = []
            trace_hits: list[dict[str, Any]] = []
            for index, hit in enumerate(raw_hits, start=1):
                source = hit.get("_source", {})
                score = float(hit.get("_score", 0.0) or 0.0)
                evidence.append(
                    Evidence(
                        evidence_id=f"F{index}",
                        kind="financial_filing",
                        source="Financial filings OpenSearch RAG",
                        title=str(source.get("title") or source.get("path") or "SEC filing"),
                        content=_trim(str(source.get("text") or "")),
                        as_of=str(source.get("filing_date") or source.get("fiscal_period") or ""),
                        source_url=str(source.get("source_url") or ""),
                        metadata={
                            "company": source.get("company", ""),
                            "symbol": source.get("symbol", ""),
                            "filing_type": source.get("filing_type", ""),
                            "fiscal_period": source.get("fiscal_period", ""),
                            "accession_number": source.get("accession_number", ""),
                            "path": source.get("path", ""),
                            "chunk_index": source.get("chunk_index", 0),
                            "score": round(score, 6),
                        },
                    )
                )
                trace_hits.append(
                    {
                        "evidence_id": f"F{index}",
                        "path": source.get("path", ""),
                        "chunk_index": source.get("chunk_index", 0),
                        "score": score,
                        "content_sha256": source.get("content_sha256", ""),
                    }
                )

            trace = RetrievalTrace(
                index=self.settings.opensearch_index,
                symbol_filter=symbol.upper(),
                hit_count=len(evidence),
                latency_ms=(time.perf_counter() - started) * 1000,
                hits=trace_hits,
            )
            LOGGER.info(
                "Filing retrieval completed index=%s symbol=%s hits=%d latency_ms=%.2f",
                trace.index,
                trace.symbol_filter,
                trace.hit_count,
                trace.latency_ms,
            )
            log_payload(
                LOGGER,
                "Filing retrieval output payload",
                {
                    "trace": trace.to_dict(),
                    "evidence": [item.to_dict() for item in evidence],
                },
            )
            return evidence, trace
        except Exception as exc:
            trace = RetrievalTrace(
                index=self.settings.opensearch_index,
                symbol_filter=symbol.upper(),
                hit_count=0,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=str(exc),
            )
            LOGGER.exception(
                "Filing retrieval failed index=%s symbol=%s latency_ms=%.2f",
                trace.index,
                trace.symbol_filter,
                trace.latency_ms,
            )
            log_payload(LOGGER, "Filing retrieval error payload", trace.to_dict())
            return [], trace

    async def retrieve(self, question: str, symbol: str) -> tuple[list[Evidence], RetrievalTrace]:
        return await asyncio.to_thread(self._retrieve_sync, question, symbol)
