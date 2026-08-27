"""Historical technology-news retrieval from the CSV-backed OpenSearch corpus."""
from __future__ import annotations

from typing import Any

from ..common.config import Settings
from ..common.logging import get_logger
from .models import EvidenceItem
from .utils import clean_text, utc_now_iso

LOGGER = get_logger(__name__)


def _as_optional_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _as_optional_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _optional_text(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _evidence_from_hit(
    hit: dict[str, Any],
    settings: Settings,
    retrieved_at: str,
) -> EvidenceItem | None:
    """Convert one OpenSearch chunk hit into the governed evidence contract."""
    source = hit.get("_source")
    if not isinstance(source, dict):
        return None

    text = clean_text(source.get("text", ""), settings.evidence_snippet_max_chars)
    if not text:
        return None

    chunk_index = _as_optional_int(source.get("chunk_index"))
    article_source_id = str(source.get("source_id") or "").strip()
    chunk_source_id = str(hit.get("_id") or "").strip()
    if not chunk_source_id and article_source_id:
        chunk_source_id = (
            f"{article_source_id}:{chunk_index}"
            if chunk_index is not None
            else article_source_id
        )
    if not chunk_source_id:
        return None

    path = str(source.get("source_path") or source.get("path") or "").strip()
    tags = source.get("tags")
    if not isinstance(tags, list):
        tags = [] if tags in (None, "") else [str(tags)]

    return EvidenceItem(
        source_kind="historical",
        source_id=chunk_source_id,
        title=str(source.get("title") or path or "Historical article"),
        text=text,
        score=_as_optional_float(hit.get("_score")),
        path=path or None,
        url=_optional_text(source.get("url")),
        category=_optional_text(source.get("category")),
        chunk_index=chunk_index,
        published_at=_optional_text(source.get("published_at")),
        retrieved_at=retrieved_at,
        metadata={
            "index": settings.opensearch_index,
            "dataset_id": source.get("dataset_id"),
            "record_id": source.get("record_id"),
            "article_source_id": article_source_id or None,
            "domain": source.get("domain"),
            "tags": tags,
            "ingested_at": source.get("ingested_at"),
            "content_sha256": source.get("content_sha256"),
            "source_content_sha256": source.get("source_content_sha256"),
        },
    )


def _select_diverse_evidence(
    candidates: list[EvidenceItem],
    *,
    limit: int,
    max_chunks_per_source: int,
) -> list[EvidenceItem]:
    """Prefer article diversity, then fill remaining slots by vector rank."""
    if limit <= 0:
        return []
    max_chunks_per_source = max(1, max_chunks_per_source)

    selected: list[EvidenceItem] = []
    deferred: list[EvidenceItem] = []
    per_article: dict[str, int] = {}

    for item in candidates:
        article_id = str(item.metadata.get("article_source_id") or item.source_id)
        count = per_article.get(article_id, 0)
        if count < max_chunks_per_source and len(selected) < limit:
            selected.append(item)
            per_article[article_id] = count + 1
        else:
            deferred.append(item)

    if len(selected) < limit:
        selected_ids = {item.source_id for item in selected}
        for item in deferred:
            if item.source_id in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(item.source_id)
            if len(selected) >= limit:
                break

    return selected[:limit]


class HistoricalNewsRetriever:
    """Lazy-loading synchronous retriever suitable for ``asyncio.to_thread``."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._embedder: Any | None = None
        self._client: Any | None = None

    def _ensure_runtime(self) -> tuple[Any, Any]:
        if self._embedder is None or self._client is None:
            # Keep heavy ML and OpenSearch imports out of A2A process startup tests.
            from ..common.embeddings import EmbeddingModel
            from ..common.opensearch_client import create_client, ensure_index

            self._embedder = EmbeddingModel(self.settings)
            self._client = create_client(self.settings)
            ensure_index(
                self.settings,
                self._embedder.dimension,
                client=self._client,
            )
        return self._embedder, self._client

    def search(self, question: str) -> list[EvidenceItem]:
        """Embed the exact question and retrieve ranked historical article chunks."""
        from ..common.embeddings import to_list
        from ..common.opensearch_client import knn_search

        embedder, client = self._ensure_runtime()
        embedding = embedder.encode([question])[0]

        # Fetch more chunks than the final evidence count so overlapping chunks from
        # one long CSV article do not crowd every other source out of the answer.
        fetch_k = max(
            self.settings.rag_top_k,
            min(
                self.settings.rag_num_candidates,
                self.settings.rag_top_k * 4,
            ),
        )
        response = knn_search(
            client,
            self.settings.opensearch_index,
            to_list(embedding),
            k=fetch_k,
            num_candidates=self.settings.rag_num_candidates,
        )

        retrieved_at = utc_now_iso()
        candidates: list[EvidenceItem] = []
        seen_chunk_ids: set[str] = set()
        for hit in response.get("hits", {}).get("hits", []):
            if not isinstance(hit, dict):
                continue
            item = _evidence_from_hit(hit, self.settings, retrieved_at)
            if item is None or item.source_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(item.source_id)
            candidates.append(item)

        evidence = _select_diverse_evidence(
            candidates,
            limit=self.settings.rag_top_k,
            max_chunks_per_source=self.settings.rag_max_chunks_per_source,
        )
        LOGGER.info(
            "event=historical_retrieval index=%s raw_hits=%d evidence_count=%d",
            self.settings.opensearch_index,
            len(candidates),
            len(evidence),
        )
        return evidence


__all__ = ["HistoricalNewsRetriever"]
