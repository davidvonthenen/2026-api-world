"""OpenSearch client utilities for filtered financial-filings retrieval."""
from __future__ import annotations

from typing import Any, List, Optional, Sequence

from .config import Settings
from .logging import get_logger, log_payload

LOGGER = get_logger(__name__)


def create_client(settings: Settings):
    """Create the unsecured local OpenSearch client used by the demo."""
    from opensearchpy import OpenSearch

    LOGGER.info(
        "Connecting to OpenSearch at %s:%s",
        settings.opensearch_host,
        settings.opensearch_port,
    )
    return OpenSearch(
        hosts=[
            {
                "host": settings.opensearch_host,
                "port": settings.opensearch_port,
                "scheme": "http",
            }
        ],
        http_compress=True,
        http_auth=None,
        use_ssl=False,
        verify_certs=False,
    )


def ensure_index(settings: Settings, dim: int) -> None:
    """Create the vector index when absent; existing mappings remain untouched."""
    client = create_client(settings)
    index_name = settings.opensearch_index
    if client.indices.exists(index=index_name):
        LOGGER.info("OpenSearch index '%s' already exists", index_name)
        return

    body = {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 256,
            }
        },
        "mappings": {
            "properties": {
                "path": {"type": "keyword"},
                "title": {"type": "keyword"},
                "company": {"type": "keyword"},
                "symbol": {"type": "keyword"},
                "filing_type": {"type": "keyword"},
                "filing_date": {"type": "date", "ignore_malformed": True},
                "fiscal_period": {"type": "keyword"},
                "accession_number": {"type": "keyword"},
                "source_url": {"type": "keyword", "ignore_above": 4096},
                "chunk_index": {"type": "integer"},
                "content_sha256": {"type": "keyword"},
                "text": {"type": "text"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dim,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "lucene",
                    },
                },
            }
        },
    }
    LOGGER.info("Creating OpenSearch index '%s'", index_name)
    client.indices.create(index=index_name, body=body)


def knn_search(
    client: Any,
    index: str,
    query_vec: List[float],
    k: int,
    num_candidates: Optional[int] = None,
    *,
    filters: Sequence[dict[str, Any]] | None = None,
):
    """Run a vector search with optional Lucene k-NN filters."""
    knn_clause: dict[str, Any] = {
        "vector": query_vec,
        "k": k,
    }
    if filters:
        knn_clause["filter"] = {"bool": {"filter": list(filters)}}
    if num_candidates and num_candidates > k:
        knn_clause["rescore"] = {
            "oversample_factor": float(num_candidates) / float(max(k, 1))
        }

    body = {
        "size": k,
        "query": {"knn": {"embedding": knn_clause}},
        "_source": [
            "path",
            "title",
            "company",
            "symbol",
            "filing_type",
            "filing_date",
            "fiscal_period",
            "accession_number",
            "source_url",
            "chunk_index",
            "content_sha256",
            "text",
        ],
    }

    log_payload(
        LOGGER,
        "OpenSearch request payload",
        {"index": index, "body": body},
    )
    response = client.search(index=index, body=body)
    LOGGER.info("OpenSearch knn_search took %.2f ms", response.get("took", 0.0))
    log_payload(LOGGER, "OpenSearch response payload", response)
    return response


__all__ = ["create_client", "ensure_index", "knn_search"]
