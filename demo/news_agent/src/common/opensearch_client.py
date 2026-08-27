"""OpenSearch helpers for the historical technology-news corpus."""
from __future__ import annotations

from typing import Any, Optional, Sequence

from opensearchpy import OpenSearch
from opensearchpy.exceptions import NotFoundError

from .config import Settings
from .logging import get_logger

LOGGER = get_logger(__name__)


def create_client(settings: Settings) -> Any:
    """Create an unauthenticated local OpenSearch client."""
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


def _index_properties(dim: int) -> dict[str, Any]:
    if dim <= 0:
        raise ValueError("Embedding dimension must be greater than 0")

    return {
        "dataset_id": {"type": "keyword"},
        "record_id": {"type": "keyword"},
        "source_id": {"type": "keyword"},
        "source_path": {"type": "keyword", "ignore_above": 2048},
        # ``path`` is retained for compatibility with the original RAG evidence model.
        "path": {"type": "keyword", "ignore_above": 2048},
        "title": {"type": "keyword", "ignore_above": 1024},
        "category": {"type": "keyword"},
        "domain": {"type": "keyword"},
        "url": {"type": "keyword", "ignore_above": 4096},
        "published_at": {
            "type": "date",
            "format": "strict_date_optional_time||yyyy-MM-dd",
        },
        "tags": {"type": "keyword"},
        "chunk_index": {"type": "integer"},
        "ingested_at": {"type": "date"},
        "content_sha256": {"type": "keyword"},
        "source_content_sha256": {"type": "keyword"},
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


def _mapping_properties(mapping: dict[str, Any], index_name: str) -> dict[str, Any]:
    index_mapping = mapping.get(index_name)
    if index_mapping is None and len(mapping) == 1:
        index_mapping = next(iter(mapping.values()))
    if not isinstance(index_mapping, dict):
        return {}
    mappings = index_mapping.get("mappings", {})
    if not isinstance(mappings, dict):
        return {}
    properties = mappings.get("properties", {})
    return properties if isinstance(properties, dict) else {}


def ensure_index(settings: Settings, dim: int, *, client: Any | None = None) -> Any:
    """Create the vector index or add CSV metadata fields to an existing index.

    OpenSearch cannot change the dimension of an existing ``knn_vector`` field. A
    mismatched index therefore fails with an actionable message rather than producing
    a much more theatrical error during bulk ingestion.
    """
    client = client or create_client(settings)
    index_name = settings.opensearch_index
    expected_properties = _index_properties(dim)

    if client.indices.exists(index=index_name):
        mapping = client.indices.get_mapping(index=index_name)
        existing_properties = _mapping_properties(mapping, index_name)
        existing_embedding = existing_properties.get("embedding", {})
        existing_dim = existing_embedding.get("dimension")
        if existing_dim is not None and int(existing_dim) != dim:
            raise ValueError(
                f"OpenSearch index '{index_name}' uses embedding dimension "
                f"{existing_dim}, but model '{settings.embedding_model}' uses {dim}. "
                "Re-run ingestion with --recreate-index."
            )

        missing_properties = {
            name: definition
            for name, definition in expected_properties.items()
            if name not in existing_properties
        }
        if missing_properties:
            LOGGER.info(
                "Adding %d CSV metadata field(s) to OpenSearch index '%s'",
                len(missing_properties),
                index_name,
            )
            client.indices.put_mapping(
                index=index_name,
                body={"properties": missing_properties},
            )
        else:
            LOGGER.info("OpenSearch index '%s' already exists", index_name)
        return client

    body = {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 256,
            }
        },
        "mappings": {"properties": expected_properties},
    }
    LOGGER.info("Creating OpenSearch index '%s'", index_name)
    client.indices.create(index=index_name, body=body)
    return client


def knn_search(
    client: Any,
    index: str,
    query_vec: Sequence[float],
    k: int,
    num_candidates: Optional[int] = None,
) -> dict[str, Any]:
    """Execute Lucene HNSW search with explicit query-time candidate breadth."""
    if k <= 0:
        raise ValueError("k must be greater than 0")
    if not query_vec:
        raise ValueError("query_vec must not be empty")

    query: dict[str, Any] = {
        "vector": [float(value) for value in query_vec],
        "k": k,
    }
    if num_candidates is not None:
        if num_candidates <= 0:
            raise ValueError("num_candidates must be greater than 0")
        # For Lucene HNSW, ef_search controls how many vectors are examined. The
        # larger of k and ef_search is passed to the engine; top-level size still
        # limits the final response to k hits.
        query["method_parameters"] = {
            "ef_search": max(k, int(num_candidates)),
        }

    body = {
        "size": k,
        "query": {"knn": {"embedding": query}},
        "_source": [
            "dataset_id",
            "record_id",
            "path",
            "source_path",
            "source_id",
            "title",
            "category",
            "domain",
            "url",
            "published_at",
            "tags",
            "chunk_index",
            "ingested_at",
            "content_sha256",
            "source_content_sha256",
            "text",
        ],
    }

    response = client.search(index=index, body=body)
    LOGGER.info("OpenSearch knn_search took %.2f ms", response.get("took", 0.0))
    return response


__all__ = ["create_client", "ensure_index", "knn_search"]
