"""Ingest historical technology-news stories from a CSV file into OpenSearch."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from collections.abc import Iterator, Sequence
from itertools import chain
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from tqdm import tqdm

from .common.config import Settings, load_settings
from .common.logging import get_logger

LOGGER = get_logger(__name__)

DEFAULT_CSV_FILE = "./ai_media_dataset_20250911-1000lines.csv"
_REQUIRED_COLUMNS = {"title", "content"}
_IDENTIFIER_COLUMNS = ("id", "source_id", "article_id", "record_id", "index", "")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest historical technology-news stories from CSV into OpenSearch"
    )
    parser.add_argument(
        "--csv-file",
        default=DEFAULT_CSV_FILE,
        help=f"CSV dataset path (default: {DEFAULT_CSV_FILE})",
    )
    parser.add_argument(
        "--dataset-id",
        default=None,
        help=(
            "Stable dataset identifier used for snapshot replacement. "
            "Defaults to the CSV filename."
        ),
    )
    parser.add_argument("--index-name", default=None, help="Target OpenSearch index")
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--chunk-overlap", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--recreate-index",
        action="store_true",
        help="Delete and recreate the target index before ingestion",
    )
    return parser.parse_args(argv)


def _normalize_header(value: str | None) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def _normalize_text(value: object) -> str:
    return str(value or "").replace("\x00", "").replace("\r\n", "\n").strip()


def _parse_serialized_list(value: object) -> list[str]:
    """Parse JSON/Python-list columns while accepting ordinary text as a fallback."""
    raw = _normalize_text(value)
    if not raw:
        return []

    parsed: object = raw
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(raw)
            break
        except (ValueError, SyntaxError, TypeError):
            continue

    if isinstance(parsed, (list, tuple, set)):
        values = [_normalize_text(item) for item in parsed]
        return [item for item in values if item]
    if isinstance(parsed, str):
        text = _normalize_text(parsed)
        return [text] if text else []

    text = _normalize_text(parsed)
    return [text] if text else []


def _normalize_url(value: object) -> str:
    raw = _normalize_text(value)
    if not raw:
        return ""
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return ""
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return ""
    return urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path or "/",
            parsed.query,
            "",  # Fragments do not identify a different article.
        )
    )


def _normalize_published_at(value: object) -> str | None:
    raw = _normalize_text(value)
    if not raw:
        return None

    candidate = raw.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(candidate)
        return parsed.isoformat() if "T" in raw else parsed.date().isoformat()
    except ValueError:
        pass

    try:
        return date.fromisoformat(raw).isoformat()
    except ValueError:
        return None


def _domain_from_url(url: str) -> str:
    if not url:
        return ""
    host = urlsplit(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _row_value(row: dict[str, str], normalized_columns: dict[str, str], name: str) -> str:
    original = normalized_columns.get(name)
    return row.get(original, "") if original is not None else ""


def _row_identifier(
    row: dict[str, str],
    normalized_columns: dict[str, str],
    row_number: int,
) -> str:
    for name in _IDENTIFIER_COLUMNS:
        value = _normalize_text(_row_value(row, normalized_columns, name))
        if value:
            return value
    return str(row_number)


def _source_id(
    *,
    dataset_id: str,
    record_id: str,
    title: str,
    url: str,
) -> str:
    identity = url or f"{dataset_id}:{record_id}:{title}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _iter_documents(csv_file: Path, dataset_id: str | None = None) -> Iterator[dict[str, Any]]:
    """Yield normalized article records from the supplied AI media CSV dataset."""
    dataset_id = _normalize_text(dataset_id) or csv_file.name

    with csv_file.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV file has no header row: {csv_file}")

        normalized_columns = {
            _normalize_header(original): original for original in reader.fieldnames
        }
        missing = sorted(_REQUIRED_COLUMNS - set(normalized_columns))
        if missing:
            raise ValueError(
                f"CSV file is missing required column(s): {', '.join(missing)}"
            )

        for row_number, row in enumerate(reader, start=2):
            # Injecting our real-time visibility printout
            print(f"Currently processing line number: {row_number}", end="\r", flush=True)

            paragraphs = _parse_serialized_list(
                _row_value(row, normalized_columns, "content")
            )
            text = "\n\n".join(paragraphs).strip()
            if not text:
                LOGGER.warning("Skipping CSV row %d because content is empty", row_number)
                continue

            record_id = _row_identifier(row, normalized_columns, row_number)
            title = _normalize_text(_row_value(row, normalized_columns, "title"))
            if not title:
                title = text.splitlines()[0][:300] or f"Article {record_id}"
            title = title[:500]

            url = _normalize_url(_row_value(row, normalized_columns, "url"))
            domain = _normalize_text(_row_value(row, normalized_columns, "domain"))
            domain = domain.lower() or _domain_from_url(url)
            tags = _parse_serialized_list(_row_value(row, normalized_columns, "tags"))
            published_at = _normalize_published_at(
                _row_value(row, normalized_columns, "date")
            )
            source_id = _source_id(
                dataset_id=dataset_id,
                record_id=record_id,
                title=title,
                url=url,
            )
            source_path = f"{csv_file.name}#row={row_number}&id={record_id}"

            yield {
                "dataset_id": dataset_id,
                "record_id": record_id,
                "source_id": source_id,
                "source_path": source_path,
                # Keep the established field for downstream compatibility.
                "path": source_path,
                "title": title,
                "category": "technology_news",
                "domain": domain,
                "url": url,
                "published_at": published_at,
                "tags": tags,
                "text": text,
                "source_content_sha256": hashlib.sha256(
                    text.encode("utf-8")
                ).hexdigest(),
            }


def _iter_chunks(text: str, chunk_size: int, chunk_overlap: int) -> Iterator[str]:
    """Yield overlapping character chunks using the established RAG algorithm."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be greater than or equal to 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    if not text:
        return

    step = chunk_size - chunk_overlap
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        yield text[start:end]
        if end >= len(text):
            break
        start += step


def _doc_id(dataset_id: str, source_id: str, chunk_index: int) -> str:
    """Return a stable chunk ID scoped to one logical dataset snapshot."""
    identity = f"{dataset_id}:{source_id}:{chunk_index}"
    return hashlib.sha1(identity.encode("utf-8")).hexdigest()


def _embedding_text(document: dict[str, Any], chunk_text: str) -> str:
    """Add article metadata to the vector input without changing cited chunk text."""
    lines = [f"Title: {document['title']}"]
    if document.get("published_at"):
        lines.append(f"Published: {document['published_at']}")
    if document.get("domain"):
        lines.append(f"Source: {document['domain']}")
    tags = document.get("tags") or []
    if tags:
        lines.append("Tags: " + ", ".join(str(tag) for tag in tags))
    lines.append("")
    lines.append(chunk_text)
    return "\n".join(lines)


def ingest(
    csv_file: Path,
    settings: Settings,
    *,
    dataset_id: str | None = None,
    chunk_size: int = 2048,
    chunk_overlap: int = 256,
    batch_size: int = 32,
    recreate_index: bool = False,
) -> dict[str, int | str]:
    """Parse, chunk, batch-embed, and bulk-index the historical news CSV."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    effective_dataset_id = _normalize_text(dataset_id) or csv_file.name
    document_iterator = _iter_documents(csv_file, effective_dataset_id)
    try:
        first_document = next(document_iterator)
    except StopIteration:
        LOGGER.warning("No usable CSV stories were found in %s", csv_file)
        return {
            "dataset_id": effective_dataset_id,
            "articles": 0,
            "chunks": 0,
            "index": settings.opensearch_index,
        }

    from opensearchpy.helpers import bulk

    try:
        from .common.embeddings import EmbeddingModel, to_list
        from .common.opensearch_client import create_client, ensure_index
    except ImportError:  # pragma: no cover - direct script execution fallback
        from common.embeddings import EmbeddingModel, to_list
        from common.opensearch_client import create_client, ensure_index

    client = create_client(settings)
    try:
        embedder = EmbeddingModel(settings)

        if recreate_index and client.indices.exists(index=settings.opensearch_index):
            LOGGER.warning("Deleting OpenSearch index '%s'", settings.opensearch_index)
            client.indices.delete(index=settings.opensearch_index)

        ensure_index(settings, embedder.dimension, client=client)

        # The CSV is treated as a dataset snapshot. Re-ingestion replaces prior rows
        # from this dataset, including records removed from a newer snapshot.
        deletion = client.delete_by_query(
            index=settings.opensearch_index,
            body={"query": {"term": {"dataset_id": effective_dataset_id}}},
            conflicts="proceed",
            refresh=True,
        )
        LOGGER.info(
            "Removed %s prior chunks for dataset '%s'",
            deletion.get("deleted", 0),
            effective_dataset_id,
        )

        ingested_at = datetime.now(timezone.utc).isoformat()
        pending: list[tuple[dict[str, Any], int, str]] = []
        article_count = 0
        total_indexed = 0
        progress = tqdm(desc="Indexing", unit="chunks")

        def flush_pending() -> int:
            nonlocal pending
            if not pending:
                return 0

            texts = [
                _embedding_text(document, chunk_text)
                for document, _, chunk_text in pending
            ]
            embeddings = embedder.encode(texts)
            actions: list[dict[str, Any]] = []
            for (document, chunk_index, chunk_text), embedding in zip(
                pending, embeddings, strict=True
            ):
                body = {
                    **document,
                    "text": chunk_text,
                    "chunk_index": chunk_index,
                    "ingested_at": ingested_at,
                    "content_sha256": hashlib.sha256(
                        chunk_text.encode("utf-8")
                    ).hexdigest(),
                    "embedding": to_list(embedding),
                }
                actions.append(
                    {
                        "_op_type": "index",
                        "_index": settings.opensearch_index,
                        "_id": _doc_id(
                            document["dataset_id"],
                            document["source_id"],
                            chunk_index,
                        ),
                        "_source": body,
                    }
                )

            indexed, _ = bulk(
                client,
                actions,
                raise_on_error=True,
                raise_on_exception=True,
                refresh=False,
            )
            progress.update(indexed)
            pending = []
            return int(indexed)

        try:
            for document in chain((first_document,), document_iterator):
                article_count += 1
                chunk_count = 0
                for chunk_index, chunk_text in enumerate(
                    _iter_chunks(document["text"], chunk_size, chunk_overlap)
                ):
                    chunk_count += 1
                    pending.append((document, chunk_index, chunk_text))
                    if len(pending) >= batch_size:
                        total_indexed += flush_pending()
                LOGGER.debug(
                    "Prepared article %s with %d chunks",
                    document["source_path"],
                    chunk_count,
                )
            total_indexed += flush_pending()
        finally:
            progress.close()

        if total_indexed:
            client.indices.refresh(index=settings.opensearch_index)
        else:
            LOGGER.warning("No CSV chunks were ingested from %s", csv_file)

        LOGGER.info(
            "Ingested %d articles and %d chunks from '%s' into index '%s'",
            article_count,
            total_indexed,
            csv_file,
            settings.opensearch_index,
        )
        return {
            "dataset_id": effective_dataset_id,
            "articles": article_count,
            "chunks": total_indexed,
            "index": settings.opensearch_index,
        }
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    csv_file = Path(args.csv_file).expanduser()
    if not csv_file.is_file():
        raise FileNotFoundError(f"CSV dataset not found: {csv_file}")
    if csv_file.suffix.lower() != ".csv":
        raise ValueError(f"Expected a .csv dataset, received: {csv_file}")

    settings = load_settings()
    if args.index_name:
        settings.opensearch_index = args.index_name

    result = ingest(
        csv_file=csv_file,
        settings=settings,
        dataset_id=args.dataset_id,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        recreate_index=args.recreate_index,
    )
    print(
        "Ingested {articles} articles as {chunks} chunks into '{index}' "
        "(dataset_id={dataset_id}).".format(**result)
    )


if __name__ == "__main__":
    main()
