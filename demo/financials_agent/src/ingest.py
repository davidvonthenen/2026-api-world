"""Ingest text-converted financial filings into an OpenSearch vector index."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterator, Sequence

from tqdm import tqdm

from .common.config import Settings, load_settings
from .common.logging import get_logger
from .common.embeddings import EmbeddingModel, to_list
from .common.opensearch_client import create_client, ensure_index


LOGGER = get_logger(__name__)
_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.-]{0,9}$")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the same set of ingestion controls as the vector-RAG base."""
    parser = argparse.ArgumentParser(description="Ingest financial filings into OpenSearch")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./quarterly_filings",
        help="Root containing text filings and optional JSON metadata sidecars",
    )
    parser.add_argument("--index-name", type=str, default=None, help="Target OpenSearch index")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        help="Maximum characters per vector chunk",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=256,
        help="Character overlap between consecutive chunks",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Embedding batch size",
    )
    return parser.parse_args(argv)


def _metadata_candidates(file_path: Path) -> tuple[Path, Path]:
    return file_path.with_suffix(".json"), file_path.with_suffix(file_path.suffix + ".json")


def _load_metadata(file_path: Path) -> dict[str, Any]:
    for candidate in _metadata_candidates(file_path):
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid metadata sidecar {candidate}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Metadata sidecar {candidate} must contain a JSON object")
        return payload
    return {}


def _infer_symbol(data_dir: Path, file_path: Path, metadata: dict[str, Any]) -> str:
    explicit = str(metadata.get("symbol") or metadata.get("ticker") or "").upper().strip()
    if explicit:
        return explicit

    # Extract symbol directly from the filename prefix (e.g., aapl-20250628.htm.md)
    filename_prefix = file_path.name.split("-")[0].upper()
    if _SYMBOL_RE.fullmatch(filename_prefix):
        return filename_prefix

    for parent in file_path.relative_to(data_dir).parents:
        if parent == Path("."):
            continue
        candidate = parent.name.upper()
        if _SYMBOL_RE.fullmatch(candidate):
            return candidate
    return ""


def _normalise_filing_type(metadata: dict[str, Any]) -> str:
    return str(
        metadata.get("filing_type")
        or metadata.get("form_type")
        or metadata.get("form")
        or ""
    ).upper().strip()


def _iter_documents(data_dir: Path) -> Iterator[dict[str, Any]]:
    """Yield text filings and governance metadata from any nested directory."""
    files = sorted(path for path in data_dir.rglob("*.md") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No .md filings found under {data_dir}")

    for file_number, file_path in enumerate(files, start=1):
        metadata = _load_metadata(file_path)
        text = file_path.read_text(encoding="utf-8", errors="replace")
        rel_path = file_path.relative_to(data_dir).as_posix()
        symbol = _infer_symbol(data_dir, file_path, metadata)
        if not _SYMBOL_RE.fullmatch(symbol):
            raise ValueError(
                f"Filing {rel_path} requires a valid ticker in its metadata sidecar "
                "or an uppercase ticker parent directory."
            )

        LOGGER.info(
            "Reading file %d/%d: %s (symbol=%s, filing_type=%s)",
            file_number,
            len(files),
            rel_path,
            symbol or "unknown",
            _normalise_filing_type(metadata) or "unknown",
        )

        yield {
            "path": rel_path,
            "title": str(metadata.get("title") or file_path.stem),
            "company": str(metadata.get("company") or metadata.get("company_name") or "").strip(),
            "symbol": symbol,
            "filing_type": _normalise_filing_type(metadata),
            "filing_date": str(metadata.get("filing_date") or metadata.get("filed_at") or "").strip(),
            "fiscal_period": str(metadata.get("fiscal_period") or metadata.get("period") or "").strip(),
            "accession_number": str(metadata.get("accession_number") or "").strip(),
            "source_url": str(metadata.get("source_url") or metadata.get("url") or "").strip(),
            "text": text,
            "_file_number": file_number,
            "_file_count": len(files),
        }


def _iter_chunks(text: str, chunk_size: int, chunk_overlap: int) -> Iterator[str]:
    """Yield the original vector-RAG overlapping character chunks unchanged."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be greater than or equal to 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    if not text:
        yield text
        return

    step = chunk_size - chunk_overlap
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        yield text[start:end]
        if end >= text_length:
            break
        start += step


def _doc_id(path: str, chunk_index: int) -> str:
    return hashlib.sha1(f"{path}:{chunk_index}".encode("utf-8")).hexdigest()


def ingest(
    data_dir: Path,
    settings: Settings,
    chunk_size: int = 2048,
    chunk_overlap: int = 256,
    batch_size: int = 32,
) -> None:
    """Chunk, embed, and index filings while replacing each source atomically by path."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    client = create_client(settings)
    embedder = EmbeddingModel(settings)
    ensure_index(settings, embedder.dimension)

    total_indexed = 0
    progress = tqdm(desc="Indexing", unit="chunks")
    try:
        for doc in _iter_documents(data_dir):
            file_number = int(doc.pop("_file_number"))
            file_count = int(doc.pop("_file_count"))
            client.delete_by_query(
                index=settings.opensearch_index,
                body={"query": {"term": {"path": doc["path"]}}},
                conflicts="proceed",
            )

            chunks = list(_iter_chunks(str(doc["text"]), chunk_size, chunk_overlap))
            LOGGER.info(
                "Chunking file %d/%d: %s -> %d chunks",
                file_number,
                file_count,
                doc["path"],
                len(chunks),
            )

            for batch_start in range(0, len(chunks), batch_size):
                batch_texts = chunks[batch_start : batch_start + batch_size]
                embeddings = embedder.encode(batch_texts)
                for offset, embedding in enumerate(embeddings):
                    chunk_index = batch_start + offset
                    chunk_text = batch_texts[offset]
                    chunk_doc = {
                        **doc,
                        "text": chunk_text,
                        "chunk_index": chunk_index,
                        "content_sha256": hashlib.sha256(chunk_text.encode("utf-8")).hexdigest(),
                        "embedding": to_list(embedding),
                    }
                    response = client.index(
                        index=settings.opensearch_index,
                        id=_doc_id(str(doc["path"]), chunk_index),
                        body=chunk_doc,
                    )
                    LOGGER.debug(
                        "Indexed %s chunk %d with status '%s'",
                        doc["path"],
                        chunk_index + 1,
                        response.get("result", "unknown"),
                    )
                    progress.update(1)
                    total_indexed += 1
    finally:
        progress.close()

    if total_indexed == 0:
        LOGGER.warning("No filing chunks were ingested")
        return

    client.indices.refresh(index=settings.opensearch_index)
    LOGGER.info(
        "Ingested %d chunks into index '%s'",
        total_indexed,
        settings.opensearch_index,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} not found")

    settings = load_settings()
    if args.index_name:
        settings.opensearch_index = args.index_name

    ingest(
        data_dir=data_dir,
        settings=settings,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
