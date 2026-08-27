# Governed News Agent

This package implements the technology-news specialist used by the API World agent-routing demonstration. It combines a historical OpenSearch corpus with current Tavily news, generates an evidence-grounded answer, verifies the result, and records an append-only audit event before release.

The server exposes one A2A interface:

- Protocol binding: `JSONRPC`
- Protocol version: `1.0`
- Default endpoint: `http://127.0.0.1:9001/`

The package does not implement stock prices, SEC filings, earnings analysis, valuation, or investment recommendations. Those requests are redirected to the Financial Agent boundary before any model or retrieval call.

## Architecture

```text
Existing Host Orchestrator
        |
        | A2A 1.0 JSON-RPC
        v
+------------------------------------------------------+
| Governed News Agent                                  |
|                                                      |
|  1. Extract effective user question                  |
|  2. Enforce News-versus-Financial authority          |
|  3. Nemotron route planning                          |
|  4. Parallel historical and current retrieval        |
|  5. Qwen evidence-grounded synthesis                 |
|  6. Deterministic citation and domain checks         |
|  7. Nemotron release verification                    |
|  8. Bounded correction attempt                       |
|  9. Release, redirect, or withhold                    |
| 10. Append-only audit record                         |
+------------------------+-----------------------------+
                         |
             +-----------+-----------+
             |                       |
             v                       v
   Historical OpenSearch      Tavily MCP Server
   CSV-backed vector RAG      Streamable HTTP
```

## Current implementation changes

### A2A protocol

The server now uses the current A2A task lifecycle only:

- The Agent Card advertises one JSON-RPC protocol `1.0` interface.
- The Starlette application mounts the current `create_jsonrpc_routes()` route factory.
- A task is published before status and artifact events for new and continued tasks.
- The governed answer is returned in the `news_result` artifact.
- The terminal task status contains a concise completion message rather than a second copy of the answer.
- Server shutdown drains active task work through `DefaultRequestHandler.aclose()`.
- The direct scaffold stream shim and its completion flags have been removed.

### Historical CSV ingestion

`src.ingest` now reads news stories from a CSV file. The supplied sample dataset is included at:

```text
data/ai_media_dataset_20250911-SAMPLE.csv
```

The parser supports the sample schema:

| Column | Use |
| --- | --- |
| Blank first column | Stable source record identifier when present |
| `title` | Article title |
| `date` | Publication date |
| `content` | Serialized JSON or Python list of article paragraphs |
| `domain` | Publisher or source label |
| `url` | Original article URL |
| `tags` | Serialized JSON or Python list of tags |

Only `title` and `content` are required. Invalid or missing URLs are omitted rather than promoted into trusted source metadata.

The ingestion path:

1. Parses and normalizes each CSV row.
2. Joins serialized content paragraphs into article text.
3. Assigns a stable article identifier from the canonical URL, or from dataset and record metadata when no URL exists.
4. Assigns stable chunk identifiers scoped to the selected `dataset_id`.
5. Splits article text into overlapping character chunks.
6. Embeds a retrieval representation containing the title, date, source, tags, and chunk text.
7. Stores the original chunk text separately for evidence and citations.
8. Indexes article and chunk metadata in OpenSearch.
9. Replaces prior chunks with the same `dataset_id` when the CSV snapshot is re-ingested.

### Historical retrieval

The OpenSearch index stores:

- Dataset and record identifiers
- Stable article and chunk identifiers
- CSV record path
- Article title and category
- Source domain and article URL
- Publication and ingestion timestamps
- Tags and chunk index
- Chunk and article content hashes
- Chunk text and normalized embedding

Retrieval uses Lucene HNSW cosine similarity. Query-time breadth is controlled with `method_parameters.ef_search`. The retriever fetches more chunks than it releases, then limits how many chunks one article can occupy before filling remaining evidence slots. This reduces the tendency of overlapping chunks from one long article to crowd out every other source.

### A2A query client

`src.query` is an A2A client rather than an in-process agent call. It:

1. Resolves the Agent Card from the configured server URL.
2. Requires a JSON-RPC protocol `1.0` interface.
3. Creates an SDK client with `create_client()`.
4. Sends a `SendMessageRequest` with a user-role text message.
5. Processes task, message, status, and artifact stream responses.
6. Prints status updates to standard error.
7. Prints the final `news_result` artifact to standard output.
8. Exits with an error when the A2A task is failed, canceled, or rejected.

This allows command-line testing to exercise the same network and task lifecycle used by the host orchestrator.

## Repository layout

```text
news_agent/
├── data/
│   └── ai_media_dataset_20250911-SAMPLE.csv
├── src/
│   ├── common/
│   │   ├── config.py
│   │   ├── embeddings.py
│   │   ├── logging.py
│   │   ├── opensearch_client.py
│   │   └── tavily_client.py
│   ├── news_agent/
│   │   ├── __main__.py
│   │   ├── agent_executor.py
│   │   ├── audit.py
│   │   ├── governance.py
│   │   ├── llm.py
│   │   ├── mcp_news.py
│   │   ├── models.py
│   │   ├── news_agent.py
│   │   ├── rag.py
│   │   └── utils.py
│   ├── ingest.py
│   ├── query.py
│   └── tavily_mcp_server.py
├── tests/
├── .env.example
├── Makefile
└── requirements.txt
```

## Prerequisites

The complete demonstration expects:

- Python 3.11 or 3.12
- OpenSearch at `127.0.0.1:9200`
- Nemotron Orchestrator through an OpenAI-compatible endpoint at port `8002`
- Qwen through an OpenAI-compatible endpoint at port `8001`
- A Tavily API key for current-news retrieval

Create an isolated environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

Copy the environment template:

```bash
cp .env.example .env
```

Set at least:

```bash
TAVILY_API_KEY=<your-key>
```

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENSEARCH_HOST` | `127.0.0.1` | Historical corpus host |
| `OPENSEARCH_PORT` | `9200` | Historical corpus port |
| `OPENSEARCH_INDEX` | `techcomp-vector-chunks` | Vector index |
| `EMBEDDING_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | Document and query embeddings |
| `RAG_TOP_K` | `5` | Final historical evidence count |
| `RAG_NUM_CANDIDATES` | `5` | Lucene HNSW query breadth |
| `RAG_MAX_CHUNKS_PER_SOURCE` | `2` | Preferred maximum chunks per article |
| `ORCH_URL` | `http://127.0.0.1:8002/v1` | Nemotron endpoint |
| `LLM_URL` | `http://127.0.0.1:8001/v1` | Qwen endpoint |
| `TAVILY_MCP_URL` | `http://127.0.0.1:8765/mcp` | Current-news MCP endpoint |
| `NEWS_MAX_RETRIES` | `1` | Number of correction attempts after the initial draft |
| `RELEASE_CHECKS_ENABLED` | `true` | Enable advisory Nemotron verification and fail-closed audit persistence |
| `POLICY_CHECKS_ENABLED` | `true` | Enforce the deterministic News-versus-Financial authority boundary |
| `EVIDENCE_CHECKS_ENABLED` | `true` | Require registered inline citations and stop when no evidence is available |
| `AUDIT_LOG_PATH` | `./logs/news-agent-audit.jsonl` | Append-only audit log |
| `AUDIT_INCLUDE_QUERY` | `false` | Whether raw query text is persisted |
| `A2A_HOST` | `0.0.0.0` | A2A bind address |
| `A2A_PORT` | `9001` | A2A server port |
| `APP_URL` | `http://127.0.0.1:9001` | Public URL published in the Agent Card |

## Ingest the supplied CSV dataset

Run:

```bash
python -m src.ingest \
  --csv-file ./data/ai_media_dataset_20250911-SAMPLE.csv \
  --dataset-id ai-media-20250911-sample \
  --recreate-index
```

The equivalent Make target is:

```bash
make ingest-recreate
```

For later snapshot updates that should retain the index mapping:

```bash
make ingest
```

Useful ingestion options:

```text
--csv-file PATH
--dataset-id ID
--index-name NAME
--chunk-size 2048
--chunk-overlap 256
--batch-size 32
--recreate-index
```

`--dataset-id` defines the replacement boundary. Re-ingesting the same dataset ID removes its prior chunks before indexing the new snapshot. Other dataset IDs in the same index remain untouched.

## Start the services

Run each component in its own terminal after OpenSearch and the two model endpoints are available.

### Tavily MCP server

```bash
python -m src.tavily_mcp_server
```

Default endpoint:

```text
http://127.0.0.1:8765/mcp
```

### A2A News Agent

```bash
python -m src.news_agent
```

Default endpoints:

```text
Agent Card:  http://127.0.0.1:9001/.well-known/agent-card.json
A2A JSON-RPC: http://127.0.0.1:9001/
Health:       http://127.0.0.1:9001/health
```

The health response reports the configured protocol, OpenSearch index, MCP URL, and model identifiers.

## Query the News Agent over A2A

```bash
python -m src.query \
  --url http://127.0.0.1:9001 \
  --question "What is Microsoft doing in AI?"
```

The Make target uses the same A2A client:

```bash
make query QUESTION="What is Microsoft doing in AI?"
```

Client options:

```text
--url URL
--timeout SECONDS
--stream / --no-stream
--show-progress / --no-show-progress
```

Progress appears on standard error. The governed final artifact appears on standard output, which keeps the command usable in scripts and demonstrations.

## Governance behavior

### Authority boundary

With the default `POLICY_CHECKS_ENABLED=true`, financial-only requests are redirected
before Nemotron, OpenSearch, Tavily, or Qwen is called. Mixed requests retain the news
portion and add a scope notice for financial content. Policy enforcement may be disabled
for diagnostics, while the agent still retrieves only from its historical-news corpus
and Tavily MCP tool.

### Application-owned citations

Historical evidence receives `H1`, `H2`, and related identifiers. Current Tavily evidence receives `W1`, `W2`, and related identifiers. The synthesis model may cite only those registered identifiers. Application code constructs the final source appendix from indexed metadata.

Historical source entries include the original CSV article URL and publication date when available, plus the CSV record locator and chunk index used for retrieval.

### Verification and retry

Enabled checks apply independently:

1. Evidence checks require at least one registered `H#` or `W#` citation and reject unknown identifiers.
2. Policy checks reject financial-domain output patterns.
3. Release checks invoke Nemotron for advisory support, date, injection, and boundary review; deterministic checks own the release decision.
4. One bounded correction attempt runs when an enabled check rejects the draft.
5. The corrected draft is withheld only when an enabled check still fails.

`REQUIRE_LLM_VERIFIER` has an effect only when
`RELEASE_CHECKS_ENABLED=true`.

### Audit persistence

Each request produces one JSON Lines audit record. Raw evidence text is excluded.
Query text is excluded unless `AUDIT_INCLUDE_QUERY=true`. When release checks are
enabled, an audit write failure withholds the answer. When release checks are disabled,
the answer is released with an explicit audit warning.

## Tests

Run:

```bash
make check
```

The suite covers:

- A2A protocol-card and route configuration
- A2A query-client message flow and artifact extraction
- CSV list parsing, metadata normalization, chunking, and stable identities
- Dataset-snapshot replacement and bulk-index payloads
- OpenSearch mappings and Lucene `ef_search` query construction
- Article-diverse historical evidence selection
- News-versus-financial routing
- Citation ownership and deterministic verification
- Bounded retry, verifier rejection, and audit behavior
- Tavily news parameters and URL filtering

## Common failures

### Existing index has a different embedding dimension

Recreate the index with the currently configured embedding model:

```bash
python -m src.ingest \
  --csv-file ./data/ai_media_dataset_20250911-SAMPLE.csv \
  --recreate-index
```

### Query client rejects the Agent Card

Confirm that `--url` points to this News Agent and that its Agent Card advertises JSON-RPC protocol `1.0`.

### Historical retrieval returns no evidence

Check the configured index and document count, then verify that ingestion and query processes use the same `EMBEDDING_MODEL` and `OPENSEARCH_INDEX` values.

### Current-news retrieval fails

Check `TAVILY_API_KEY`, `TAVILY_MCP_URL`, and the MCP server health and logs. Historical retrieval can still provide evidence when the routing plan permits it, but the released answer will disclose the missing source.
