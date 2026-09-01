# Vertical API Orchestrator Agent

> **Conference demonstration only.** This package illustrates routing, A2A delegation,
> bounded orchestration, external answer synthesis, and a lightweight audit trail. It is
> not a production compliance, investment-advice, or trading system.

This package implements the **Orchestrator Agent** for the API World session
**“Vertical APIs for Agentic AI: Routing the Right Context to the Right Expert.”**
It exposes a narrow OpenAI-compatible API while preserving the existing
`src/host_agent` layout so changes remain readable in a directory diff.

The News Agent and Financial Agent remain independent external A2A services. They own
retrieval, MCP access, source selection, citations, and domain answer generation. The
Orchestrator does not import or directly call Tavily, Finnhub, OpenSearch, SEC APIs,
MCP servers, or other domain-data providers.

## Architecture

```text
OpenAI Python SDK client
        |
        | POST /v1/chat/completions
        v
+-------------------------------------------------------------+
| Orchestrator Agent :10000                                   |
|                                                             |
|  1. Deterministic request and identity assessment           |
|  2. Nemotron routing plan via :8002/v1                      |
|  3. Optional policy validation and deterministic repair     |
|  4. Bounded A2A 1.1.2 specialist calls                     |
|  5. Nemotron completion decision and synthesis guidance     |
|  6. Qwen combined-answer synthesis via :8001/v1             |
|  7. Configurable mechanical governance checks               |
|  8. Hash-chained JSONL audit append                         |
+------------------------+------------------------------------+
                         |
              +----------+-----------+
              |                      |
              v                      v
      News Agent :9001       Financial Agent :9002
      A2A JSON-RPC           A2A JSON-RPC
      Historical/current     Filings/current market data
      news and citations     and citations
```

Nemotron has two control-plane responsibilities:

1. Produce a structured routing and specialist-call plan.
2. Decide whether the bounded workflow is complete and, for a combined result, provide
   fact-free guidance to the separate synthesis model.

Nemotron does not generate the domain answer. The planning call receives only the
current user turn plus deterministic routing metadata. Earlier-turn identity continuity
is represented in that metadata and cannot broaden the current domain route. The
completion call receives the user request and
specialist status metadata, including status, citation count, warning count, and
private/unlisted state. It does not receive specialist evidence or response prose.

For a combined result, the independently served synthesis model receives the News and
Financial evidence packages and produces one cohesive answer. The Orchestrator rejects
synthesis output that introduces citations absent from the specialist responses. If
synthesis is unavailable or invalid, it preserves the two audited specialist results in
separately labeled sections instead of discarding otherwise usable output.

## Supported routes

| Route | Behavior |
|---|---|
| `news` | Calls only the News Agent for announcements, AI products, partnerships, acquisitions, leadership, strategy, recent events, and historical news context. |
| `financial` | Calls only the Financial Agent for symbol resolution, stock prices, earnings, quarterly results, filings, and financial performance. |
| `combined` | Calls both specialists concurrently and releases either a validated synthesis or separately labeled audited specialist sections. |
| `clarification` | Asks one focused question when one company or one requested domain cannot be resolved. |
| `unsupported` | Returns a bounded notice for requests outside company news and financial information. |

The demo supports one primary company or security identity per request. A company and
one user-supplied linked ticker, such as `NVIDIA` and `NVDA`, may be treated as one
identity. Actual multi-company or multi-security comparisons remain ambiguous rather
than being silently collapsed.

## Financial evidence routing

The Orchestrator still does not retrieve financial data. It deterministically preserves
which Financial Agent evidence path the request requires when constructing the A2A
specialist prompt:

- Stock price, quote, market capitalization, trading, and other market-data questions
  explicitly request the Financial Agent's approved current-data MCP path, including
  Finnhub when configured.
- Latest or current earnings and financial-result questions request current-data
  evidence and filing evidence.
- SEC-specific, quarterly, historical, comparative, and longitudinal questions request
  filing retrieval.
- User-supplied tickers are preserved. The Orchestrator never invents or substitutes a
  ticker.

This request is constructed deterministically after the route is selected. A model
rewrite therefore cannot accidentally remove the current-price or current-results
intent before the Financial Agent receives it.

## Combined-answer synthesis

The News and Financial calls remain independent and run with `asyncio.gather()` for a
`combined` route. Their raw responses are retained only as bounded inputs to the
external synthesis model and as hashed specialist metadata in the Orchestrator audit.

The synthesis workflow:

1. Nemotron receives status-only metadata and decides whether the workflow is complete.
2. Nemotron supplies fact-free synthesis guidance for the user's requested relationship
   or comparison.
3. Qwen, or another configured OpenAI-compatible synthesis model, receives the original
   user request, the guidance, and both specialist evidence packages.
4. The Orchestrator validates citation provenance against the specialist responses.
5. Invalid synthesis output may be regenerated within the existing bounded model retry.
6. If synthesis remains unavailable or invalid, the user receives the two governed
   specialist results in clearly separated sections with a composition notice.

The synthesis model may use only specialist-provided evidence. Claim-level evidence
verification remains the responsibility of the specialist agents; the Orchestrator
checks citation identity and response-level evidence boundaries.

## Private-company behavior

The Orchestrator never invents a ticker.

For a financial request whose public-company status is unknown:

1. The Financial Agent resolves the company or security from approved evidence.
2. If it reports no verified public security, the Orchestrator marks the result as
   private or unlisted for that request.
3. That result does not automatically require a News Agent call.
4. The released response states that public market data is unavailable and that no
   ticker was inferred or substituted.

A news-only request for a private company does not call the Financial Agent.

## Bounded execution contract

The following limits are fixed in `Settings` and are not ordinary tuning flags:

| Limit | Value |
|---|---:|
| Primary companies per request | 1 |
| Specialist calls per request | 2 |
| Clarification turns | 1 |
| Orchestration rounds | 3 |
| Nemotron schema retries after the first attempt | 1 |
| Synthesis attempts | At most 2, using `max_specialist_calls` |

The three orchestration rounds are:

1. Request assessment and route planning.
2. Specialist execution.
3. Completion decision, optional unused-specialist escalation, and final synthesis.

## Governance behavior

Governance categories are independently controlled by `Settings`:

- **Policy checks** validate route authority, grounded company identity, user-supplied
  tickers, and specialist-call shape. When disabled, deterministic assessment becomes
  advisory, but plans are still repaired to one grounded identity and the News and
  Financial allowlist remains fixed.
- **Release checks** cover call budgets, allowlisted specialists, duplicate envelopes,
  route shape, completion decisions, combined-response composition, and fail-closed audit
  persistence.
- **Evidence checks** require cited specialist evidence or an explicit evidence-limit
  result and verify that every citation emitted by the synthesis model came from a
  specialist package.

The shipped `.env.example` and the `Settings` dataclass enable all three categories:

```text
RELEASE_CHECKS_ENABLED=true
POLICY_CHECKS_ENABLED=true
EVIDENCE_CHECKS_ENABLED=true
```

Disabling policy or release checks does not create unrestricted tools or routes. Only the
two configured A2A specialist connections exist. When combined synthesis is unavailable,
the already governed specialist results may be released in separate sections.

## Audit trail

Each request appends one JSON object to `logs/orchestrator-audit.jsonl` by default.
Records are linked through `previous_hash` and `entry_hash` SHA-256 fields. Before a
new record is written, the existing chain is verified.

The default audit record includes:

- Request and audit identifiers.
- A hash of the caller or session identity.
- Query hash and length rather than raw query text.
- Deterministic assessment and redacted routing-plan metadata.
- Model endpoint, model name, purpose, attempts, latency, and validation outcome.
- Hashes and lengths for specialist prompts, model reasons, and synthesis guidance.
- Specialist statuses, A2A task identifiers, response hashes, citation counts,
  warnings, latency, and specialist audit identifiers.
- Synthesis attempt count, model and endpoint, source/output citation provenance,
  output hash and length, and release status. Raw synthesized prose is not stored.
- Mechanical governance checks and the final outcome.

Raw user queries can be enabled for stage debugging with
`AUDIT_INCLUDE_QUERY=true`. Raw specialist and synthesized responses are not written to
the Orchestrator audit file.

Verify the chain with:

```bash
make audit
```

or:

```bash
python -m src.host_agent.audit ./logs/orchestrator-audit.jsonl
```

This is a tamper-evident demonstration log, not a signed or externally anchored
compliance ledger.

## Public API

The Flask service exposes one application route:

```text
POST /v1/chat/completions
```

Streaming responses and additional public routes are not implemented. The response
uses the Chat Completions shape expected by the OpenAI Python SDK.

Example request body:

```json
{
  "model": "vertical-api-orchestrator",
  "messages": [
    {
      "role": "user",
      "content": "What is NVIDIA doing in AI, and how is its stock performing?"
    }
  ],
  "stream": false,
  "user": "api-world-demo"
}
```

A released combined response ends with a visible trace similar to:

```text
route=combined | company=NVIDIA | specialists=news,financial |
execution=parallel | composition=synthesized | rounds=3 |
policy=passed | audit_id=<uuid>
```

## Prerequisites

- Python 3.10 or newer.
- Nemotron Orchestrator available through an OpenAI-compatible endpoint at
  `http://127.0.0.1:8002/v1` by default.
- Qwen synthesis service available through an OpenAI-compatible endpoint at
  `http://127.0.0.1:8001/v1` by default.
- News Agent A2A service at `http://127.0.0.1:9001` by default.
- Financial Agent A2A service at `http://127.0.0.1:9002` by default.

## Installation

From the package root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

The outbound A2A dependency is pinned as required:

```text
a2a-sdk==1.1.2
```

## Running the Orchestrator

Start Nemotron, Qwen, the News Agent, and the Financial Agent in their respective
packages. Then run:

```bash
make agent
```

Equivalent command:

```bash
python -m src.host_agent
```

The service listens on `0.0.0.0:10000` by default.

## Running the conversation client

The supplied `client.py` uses the OpenAI Python SDK and retains conversation history in
memory until the process exits.

```bash
make client
```

Example prompts:

```text
What is NVIDIA doing in generative AI?
What is NVIDIA's current stock price?
What are NVIDIA's latest quarterly financial results?
What is NVIDIA doing in AI, and how has that affected its stock price? The ticker is NVDA.
How is Anthropic stock performing?
Tell me about Apple.
What is the weather today?
```

A clarification can be answered in the same process:

```text
You: Tell me about Apple.
Orchestrator: For Apple, do you want company news, financial information, or both?
You: Both.
```

## Configuration

| Environment variable | Code default | Purpose |
|---|---|---|
| `ORCHESTRATOR_HOST` | `0.0.0.0` | Public API bind address. |
| `ORCHESTRATOR_PORT` | `10000` | Public API port. |
| `ORCHESTRATOR_PUBLIC_MODEL` | `vertical-api-orchestrator` | Public model label. |
| `ORCH_URL` | `http://127.0.0.1:8002/v1` | OpenAI-compatible Nemotron endpoint. |
| `ORCH_API_KEY` | `not-needed` | Nemotron endpoint API key. |
| `ORCH_MODEL` | `nvidia/Nemotron-Orchestrator-8B` | Orchestration model identifier. |
| `ORCH_REQUEST_TIMEOUT` | `600` | Nemotron request timeout. |
| `ORCH_MAX_TOKENS` | `131072` | Maximum Nemotron response tokens requested. |
| `LLM_URL` | `http://127.0.0.1:8001/v1` | OpenAI-compatible synthesis endpoint. |
| `LLM_API_KEY` | `not-needed` | Synthesis endpoint API key. |
| `LLM_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | Synthesis model identifier. |
| `LLM_TEMPERATURE` | `0.2` | Synthesis temperature. |
| `LLM_TOP_P` | `0.9` | Synthesis nucleus-sampling value. |
| `LLM_MAX_TOKENS` | `131072` | Maximum synthesis response tokens requested. |
| `LLM_REQUEST_TIMEOUT` | `600` | Synthesis request timeout. |
| `NEWS_AGENT_URL` | `http://127.0.0.1:9001` | News Agent A2A base URL. |
| `FINANCIAL_AGENT_URL` | `http://127.0.0.1:9002` | Financial Agent A2A base URL. |
| `A2A_TIMEOUT_SECONDS` | `600` | Specialist A2A timeout. |
| `AUDIT_LOG_PATH` | `./logs/orchestrator-audit.jsonl` | Audit JSONL path. |
| `AUDIT_INCLUDE_QUERY` | `false` | Retain raw user queries in the audit record. |
| `MAX_QUERY_CHARS` | `32768` | Maximum accepted user-turn length. |
| `RELEASE_CHECKS_ENABLED` | `true` | Enable mechanical release gates and fail-closed audit persistence. |
| `POLICY_CHECKS_ENABLED` | `true` | Enable strict deterministic plan validation. |
| `EVIDENCE_CHECKS_ENABLED` | `true` | Require specialist evidence limits/citations and synthesis citation provenance. |

Optional external routing uses the existing flags and corresponding endpoint settings:

```text
USE_EXTERNAL_AI
USE_EXTERNAL_ORCH_AI
USE_EXTERNAL_LLM_AI
USE_EXTERNAL_OPENAI
EXTERNAL_ORCH_URL / API_KEY / MODEL / MAX_TOKENS
EXTERNAL_LLM_URL / API_KEY / MODEL / MAX_TOKENS
OPENAI_API_KEY / OPENAI_MODEL
```

`load_settings()` loads `.env` before evaluating these flags. The orchestration-round,
specialist-call, clarification, and retry limits remain fixed regardless of environment
values.

## Tests

Install development dependencies and run:

```bash
python -m pip install -r requirements-dev.txt
make check
```

The focused test suite covers:

- Current stock-price routing with and without a user-supplied ticker.
- Same-thread combined-to-financial follow-up routing based only on the newest user turn.
- Recovery from a failed stored A2A context continuation by retrying one fresh task.
- Latest financial-result routing to current-data and filing evidence paths.
- Historical financial routing to filing context without an unnecessary current-data
  directive.
- Combined answer release with citation provenance preserved.
- Synthesis failure fallback to separately labeled audited specialist sections.
- Citation-validation retry bounded by `max_specialist_calls`.
- `.env` loading before external orchestration/synthesis model selection.
- Audit redaction and synthesis metadata for a combined request.

## Source layout

```text
.
├── client.py
├── requirements.txt
├── requirements-dev.txt
├── Makefile
├── .env.example
├── src
│   └── host_agent
│       ├── __main__.py                 # Flask POST /v1/chat/completions
│       ├── audit.py                    # Hash-chained JSONL audit
│       ├── config.py                   # Local/external model and A2A settings
│       ├── llm_client.py               # Nemotron control plane and Qwen synthesis
│       ├── models.py                   # Structured contracts
│       ├── policy_manager.py           # Deterministic assessment and validation
│       ├── remote_agent_connection.py  # A2A 1.1.2 specialist client
│       └── routing_agent.py            # Bounded execution and response composition
└── tests
```

See `IMPLEMENTATION_NOTES.md` for the exact execution path and validation scope.
