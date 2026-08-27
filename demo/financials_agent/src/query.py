"""A2A 1.0 CLI client for the governed Financials Agent."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections.abc import Sequence
from pathlib import Path
import sys
from typing import Any

import httpx

from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.helpers import get_artifact_text, get_message_text, new_text_message
from a2a.types import Role, SendMessageRequest, TaskState

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from common.config import load_settings
from common.logging import get_logger, log_payload

LOGGER = get_logger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask the Financials Agent one question over A2A 1.0"
    )
    parser.add_argument("--question", "-q", required=True)
    return parser.parse_args(argv)


def _validate_agent_card(card: Any) -> None:
    interfaces = list(card.supported_interfaces)
    jsonrpc_interfaces = [
        interface
        for interface in interfaces
        if str(interface.protocol_binding).upper() == "JSONRPC"
    ]
    if len(interfaces) != 1 or len(jsonrpc_interfaces) != 1:
        raise RuntimeError(
            "Financial Agent must advertise exactly one JSON-RPC interface."
        )
    if jsonrpc_interfaces[0].protocol_version != "1.0":
        raise RuntimeError(
            "Financial Agent did not advertise the required A2A protocol version 1.0."
        )


async def query_agent(
    question: str,
    agent_url: str,
    *,
    httpx_client: httpx.AsyncClient | None = None,
) -> tuple[str, dict[str, str]]:
    """Resolve the current agent card, enforce A2A 1.0, and send one message."""
    clean_question = question.strip()
    if not clean_question:
        raise ValueError("Question cannot be empty")

    owns_httpx_client = httpx_client is None
    if httpx_client is None:
        httpx_client = httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, connect=10.0),
        )

    a2a_client: Any | None = None
    try:
        LOGGER.info("A2A client resolving agent card url=%s", agent_url.rstrip("/"))
        log_payload(
            LOGGER,
            "A2A client request payload",
            {"agent_url": agent_url.rstrip("/"), "question": clean_question},
        )
        resolver = A2ACardResolver(httpx_client, agent_url.rstrip("/"))
        card = await resolver.get_agent_card()
        _validate_agent_card(card)
        log_payload(
            LOGGER,
            "A2A client agent-card payload",
            card,
        )

        config = ClientConfig(
            streaming=True,
            httpx_client=httpx_client,
            supported_protocol_bindings=["JSONRPC"],
            use_client_preference=True,
            accepted_output_modes=["text/plain"],
        )
        a2a_client = await create_client(card, client_config=config)
        request = SendMessageRequest(
            message=new_text_message(clean_question, role=Role.ROLE_USER)
        )

        artifacts: dict[str, str] = {}
        artifact_order: list[str] = []
        message_answer = ""
        status_text = ""
        task_id = ""
        terminal_state = ""

        async for event in a2a_client.send_message(request):
            if event.HasField("message"):
                message_answer = get_message_text(event.message).strip()
                continue

            if event.HasField("task"):
                task_id = event.task.id
                for artifact in event.task.artifacts:
                    key = artifact.artifact_id or artifact.name
                    if key not in artifacts:
                        artifact_order.append(key)
                    artifacts[key] = get_artifact_text(artifact).strip()
                state_name = TaskState.Name(event.task.status.state)
                if state_name != "TASK_STATE_UNSPECIFIED":
                    terminal_state = state_name
                if event.task.status.HasField("message"):
                    status_text = get_message_text(event.task.status.message).strip()
                continue

            if event.HasField("artifact_update"):
                update = event.artifact_update
                artifact = update.artifact
                key = artifact.artifact_id or artifact.name
                text = get_artifact_text(artifact)
                if key not in artifacts:
                    artifact_order.append(key)
                artifacts[key] = (
                    artifacts.get(key, "") + text if update.append else text
                )
                continue

            if event.HasField("status_update"):
                update = event.status_update
                task_id = task_id or update.task_id
                terminal_state = TaskState.Name(update.status.state)
                if update.status.HasField("message"):
                    status_text = get_message_text(update.status.message).strip()

        failure_states = {
            "TASK_STATE_FAILED",
            "TASK_STATE_CANCELED",
            "TASK_STATE_REJECTED",
        }
        if terminal_state in failure_states:
            detail = status_text or terminal_state
            raise RuntimeError(f"Financial Agent task failed: {detail}")

        artifact_answer = "\n".join(
            artifacts[key].strip()
            for key in artifact_order
            if artifacts.get(key, "").strip()
        ).strip()
        answer = artifact_answer or message_answer or status_text
        if not answer:
            raise RuntimeError("Financial Agent returned no text artifact or message")

        trace = {
            "agent": card.name,
            "protocol_binding": "JSONRPC",
            "protocol_version": "1.0",
            "task_id": task_id,
            "terminal_state": terminal_state,
        }
        LOGGER.info(
            "A2A client request completed agent=%s task_id=%s state=%s",
            card.name,
            task_id,
            terminal_state,
        )
        log_payload(
            LOGGER,
            "A2A client response payload",
            {"answer": answer, "trace": trace},
        )
        return answer, trace
    finally:
        if a2a_client is not None:
            await a2a_client.close()
        if owns_httpx_client and not httpx_client.is_closed:
            await httpx_client.aclose()


async def _run(question: str) -> None:
    settings = load_settings()
    answer, trace = await query_agent(question, settings.financials_agent_url)
    print(answer)
    if os.getenv("SHOW_TRACE", "false").strip().lower() in {"1", "true", "yes", "on"}:
        print("\n--- A2A client trace ---")
        print(json.dumps(trace, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        asyncio.run(_run(args.question))
    except Exception as exc:
        print(f"Financial Agent query failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
