"""A2A protocol 1.0 command-line client for the governed News Agent."""
from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.types import Role, SendMessageRequest, TaskState
from a2a.helpers import get_artifact_text, get_message_text, new_text_message

from .common.config import load_settings

@dataclass
class ResponseCollector:
    """Collect the final artifact while keeping status updates out of stdout."""

    artifacts: dict[str, str] = field(default_factory=dict)
    artifact_names: dict[str, str] = field(default_factory=dict)
    artifact_order: list[str] = field(default_factory=list)
    message_text: str = ""
    latest_status_text: str = ""
    latest_task_state: str = ""

    def record_artifact(
        self,
        *,
        artifact_id: str | None,
        name: str | None,
        text: str,
        append: bool,
    ) -> None:
        if not text or not text.strip():
            return
        key = (artifact_id or name or f"artifact-{len(self.artifact_order) + 1}").strip()
        if key not in self.artifact_order:
            self.artifact_order.append(key)
        if name:
            self.artifact_names[key] = name
        if append and key in self.artifacts:
            self.artifacts[key] += text
        else:
            self.artifacts[key] = text

    def record_message(self, text: str) -> None:
        text = text.strip()
        if text:
            self.message_text = text

    def record_status(self, text: str) -> None:
        text = text.strip()
        if text:
            self.latest_status_text = text

    def record_task_state(self, state: object) -> None:
        """Normalize a protobuf task state so terminal failures are not success."""
        if state is None:
            return
        name = ""
        try:
            name = str(TaskState.Name(int(state)))
        except (ImportError, AttributeError, TypeError, ValueError):
            name = str(state)
        self.latest_task_state = name.strip().upper()

    def final_text(self) -> str:
        failed_states = (
            "TASK_STATE_FAILED",
            "TASK_STATE_CANCELED",
            "TASK_STATE_REJECTED",
        )
        if self.latest_task_state.endswith(failed_states):
            detail = self.latest_status_text or self.latest_task_state
            raise RuntimeError(
                f"A2A task ended in {self.latest_task_state}: {detail}"
            )
        for key in reversed(self.artifact_order):
            if self.artifact_names.get(key) == "news_result":
                text = self.artifacts.get(key, "").strip()
                if text:
                    return text
        for key in reversed(self.artifact_order):
            text = self.artifacts.get(key, "").strip()
            if text:
                return text
        if self.message_text:
            return self.message_text
        if self.latest_status_text:
            return self.latest_status_text
        raise RuntimeError("The A2A task completed without a text message or artifact")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    settings = load_settings()
    parser = argparse.ArgumentParser(
        description="Send one question to the governed News Agent over A2A 1.0"
    )
    parser.add_argument("--question", "-q", required=True)
    parser.add_argument(
        "--url",
        default=settings.a2a_app_url,
        help=f"News Agent base URL (default: {settings.a2a_app_url})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="HTTP connect/read/write timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use A2A streaming when the Agent Card supports it",
    )
    parser.add_argument(
        "--show-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write A2A task status updates to stderr",
    )
    return parser.parse_args(argv)


def _interface_value(interface: object, field_name: str) -> str:
    if isinstance(interface, dict):
        camel_case = field_name.split("_")[0] + "".join(
            part.title() for part in field_name.split("_")[1:]
        )
        value = interface.get(field_name, interface.get(camel_case))
    else:
        value = getattr(interface, field_name, None)
    return str(value or "").strip()


def _validate_agent_card(agent_card: object) -> None:
    interfaces = getattr(agent_card, "supported_interfaces", None)
    if interfaces is None and isinstance(agent_card, dict):
        interfaces = agent_card.get("supported_interfaces") or agent_card.get(
            "supportedInterfaces"
        )
    interfaces = list(interfaces or [])
    has_jsonrpc_v1 = any(
        _interface_value(interface, "protocol_binding").upper() == "JSONRPC"
        and _interface_value(interface, "protocol_version") == "1.0"
        for interface in interfaces
    )
    if not has_jsonrpc_v1:
        raise RuntimeError(
            "The Agent Card does not advertise an A2A 1.0 JSON-RPC interface"
        )


def _consume_stream_response(
    response: Any,
    collector: ResponseCollector,
    *,
    show_progress: bool,
) -> None:
    if response.HasField("artifact_update"):
        update = response.artifact_update
        artifact = update.artifact
        collector.record_artifact(
            artifact_id=str(artifact.artifact_id or ""),
            name=str(artifact.name or ""),
            text=get_artifact_text(artifact),
            append=bool(update.append),
        )
        return

    if response.HasField("task"):
        task = response.task
        for artifact in task.artifacts:
            collector.record_artifact(
                artifact_id=str(artifact.artifact_id or ""),
                name=str(artifact.name or ""),
                text=get_artifact_text(artifact),
                append=False,
            )
        if task.HasField("status"):
            collector.record_task_state(task.status.state)
            if task.status.HasField("message"):
                text = get_message_text(task.status.message)
                collector.record_status(text)
                if show_progress and text:
                    print(f"[news-agent] {text}", file=sys.stderr)
        return

    if response.HasField("message"):
        collector.record_message(get_message_text(response.message))
        return

    if response.HasField("status_update"):
        status = response.status_update.status
        collector.record_task_state(status.state)
        if status.HasField("message"):
            text = get_message_text(status.message)
            collector.record_status(text)
            if show_progress and text:
                print(f"[news-agent] {text}", file=sys.stderr)


async def ask_news_agent(
    question: str,
    *,
    url: str,
    timeout: float,
    stream: bool,
    show_progress: bool,
) -> str:
    """Resolve the Agent Card, require A2A 1.0 JSON-RPC, and send a message."""
    if not question.strip():
        raise ValueError("question must not be empty")
    if timeout <= 0:
        raise ValueError("timeout must be greater than 0")
    base_url = url.rstrip("/")
    httpx_client = httpx.AsyncClient(timeout=httpx.Timeout(timeout))
    client: Any | None = None
    try:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
        agent_card = await resolver.get_agent_card()
        _validate_agent_card(agent_card)

        client_config = ClientConfig(
            streaming=stream,
            httpx_client=httpx_client,
            supported_protocol_bindings=["JSONRPC"],
            accepted_output_modes=["text/plain"],
        )
        client = await create_client(agent=agent_card, client_config=client_config)
        request = SendMessageRequest(
            message=new_text_message(question.strip(), role=Role.ROLE_USER)
        )

        collector = ResponseCollector()
        async for response in client.send_message(request):
            _consume_stream_response(
                response,
                collector,
                show_progress=show_progress,
            )
        return collector.final_text()
    finally:
        if client is not None:
            await client.close()
        else:
            await httpx_client.aclose()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        answer = asyncio.run(
            ask_news_agent(
                args.question,
                url=args.url,
                timeout=args.timeout,
                stream=args.stream,
                show_progress=args.show_progress,
            )
        )
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        raise SystemExit(f"News Agent query failed: {exc}") from exc
    print(answer)


if __name__ == "__main__":
    main()
