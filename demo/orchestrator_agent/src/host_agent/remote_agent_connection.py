"""A2A 1.1.2 client wrapper for the external vertical specialist agents."""

from __future__ import annotations

import hashlib
import re
import time

from collections.abc import Sequence

import httpx

from a2a.client import Client, ClientCallContext, ClientConfig, create_client
from a2a.helpers import get_artifact_text, get_message_text, new_text_message
from a2a.types import Role, SendMessageRequest, TaskState

from .models import AgentName, SpecialistResponse


TERMINAL_STATES = {
    "TASK_STATE_COMPLETED",
    "TASK_STATE_FAILED",
    "TASK_STATE_CANCELED",
    "TASK_STATE_REJECTED",
    "TASK_STATE_INPUT_REQUIRED",
    "TASK_STATE_AUTH_REQUIRED",
}

PRIVATE_COMPANY_MARKERS = (
    "not verified as a publicly traded company",
    "may be private, unlisted",
    "no ticker was inferred",
    "did not return approved evidence",
    "could not verify a matching public security",
    "no public security",
)

_CITATION_GROUP_RE = re.compile(
    r"\[((?:[A-Z][A-Z0-9_-]*\d+)(?:\s*,\s*[A-Z][A-Z0-9_-]*\d+)*)\]"
)


def _deduplicate_text(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = value.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


def _extract_citations(text: str) -> list[str]:
    references: list[str] = []
    for group in _CITATION_GROUP_RE.findall(text):
        references.extend(
            f"[{part.strip()}]" for part in group.split(",") if part.strip()
        )
    references.extend(
        match.rstrip(".,);]")
        for match in re.findall(r"https?://[^\s<>()]+", text)
    )
    return _deduplicate_text(references)


def _extract_audit_id(text: str) -> str:
    match = re.search(
        r"(?:\*\*)?Audit ID(?:\*\*)?\s*:\s*`?([A-Za-z0-9-]+)`?",
        text,
        flags=re.IGNORECASE,
    )
    return match.group(1) if match else ""


def _extract_as_of(text: str) -> str:
    match = re.search(r"\bas of\s+([^;\n]+)", text, flags=re.IGNORECASE)
    return match.group(1).strip(" .") if match else ""


def _summary(text: str) -> str:
    for block in re.split(r"\n\s*\n", text):
        cleaned = re.sub(r"^#+\s*", "", block.strip())
        if cleaned and not cleaned.lower().startswith("audit id"):
            return cleaned[:1200]
    return text.strip()[:1200]


def is_private_company_result(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in PRIVATE_COMPANY_MARKERS)


def normalize_specialist_response(
    *,
    agent: AgentName,
    status: str,
    raw_text: str,
    context_id: str,
    task_id: str,
    latency_ms: float,
    warnings: Sequence[str] = (),
) -> SpecialistResponse:
    """Wrap existing specialist text in the demonstration response schema."""

    clean_text = raw_text.strip()
    all_warnings = list(warnings)
    citations = _extract_citations(clean_text)
    if status == "failed":
        all_warnings.append(f"The {agent} specialist call did not complete.")
    if clean_text and not citations:
        all_warnings.append(
            "The specialist response contained no machine-detectable citation IDs or URLs."
        )
    if not clean_text:
        all_warnings.append("The specialist returned no text result.")

    return SpecialistResponse(
        agent=agent,
        status=status,
        summary=_summary(clean_text),
        facts=[],
        data={
            "audit_id": _extract_audit_id(clean_text),
            "private_or_unlisted": (
                agent == "financial" and is_private_company_result(clean_text)
            ),
            "output_sha256": hashlib.sha256(clean_text.encode("utf-8")).hexdigest(),
            "output_length": len(clean_text),
        },
        citations=citations,
        as_of=_extract_as_of(clean_text),
        warnings=_deduplicate_text(all_warnings),
        raw_text=clean_text,
        context_id=context_id,
        task_id=task_id,
        latency_ms=round(latency_ms, 3),
    )


class RemoteAgentConnection:
    """Discover and call one external A2A specialist with request-local clients."""

    def __init__(
        self,
        *,
        agent: AgentName,
        agent_name: str,
        agent_url: str,
        timeout_seconds: float,
    ) -> None:
        self.agent = agent
        self.agent_name = agent_name
        self.agent_url = agent_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _http_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=httpx.Timeout(
                self.timeout_seconds,
                connect=min(30.0, self.timeout_seconds),
            )
        )

    async def _send_once(
        self,
        task: str,
        *,
        context_id: str | None,
    ) -> SpecialistResponse:
        started = time.perf_counter()
        artifact_texts: list[str] = []
        status_texts: list[str] = []
        message_texts: list[str] = []
        task_texts: list[str] = []
        returned_context_id = context_id or ""
        returned_task_id = ""
        terminal_state = ""
        http_client = self._http_client()
        client: Client | None = None

        try:
            config = ClientConfig(
                streaming=True,
                polling=False,
                httpx_client=http_client,
                supported_protocol_bindings=["JSONRPC"],
                accepted_output_modes=["text/plain"],
            )
            client = await create_client(
                self.agent_url,
                client_config=config,
            )
            message = new_text_message(
                task,
                media_type="text/plain",
                context_id=context_id,
                role=Role.ROLE_USER,
            )
            request = SendMessageRequest(message=message)
            stream = client.send_message(
                request,
                context=ClientCallContext(timeout=self.timeout_seconds),
            )

            async for event in stream:
                if event.HasField("task"):
                    returned_task_id = event.task.id or returned_task_id
                    returned_context_id = event.task.context_id or returned_context_id
                    task_texts.extend(
                        get_artifact_text(artifact)
                        for artifact in event.task.artifacts
                        if get_artifact_text(artifact).strip()
                    )
                    if event.task.HasField("status"):
                        terminal_state = TaskState.Name(event.task.status.state)
                elif event.HasField("artifact_update"):
                    returned_task_id = (
                        event.artifact_update.task_id or returned_task_id
                    )
                    returned_context_id = (
                        event.artifact_update.context_id or returned_context_id
                    )
                    artifact_texts.append(
                        get_artifact_text(event.artifact_update.artifact)
                    )
                elif event.HasField("status_update"):
                    returned_task_id = event.status_update.task_id or returned_task_id
                    returned_context_id = (
                        event.status_update.context_id or returned_context_id
                    )
                    state_name = TaskState.Name(event.status_update.status.state)
                    if state_name in TERMINAL_STATES:
                        terminal_state = state_name
                    if event.status_update.status.HasField("message"):
                        status_texts.append(
                            get_message_text(event.status_update.status.message)
                        )
                elif event.HasField("message"):
                    returned_task_id = event.message.task_id or returned_task_id
                    returned_context_id = event.message.context_id or returned_context_id
                    message_texts.append(get_message_text(event.message))

            preferred = _deduplicate_text([*artifact_texts, *task_texts])
            fallback = _deduplicate_text([*status_texts, *message_texts])
            candidates = preferred or fallback
            raw_text = max(candidates, key=len) if candidates else ""

            if terminal_state == "TASK_STATE_INPUT_REQUIRED":
                status = "input_required"
            elif terminal_state in {
                "TASK_STATE_FAILED",
                "TASK_STATE_CANCELED",
                "TASK_STATE_REJECTED",
                "TASK_STATE_AUTH_REQUIRED",
            }:
                status = "failed"
            elif raw_text:
                status = "completed"
            else:
                status = "failed"

            return normalize_specialist_response(
                agent=self.agent,
                status=status,
                raw_text=raw_text,
                context_id=returned_context_id,
                task_id=returned_task_id,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as exc:
            return normalize_specialist_response(
                agent=self.agent,
                status="failed",
                raw_text="",
                context_id=returned_context_id,
                task_id=returned_task_id,
                latency_ms=(time.perf_counter() - started) * 1000,
                warnings=[
                    f"A2A communication with {self.agent_name} failed: "
                    f"{type(exc).__name__}: {exc}"
                ],
            )
        finally:
            if client is not None:
                try:
                    await client.close()
                except Exception:
                    pass
            if not http_client.is_closed:
                try:
                    await http_client.aclose()
                except Exception:
                    pass

    async def send_message(
        self,
        task: str,
        *,
        context_id: str | None = None,
    ) -> SpecialistResponse:
        """Send one bounded specialist request and consume its A2A task stream."""

        started = time.perf_counter()
        result = await self._send_once(task, context_id=context_id)
        communication_failure = (
            result.status == "failed"
            and not result.raw_text.strip()
            and any(
                warning.startswith("A2A communication with ")
                for warning in result.warnings
            )
        )
        if not context_id or not communication_failure:
            return result

        retry = await self._send_once(task, context_id=None)
        retry_warning = (
            "Stored A2A context continuation failed; the specialist call was retried "
            "as a new task. "
            + next(
                (
                    warning
                    for warning in result.warnings
                    if warning.startswith("A2A communication with ")
                ),
                "",
            )
        ).strip()
        return retry.model_copy(
            update={
                "warnings": _deduplicate_text(
                    [retry_warning, *retry.warnings]
                ),
                "latency_ms": round(
                    (time.perf_counter() - started) * 1000,
                    3,
                ),
            }
        )

    async def close(self) -> None:
        # Network clients are request-local so they are never reused across Flask
        # request event loops.
        return None


__all__ = [
    "RemoteAgentConnection",
    "is_private_company_result",
    "normalize_specialist_response",
]
