"""A2A 1.0 task executor for the governed Financials Agent."""
from __future__ import annotations

from a2a.helpers import (
    get_message_text,
    new_task_from_user_message,
    new_text_artifact_update_event,
    new_text_status_update_event,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import TaskState

from ..common.logging import get_logger, log_payload
from .financials_agent import FinancialsAgent

LOGGER = get_logger(__name__)


class FinancialsExecutor(AgentExecutor):
    """Bridge A2A messages to the governed FinancialsAgent execution pipeline."""

    def __init__(self, agent: FinancialsAgent | None = None) -> None:
        super().__init__()
        self.agent = agent or FinancialsAgent()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if not context.message:
            raise RuntimeError("No A2A message was provided")

        task = context.current_task or new_task_from_user_message(context.message)
        await event_queue.enqueue_event(task)
        await event_queue.enqueue_event(
            new_text_status_update_event(
                task_id=task.id,
                context_id=task.context_id,
                state=TaskState.TASK_STATE_WORKING,
                text="Applying financial-domain policy and gathering approved evidence.",
            )
        )

        try:
            query = get_message_text(context.message).strip()
            LOGGER.debug(
                "A2A agent request started task_id=%s context_id=%s",
                task.id,
                task.context_id,
            )
            log_payload(
                LOGGER,
                "A2A agent input payload",
                {
                    "task_id": task.id,
                    "context_id": task.context_id,
                    "query": query,
                },
            )
            result = await self.agent.ainvoke(query, task.context_id)
            answer = str(result.get("content") or "I don't know.")
            metadata = result.get("metadata") or {}
            LOGGER.info(
                "A2A agent request completed task_id=%s outcome=%s audit_id=%s",
                task.id,
                metadata.get("outcome", "unknown"),
                metadata.get("audit_id", ""),
            )
            log_payload(
                LOGGER,
                "A2A agent output payload",
                result,
            )

            await event_queue.enqueue_event(
                new_text_artifact_update_event(
                    task_id=task.id,
                    context_id=task.context_id,
                    name="financial_analysis",
                    text=answer,
                    last_chunk=True,
                )
            )

            state = (
                TaskState.TASK_STATE_INPUT_REQUIRED
                if result.get("require_user_input")
                else TaskState.TASK_STATE_COMPLETED
            )
            terminal_text = (
                "Additional company identity information is required."
                if result.get("require_user_input")
                else "Financial analysis completed."
            )
            await event_queue.enqueue_event(
                new_text_status_update_event(
                    task_id=task.id,
                    context_id=task.context_id,
                    state=state,
                    text=terminal_text,
                )
            )
        except Exception:
            LOGGER.exception("Financial Agent A2A task failed task_id=%s", task.id)
            await event_queue.enqueue_event(
                new_text_status_update_event(
                    task_id=task.id,
                    context_id=task.context_id,
                    state=TaskState.TASK_STATE_FAILED,
                    text=(
                        "The Financial Agent could not produce a governed answer. "
                        "No unverified financial result was released."
                    ),
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task
        if task is None:
            raise RuntimeError("No active task is available to cancel")
        await event_queue.enqueue_event(
            new_text_status_update_event(
                task_id=task.id,
                context_id=task.context_id,
                state=TaskState.TASK_STATE_CANCELED,
                text="Financial analysis canceled before release.",
            )
        )
