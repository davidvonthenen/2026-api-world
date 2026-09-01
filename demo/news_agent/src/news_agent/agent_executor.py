"""A2A protocol 1.0 executor for the governed News Agent."""
from __future__ import annotations

from typing import Any

from a2a.helpers import new_task_from_user_message, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks.task_updater import TaskUpdater

from ..common.logging import get_logger
from .news_agent import NewsAgent

LOGGER = get_logger(__name__)


class NewsAgentExecutor(AgentExecutor):
    """Bridge the A2A task lifecycle to the governed News Agent workflow."""

    def __init__(self, agent: NewsAgent | None = None) -> None:
        self.agent = agent or NewsAgent()

    @staticmethod
    def _request_metadata(context: RequestContext) -> dict[str, Any]:
        metadata = dict(context.metadata or {})
        try:
            user = getattr(context.call_context, "user", None)
            user_name = getattr(user, "user_name", None)
            if user_name:
                metadata.setdefault("user_id", str(user_name))
        except Exception:
            pass
        return metadata

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if context.message is None:
            raise RuntimeError("No A2A message was provided")

        task = context.current_task or new_task_from_user_message(context.message)
        # A task lifecycle stream must publish the Task before any update event,
        # including when the caller is continuing an existing task.
        await event_queue.enqueue_event(task)

        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=task.id,
            context_id=task.context_id,
        )

        def status_message(text: str):
            return updater.new_agent_message(
                [new_text_part(text=text, media_type="text/plain")]
            )

        await updater.start_work(status_message("Routing request to the News Agent."))

        async def progress(message: str) -> None:
            await updater.start_work(status_message(message))

        try:
            result = await self.agent.ainvoke(
                context.get_user_input(),
                task.context_id,
                task_id=task.id,
                request_metadata=self._request_metadata(context),
                progress=progress,
            )
            artifact_metadata = {
                "domain": "technology_news",
                "audit_id": result["audit_id"],
                "request_id": result["request_id"],
                "status": result["status"],
                "verification_status": result["verification_status"],
                "evidence_count": result["evidence_count"],
            }
            await updater.add_artifact(
                parts=[new_text_part(text=result["content"], media_type="text/plain")],
                name="news_result",
                metadata=artifact_metadata,
                append=False,
                last_chunk=True,
            )
            await updater.complete(
                status_message(
                    "The governed News Agent response is available in news_result."
                )
            )
        except Exception:
            LOGGER.exception("News Agent A2A execution failed")
            await updater.failed(
                status_message(
                    "The News Agent failed closed before producing a governed response."
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task
        if task is None:
            raise RuntimeError("No active task is available to cancel")
        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=task.id,
            context_id=task.context_id,
        )
        await updater.cancel(
            updater.new_agent_message(
                [new_text_part(text="News Agent task canceled.", media_type="text/plain")]
            )
        )


__all__ = ["NewsAgentExecutor"]
