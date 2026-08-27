"""OpenAI-compatible clients for Nemotron orchestration and Qwen synthesis."""
from __future__ import annotations

from typing import Any, Protocol

from openai import AsyncOpenAI

from ..common.config import Settings
from ..common.logging import get_logger

LOGGER = get_logger(__name__)


class ModelGateway(Protocol):
    async def orchestrator_completion(self, messages: list[dict[str, str]]) -> str: ...
    async def generator_completion(self, messages: list[dict[str, str]]) -> str: ...


class LocalModelGateway:
    """Use the two externally served models without changing ``llm_service.py``."""

    def __init__(self, settings: Settings) -> None:

        self.settings = settings
        self._orchestrator = AsyncOpenAI(
            base_url=settings.orch_url,
            api_key=settings.orch_api_key or "not-needed",
            timeout=settings.orch_request_timeout,
            max_retries=0,
        )
        self._generator = AsyncOpenAI(
            base_url=settings.llm_url,
            api_key=settings.llm_api_key or "not-needed",
            timeout=settings.llm_request_timeout,
            max_retries=0,
        )

    @staticmethod
    def _content(response: Any) -> str:
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise RuntimeError("LLM service returned no choices")
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("LLM service returned an empty message")
        return content.strip()

    async def orchestrator_completion(self, messages: list[dict[str, str]]) -> str:
        LOGGER.debug("Calling Nemotron orchestrator at %s", self.settings.orch_url)
        response = await self._orchestrator.chat.completions.create(
            model=self.settings.orch_model,
            messages=messages,
            temperature=self.settings.orch_temperature,
            top_p=self.settings.orch_top_p,
            max_tokens=self.settings.orch_max_tokens,
        )
        return self._content(response)

    async def generator_completion(self, messages: list[dict[str, str]]) -> str:
        LOGGER.debug("Calling Qwen generator at %s", self.settings.llm_url)
        response = await self._generator.chat.completions.create(
            model=self.settings.llm_model,
            messages=messages,
            temperature=self.settings.llm_temperature,
            top_p=self.settings.llm_top_p,
            max_tokens=self.settings.llm_max_tokens,
        )
        return self._content(response)


__all__ = ["LocalModelGateway", "ModelGateway"]
