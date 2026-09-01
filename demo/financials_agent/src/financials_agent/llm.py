"""OpenAI-compatible clients for Nemotron planning and Qwen synthesis."""
from __future__ import annotations

import time
from typing import Any

from ..common.config import Settings
from ..common.logging import get_logger, log_payload

from .models import Evidence, FinancialPlan, ModelCallTrace
from .prompts import (
    PLANNER_SYSTEM_PROMPT,
    planner_user_prompt,
    release_verifier_messages,
    synthesis_messages,
)

LOGGER = get_logger(__name__)


class ModelGatewayError(RuntimeError):
    """Model endpoint failure paired with a data-minimized trace record."""

    def __init__(self, message: str, trace: ModelCallTrace):
        super().__init__(message)
        self.trace = trace


class OpenAIModelGateway:
    """Call the two local OpenAI-compatible model endpoints."""

    def __init__(self, settings: Settings):
        try:
            from openai import AsyncOpenAI
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "The openai package is required for local model endpoint calls."
            ) from exc

        self.settings = settings
        self._nemotron = AsyncOpenAI(
            base_url=settings.orch_url,
            api_key=settings.orch_api_key,
        )
        self._qwen = AsyncOpenAI(
            base_url=settings.llm_url,
            api_key=settings.llm_api_key,
        )

    async def _complete(
        self,
        *,
        role: str,
        client: Any,
        endpoint: str,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, ModelCallTrace]:
        started = time.perf_counter()
        LOGGER.debug(
            "Model request started role=%s model=%s endpoint=%s",
            role,
            model,
            endpoint,
        )
        log_payload(
            LOGGER,
            f"Model request payload role={role}",
            {
                "role": role,
                "endpoint": endpoint,
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": self.settings.llm_top_p,
                "max_tokens": max_tokens,
            },
        )
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=self.settings.llm_top_p,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise ValueError("Model returned an empty response")

            usage: Any = response.usage
            trace = ModelCallTrace(
                role=role,
                model=model,
                endpoint=endpoint,
                latency_ms=(time.perf_counter() - started) * 1000,
                ok=True,
                prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            )
            LOGGER.info(
                "Model request completed role=%s ok=true latency_ms=%.2f prompt_tokens=%d completion_tokens=%d",
                role,
                trace.latency_ms,
                trace.prompt_tokens,
                trace.completion_tokens,
            )
            log_payload(
                LOGGER,
                f"Model response payload role={role}",
                {
                    "content": content.strip(),
                    "trace": trace.to_dict(),
                },
            )
            return content.strip(), trace
        except Exception as exc:
            trace = ModelCallTrace(
                role=role,
                model=model,
                endpoint=endpoint,
                latency_ms=(time.perf_counter() - started) * 1000,
                ok=False,
                error=str(exc),
            )
            LOGGER.exception(
                "Model request failed role=%s model=%s endpoint=%s latency_ms=%.2f",
                role,
                model,
                endpoint,
                trace.latency_ms,
            )
            log_payload(
                LOGGER,
                f"Model error payload role={role}",
                trace.to_dict(),
            )
            raise ModelGatewayError(str(exc), trace) from exc

    async def plan(
        self,
        question: str,
        baseline: FinancialPlan,
    ) -> tuple[str, ModelCallTrace]:
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": planner_user_prompt(question, baseline)},
        ]
        return await self._complete(
            role="planner",
            client=self._nemotron,
            endpoint=self.settings.orch_url,
            model=self.settings.orch_model,
            messages=messages,
            temperature=0.0,
            max_tokens=min(self.settings.orch_max_tokens, 2048),
        )

    async def synthesize(
        self,
        question: str,
        plan: FinancialPlan,
        evidence: list[Evidence],
        audit_id: str,
        retry_reasons: list[str] | None = None,
        previous_answer: str | None = None,
    ) -> tuple[str, ModelCallTrace]:
        return await self._complete(
            role="answer_retry" if retry_reasons else "answer",
            client=self._qwen,
            endpoint=self.settings.llm_url,
            model=self.settings.llm_model,
            messages=synthesis_messages(
                question,
                plan,
                evidence,
                audit_id,
                retry_reasons=retry_reasons,
                previous_answer=previous_answer,
            ),
            temperature=self.settings.llm_temperature,
            max_tokens=self.settings.llm_max_tokens,
        )

    async def verify(
        self,
        question: str,
        answer: str,
        evidence: list[Evidence],
    ) -> tuple[str, ModelCallTrace]:
        return await self._complete(
            role="release_verifier",
            client=self._nemotron,
            endpoint=self.settings.orch_url,
            model=self.settings.orch_model,
            messages=release_verifier_messages(question, answer, evidence),
            temperature=self.settings.orch_temperature,
            max_tokens=self.settings.orch_max_tokens,
        )


__all__ = ["ModelGatewayError", "OpenAIModelGateway"]
