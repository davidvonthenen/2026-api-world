"""OpenAI-compatible HTTP entry point for the Orchestrator Agent."""

from __future__ import annotations

import asyncio
import atexit
import hashlib
import logging
import time
import uuid
from typing import Literal

from flask import Flask, request, jsonify, make_response, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .config import Settings, load_settings
from .routing_agent import RoutingAgent, create_routing_agent

LOGGER = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = "vertical-api-orchestrator"
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    user: str | None = None


def _openai_error(
    message: str,
    *,
    status_code: int,
    error_type: str = "invalid_request_error",
    param: str | None = None,
    code: str | None = None,
) -> Response:
    """Constructs an OpenAI-compatible error response."""
    response = jsonify({
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": code,
        }
    })
    return make_response(response, status_code)


def _session_id(req: ChatCompletionRequest) -> str:
    if req.user and req.user.strip():
        return req.user.strip()[:200]
    transcript = "\n".join(
        f"{message.role}:{message.content}" for message in req.messages
    )
    return hashlib.sha256(transcript.encode("utf-8")).hexdigest()


def create_app(
    settings: Settings | None = None,
    routing_agent: RoutingAgent | None = None,
) -> Flask:
    """Create a Flask application exposing POST /v1/chat/completions."""
    settings = settings or load_settings()
    agent = routing_agent or create_routing_agent(settings)

    app = Flask(__name__)

    def cleanup_agent() -> None:
        """Ensure the routing agent closes gracefully upon termination."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(agent.close())
            else:
                loop.run_until_complete(agent.close())
        except Exception as e:
            LOGGER.error("Failed to cleanly close the agent: %s", e)

    atexit.register(cleanup_agent)

    @app.route("/v1/chat/completions", methods=["POST"])
    async def chat_completions():
        try:
            payload = request.get_json()
            if payload is None:
                return _openai_error("Missing JSON payload.", status_code=400)
            req_obj = ChatCompletionRequest(**payload)
        except ValidationError as e:
            return _openai_error(
                str(e),
                status_code=400,
                param="request_body",
                code="validation_error",
            )

        if req_obj.stream:
            return _openai_error(
                "Streaming is not implemented by this conference demo.",
                status_code=400,
                param="stream",
                code="streaming_not_supported",
            )

        if not any(
            message.role == "user" and message.content.strip()
            for message in req_obj.messages
        ):
            return _openai_error(
                "messages must contain at least one non-empty user message.",
                status_code=400,
                param="messages",
                code="missing_user_message",
            )

        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        try:
            result = await agent.handle_messages(
                [message.model_dump() for message in req_obj.messages],
                session_id=_session_id(req_obj),
                request_id=request_id,
            )
        except Exception as exc:
            LOGGER.exception("orchestrator_request_failed request_id=%s", request_id)
            return _openai_error(
                "The orchestrator could not complete the request.",
                status_code=500,
                error_type="server_error",
                code="orchestrator_failure",
            )

        return jsonify({
            "id": result.request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req_obj.model or settings.public_model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result.content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        })

    return app


def main() -> None:
    settings = load_settings()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    LOGGER.info(
        "Starting Orchestrator Agent on %s:%d",
        settings.api_host,
        settings.api_port,
    )
    app = create_app(settings)
    
    # Binds the application to the provided host and port configurations
    app.run(
        host=settings.api_host,
        port=settings.api_port,
    )


if __name__ == "__main__":
    main()
