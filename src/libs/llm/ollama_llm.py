"""Ollama LLM implementation (B7.2).

Talks to a locally running Ollama server via its native `/api/chat` HTTP API.
Unlike cloud providers there is no API key; only the base URL matters, resolved
as: llm.base_url → OLLAMA_BASE_URL env var → http://localhost:11434.

HTTP goes through `httpx`; unit tests inject an `httpx.Client` built on
`httpx.MockTransport`, so no network is touched.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import httpx

from core.settings import LLMSettings

from libs.llm.base_llm import BaseLLM, ChatResponse, LLMError, Message
from libs.llm.llm_factory import LLMFactory

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT_SECONDS = 60.0


class OllamaLLM(BaseLLM):
    """Chat LLM backed by a local Ollama server (native /api/chat protocol)."""

    provider_name = "ollama"
    base_url_env = "OLLAMA_BASE_URL"

    def __init__(self, llm_settings: LLMSettings, client: httpx.Client | None = None) -> None:
        self.settings = llm_settings
        self.base_url = (
            llm_settings.base_url or os.environ.get(self.base_url_env) or DEFAULT_BASE_URL
        ).rstrip("/")
        self._client = client if client is not None else httpx.Client(
            base_url=self.base_url, timeout=DEFAULT_TIMEOUT_SECONDS
        )

    def chat(self, messages: List[Message]) -> ChatResponse:
        self.validate_messages(messages)
        payload = {
            "model": self.settings.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.settings.temperature,
                "num_predict": self.settings.max_tokens,
            },
        }
        try:
            response = self._client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.TimeoutException as exc:
            raise LLMError(
                f"[{self.provider_name}] request to {self.base_url} timed out: {exc}"
            ) from exc
        except httpx.ConnectError as exc:
            raise LLMError(
                f"[{self.provider_name}] cannot connect to Ollama at {self.base_url} "
                f"— is `ollama serve` running? ({exc})"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise LLMError(
                f"[{self.provider_name}] HTTP {exc.response.status_code} from "
                f"{self.base_url}: {exc.response.text[:200]}"
            ) from exc
        except Exception as exc:
            raise LLMError(
                f"[{self.provider_name}] chat call failed ({type(exc).__name__}): {exc}"
            ) from exc
        return self._to_chat_response(data)

    def _to_chat_response(self, data: Any) -> ChatResponse:
        if not isinstance(data, dict):
            raise LLMError(f"[{self.provider_name}] unexpected response shape: {type(data).__name__}")
        message = data.get("message") or {}
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise LLMError(f"[{self.provider_name}] response message content is missing")

        usage: Dict[str, Any] = {}
        prompt_tokens = data.get("prompt_eval_count")
        completion_tokens = data.get("eval_count")
        if isinstance(prompt_tokens, int):
            usage["prompt_tokens"] = prompt_tokens
        if isinstance(completion_tokens, int):
            usage["completion_tokens"] = completion_tokens
        if usage:
            usage["total_tokens"] = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)

        model = data.get("model") or self.settings.model
        return ChatResponse(content=content, model=model, usage=usage)


LLMFactory.register("ollama", OllamaLLM)
