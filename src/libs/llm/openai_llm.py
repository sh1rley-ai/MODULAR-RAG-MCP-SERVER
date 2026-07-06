"""OpenAI-compatible LLM implementation (B7.1).

The OpenAI chat.completions protocol is the de-facto standard, so a single
`OpenAICompatibleLLM` covers OpenAI itself plus any endpoint speaking the same
protocol — DeepSeek and self-hosted gateways subclass it and only override the
provider name, API-key env var and default base URL.

Real HTTP goes through the official `openai` SDK; unit tests inject a fake
client via the `client` constructor argument, so no network is touched.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

from core.settings import LLMSettings

from libs.llm.base_llm import BaseLLM, ChatResponse, LLMError, Message
from libs.llm.llm_factory import LLMFactory

# Matches "${ENV_VAR}" references in settings values (e.g. api_key: "${OPENAI_API_KEY}")
_ENV_REF = re.compile(r"^\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)\}$")


def resolve_secret(value: Optional[str], env_var: str) -> Optional[str]:
    """Resolve a config value that may be a literal, a ${ENV_VAR} reference, or unset."""

    if value:
        match = _ENV_REF.match(value.strip())
        if match:
            return os.environ.get(match.group("name"))
        return value
    return os.environ.get(env_var)


class OpenAICompatibleLLM(BaseLLM):
    """Chat LLM backed by any OpenAI-compatible chat.completions endpoint."""

    provider_name = "openai"
    api_key_env = "OPENAI_API_KEY"
    default_base_url: Optional[str] = None

    def __init__(self, llm_settings: LLMSettings, client: Any = None) -> None:
        self.settings = llm_settings
        self._client = client if client is not None else self._build_client()

    def _build_client(self) -> Any:
        api_key = resolve_secret(self.settings.api_key, self.api_key_env)
        if not api_key:
            raise LLMError(
                f"[{self.provider_name}] missing API key: set llm.api_key in "
                f"config/settings.yaml or the {self.api_key_env} environment variable"
            )
        base_url = self.settings.base_url or self.default_base_url
        return OpenAI(api_key=api_key, base_url=base_url)

    @property
    def _model(self) -> str:
        return self.settings.model

    def chat(self, messages: List[Message]) -> ChatResponse:
        self.validate_messages(messages)
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens,
            )
        except Exception as exc:
            raise LLMError(
                f"[{self.provider_name}] chat call failed ({type(exc).__name__}): {exc}"
            ) from exc
        return self._to_chat_response(response)

    def _to_chat_response(self, response: Any) -> ChatResponse:
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise LLMError(f"[{self.provider_name}] response contained no choices")
        content = getattr(getattr(choices[0], "message", None), "content", None)
        if not isinstance(content, str):
            raise LLMError(f"[{self.provider_name}] response message content is missing")

        usage = getattr(response, "usage", None)
        usage_dict: Dict[str, Any] = {}
        if usage is not None:
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = getattr(usage, key, None)
                if value is not None:
                    usage_dict[key] = value

        model = getattr(response, "model", "") or self._model
        return ChatResponse(content=content, model=model, usage=usage_dict)


class OpenAILLM(OpenAICompatibleLLM):
    """OpenAI native API (api.openai.com), or a custom endpoint via llm.base_url."""


LLMFactory.register("openai", OpenAILLM)
