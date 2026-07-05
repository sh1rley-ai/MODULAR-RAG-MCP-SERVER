"""LLM abstraction base class and unified response object.

All chat-capable LLM providers (OpenAI/Azure/Ollama/DeepSeek, added in B7)
implement `BaseLLM` so upper layers depend only on this contract and the
provider can be swapped via `config/settings.yaml` (llm.provider).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

# A chat message follows the OpenAI-style shape: {"role": ..., "content": ...}
Message = Dict[str, str]

VALID_ROLES = ("system", "user", "assistant")


class LLMError(RuntimeError):
    """Raised when an LLM call fails or its input is invalid."""


@dataclass(frozen=True)
class ChatResponse:
    """Unified response object returned by all LLM providers."""

    content: str
    model: str = ""
    usage: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"content": self.content, "model": self.model, "usage": dict(self.usage)}


class BaseLLM(ABC):
    """Abstract base class for chat-capable LLM providers.

    Concrete providers are constructed by `LLMFactory` and receive the
    `LLMSettings` section from the global settings as their first argument.
    """

    @abstractmethod
    def chat(self, messages: List[Message]) -> ChatResponse:
        """Send a list of chat messages and return the model response."""

    @staticmethod
    def validate_messages(messages: Any) -> None:
        """Validate the message list shape; raise LLMError with a readable message."""

        if not isinstance(messages, list) or not messages:
            raise LLMError("messages must be a non-empty list of {'role', 'content'} dicts")
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise LLMError(f"messages[{index}] must be a dict, got {type(message).__name__}")
            role = message.get("role")
            if role not in VALID_ROLES:
                raise LLMError(
                    f"messages[{index}].role must be one of {VALID_ROLES}, got {role!r}"
                )
            content = message.get("content")
            if not isinstance(content, str):
                raise LLMError(f"messages[{index}].content must be a string, got {content!r}")
