"""Embedding abstraction base class.

All embedding providers (OpenAI/Azure/Ollama, added in B7) implement
`BaseEmbedding` so upper layers depend only on this contract and the
provider can be swapped via `config/settings.yaml` (embedding.provider).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List


class EmbeddingError(RuntimeError):
    """Raised when an embedding call fails or its input is invalid."""


class BaseEmbedding(ABC):
    """Abstract base class for embedding providers.

    Concrete providers are constructed by `EmbeddingFactory` and receive the
    `EmbeddingSettings` section from the global settings as their first argument.
    """

    @abstractmethod
    def embed(self, texts: List[str], trace: Any | None = None) -> List[List[float]]:
        """Embed a batch of texts and return one vector per input text.

        The returned list preserves input order: `result[i]` is the vector
        for `texts[i]`. `trace` is an optional TraceContext (Phase F).
        """

    @staticmethod
    def validate_texts(texts: Any) -> None:
        """Validate the input batch shape; raise EmbeddingError with a readable message."""

        if not isinstance(texts, list) or not texts:
            raise EmbeddingError("texts must be a non-empty list of strings")
        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise EmbeddingError(
                    f"texts[{index}] must be a string, got {type(text).__name__}"
                )
