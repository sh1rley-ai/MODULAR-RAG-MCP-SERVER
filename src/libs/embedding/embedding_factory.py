"""Embedding factory: routes `embedding.provider` from settings to a registered implementation.

New providers (B7.x) register themselves here via `EmbeddingFactory.register`, so
adding a backend requires no change to calling code — only configuration.
"""

from __future__ import annotations

from typing import Dict, Type

from core.settings import Settings

from libs.embedding.base_embedding import BaseEmbedding


class EmbeddingFactory:
    """Creates a `BaseEmbedding` instance based on `settings.embedding.provider`."""

    _registry: Dict[str, Type[BaseEmbedding]] = {}

    @classmethod
    def register(cls, provider: str, embedding_cls: Type[BaseEmbedding]) -> None:
        """Register an embedding implementation under a provider name (case-insensitive)."""

        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("provider name must be a non-empty string")
        if not (isinstance(embedding_cls, type) and issubclass(embedding_cls, BaseEmbedding)):
            raise TypeError(f"{embedding_cls!r} must be a subclass of BaseEmbedding")
        cls._registry[provider.strip().lower()] = embedding_cls

    @classmethod
    def unregister(cls, provider: str) -> None:
        """Remove a provider from the registry (no-op if absent)."""

        cls._registry.pop(provider.strip().lower(), None)

    @classmethod
    def registered_providers(cls) -> list[str]:
        return sorted(cls._registry)

    @classmethod
    def create(cls, settings: Settings) -> BaseEmbedding:
        """Create the embedding instance configured by `settings.embedding.provider`."""

        provider = settings.embedding.provider.strip().lower()
        embedding_cls = cls._registry.get(provider)
        if embedding_cls is None:
            registered = ", ".join(cls.registered_providers()) or "(none)"
            raise ValueError(
                f"Unknown embedding provider: '{provider}'. Registered providers: {registered}"
            )
        return embedding_cls(settings.embedding)
