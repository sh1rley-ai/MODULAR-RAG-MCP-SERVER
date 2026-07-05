"""VectorStore factory: routes `vector_store.provider` from settings to a registered implementation.

New backends (B7.6) register themselves here via `VectorStoreFactory.register`, so
adding a backend requires no change to calling code — only configuration.
"""

from __future__ import annotations

from typing import Dict, Type

from core.settings import Settings

from libs.vector_store.base_vector_store import BaseVectorStore


class VectorStoreFactory:
    """Creates a `BaseVectorStore` instance based on `settings.vector_store.provider`."""

    _registry: Dict[str, Type[BaseVectorStore]] = {}

    @classmethod
    def register(cls, provider: str, store_cls: Type[BaseVectorStore]) -> None:
        """Register a vector store implementation under a provider name (case-insensitive)."""

        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("provider name must be a non-empty string")
        if not (isinstance(store_cls, type) and issubclass(store_cls, BaseVectorStore)):
            raise TypeError(f"{store_cls!r} must be a subclass of BaseVectorStore")
        cls._registry[provider.strip().lower()] = store_cls

    @classmethod
    def unregister(cls, provider: str) -> None:
        """Remove a provider from the registry (no-op if absent)."""

        cls._registry.pop(provider.strip().lower(), None)

    @classmethod
    def registered_providers(cls) -> list[str]:
        return sorted(cls._registry)

    @classmethod
    def create(cls, settings: Settings) -> BaseVectorStore:
        """Create the vector store instance configured by `settings.vector_store.provider`."""

        provider = settings.vector_store.provider.strip().lower()
        store_cls = cls._registry.get(provider)
        if store_cls is None:
            registered = ", ".join(cls.registered_providers()) or "(none)"
            raise ValueError(
                f"Unknown vector store provider: '{provider}'. Registered providers: {registered}"
            )
        return store_cls(settings.vector_store)
