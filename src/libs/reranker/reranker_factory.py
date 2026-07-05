"""Reranker factory: routes `rerank.provider` from settings to a registered implementation.

New backends (B7.7/B7.8) register themselves here via `RerankerFactory.register`, so
adding a rerank backend requires no change to calling code — only configuration.

`NoneReranker` is registered under "none" at import time as the built-in default,
and is also returned whenever `rerank.enabled` is false, regardless of provider.
"""

from __future__ import annotations

from typing import Dict, Type

from core.settings import Settings

from libs.reranker.base_reranker import BaseReranker, NoneReranker


class RerankerFactory:
    """Creates a `BaseReranker` instance based on `settings.rerank`."""

    _registry: Dict[str, Type[BaseReranker]] = {}

    @classmethod
    def register(cls, backend: str, reranker_cls: Type[BaseReranker]) -> None:
        """Register a reranker implementation under a backend name (case-insensitive)."""

        if not isinstance(backend, str) or not backend.strip():
            raise ValueError("backend name must be a non-empty string")
        if not (isinstance(reranker_cls, type) and issubclass(reranker_cls, BaseReranker)):
            raise TypeError(f"{reranker_cls!r} must be a subclass of BaseReranker")
        cls._registry[backend.strip().lower()] = reranker_cls

    @classmethod
    def unregister(cls, backend: str) -> None:
        """Remove a backend from the registry (no-op if absent)."""

        cls._registry.pop(backend.strip().lower(), None)

    @classmethod
    def registered_backends(cls) -> list[str]:
        return sorted(cls._registry)

    @classmethod
    def create(cls, settings: Settings) -> BaseReranker:
        """Create the reranker instance configured by `settings.rerank`.

        When `rerank.enabled` is false the provider is ignored and a
        `NoneReranker` is returned, so callers never need to special-case
        the disabled state.
        """

        if not settings.rerank.enabled:
            return NoneReranker(settings.rerank)
        backend = settings.rerank.provider.strip().lower()
        reranker_cls = cls._registry.get(backend)
        if reranker_cls is None:
            registered = ", ".join(cls.registered_backends()) or "(none)"
            raise ValueError(
                f"Unknown reranker backend: '{backend}'. Registered backends: {registered}"
            )
        return reranker_cls(settings.rerank)


RerankerFactory.register("none", NoneReranker)
