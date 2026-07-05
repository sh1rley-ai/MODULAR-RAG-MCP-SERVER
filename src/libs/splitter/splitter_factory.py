"""Splitter factory: routes `ingestion.splitter` from settings to a registered implementation.

New strategies (B7.5+) register themselves here via `SplitterFactory.register`, so
adding a splitting strategy requires no change to calling code — only configuration.
"""

from __future__ import annotations

from typing import Dict, Type

from core.settings import Settings

from libs.splitter.base_splitter import BaseSplitter


class SplitterFactory:
    """Creates a `BaseSplitter` instance based on `settings.ingestion.splitter`."""

    _registry: Dict[str, Type[BaseSplitter]] = {}

    @classmethod
    def register(cls, strategy: str, splitter_cls: Type[BaseSplitter]) -> None:
        """Register a splitter implementation under a strategy name (case-insensitive)."""

        if not isinstance(strategy, str) or not strategy.strip():
            raise ValueError("strategy name must be a non-empty string")
        if not (isinstance(splitter_cls, type) and issubclass(splitter_cls, BaseSplitter)):
            raise TypeError(f"{splitter_cls!r} must be a subclass of BaseSplitter")
        cls._registry[strategy.strip().lower()] = splitter_cls

    @classmethod
    def unregister(cls, strategy: str) -> None:
        """Remove a strategy from the registry (no-op if absent)."""

        cls._registry.pop(strategy.strip().lower(), None)

    @classmethod
    def registered_strategies(cls) -> list[str]:
        return sorted(cls._registry)

    @classmethod
    def create(cls, settings: Settings) -> BaseSplitter:
        """Create the splitter instance configured by `settings.ingestion.splitter`."""

        if settings.ingestion is None:
            raise ValueError(
                "settings.ingestion is missing: the 'ingestion' section "
                "(chunk_size/chunk_overlap/splitter/batch_size) is required to create a splitter"
            )
        strategy = settings.ingestion.splitter.strip().lower()
        splitter_cls = cls._registry.get(strategy)
        if splitter_cls is None:
            registered = ", ".join(cls.registered_strategies()) or "(none)"
            raise ValueError(
                f"Unknown splitter strategy: '{strategy}'. Registered strategies: {registered}"
            )
        return splitter_cls(settings.ingestion)
