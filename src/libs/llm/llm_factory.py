"""LLM factory: routes `llm.provider` from settings to a registered implementation.

New providers (B7.x) register themselves here via `LLMFactory.register`, so
adding a backend requires no change to calling code — only configuration.
"""

from __future__ import annotations

from typing import Dict, Type

from core.settings import Settings

from libs.llm.base_llm import BaseLLM


class LLMFactory:
    """Creates a `BaseLLM` instance based on `settings.llm.provider`."""

    _registry: Dict[str, Type[BaseLLM]] = {}

    @classmethod
    def register(cls, provider: str, llm_cls: Type[BaseLLM]) -> None:
        """Register an LLM implementation under a provider name (case-insensitive)."""

        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("provider name must be a non-empty string")
        if not (isinstance(llm_cls, type) and issubclass(llm_cls, BaseLLM)):
            raise TypeError(f"{llm_cls!r} must be a subclass of BaseLLM")
        cls._registry[provider.strip().lower()] = llm_cls

    @classmethod
    def unregister(cls, provider: str) -> None:
        """Remove a provider from the registry (no-op if absent)."""

        cls._registry.pop(provider.strip().lower(), None)

    @classmethod
    def registered_providers(cls) -> list[str]:
        return sorted(cls._registry)

    @classmethod
    def create(cls, settings: Settings) -> BaseLLM:
        """Create the LLM instance configured by `settings.llm.provider`."""

        provider = settings.llm.provider.strip().lower()
        llm_cls = cls._registry.get(provider)
        if llm_cls is None:
            registered = ", ".join(cls.registered_providers()) or "(none)"
            raise ValueError(
                f"Unknown LLM provider: '{provider}'. Registered providers: {registered}"
            )
        return llm_cls(settings.llm)
