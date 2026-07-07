"""LLM factory: routes `llm.provider` from settings to a registered implementation.

New providers (B7.x) register themselves here via `LLMFactory.register`, so
adding a backend requires no change to calling code — only configuration.

Vision LLM providers register via `LLMFactory.register_vision` and are created
by `LLMFactory.create_vision_llm`. Both registries are independent so text-only
and vision-capable providers can share the same provider name if desired.
"""

from __future__ import annotations

from typing import Dict, Type

from core.settings import Settings

from libs.llm.base_llm import BaseLLM
from libs.llm.base_vision_llm import BaseVisionLLM


class LLMFactory:
    """Creates LLM instances based on provider settings."""

    _registry: Dict[str, Type[BaseLLM]] = {}
    _vision_registry: Dict[str, Type[BaseVisionLLM]] = {}

    # ------------------------------------------------------------------
    # Text LLM registry
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Vision LLM registry
    # ------------------------------------------------------------------

    @classmethod
    def register_vision(cls, provider: str, vision_cls: Type[BaseVisionLLM]) -> None:
        """Register a Vision LLM implementation under a provider name (case-insensitive)."""

        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("provider name must be a non-empty string")
        if not (isinstance(vision_cls, type) and issubclass(vision_cls, BaseVisionLLM)):
            raise TypeError(f"{vision_cls!r} must be a subclass of BaseVisionLLM")
        cls._vision_registry[provider.strip().lower()] = vision_cls

    @classmethod
    def unregister_vision(cls, provider: str) -> None:
        """Remove a vision provider from the registry (no-op if absent)."""

        cls._vision_registry.pop(provider.strip().lower(), None)

    @classmethod
    def registered_vision_providers(cls) -> list[str]:
        return sorted(cls._vision_registry)

    @classmethod
    def create_vision_llm(cls, settings: Settings) -> BaseVisionLLM:
        """Create the Vision LLM instance configured by `settings.vision_llm`.

        Raises:
            ValueError: If `settings.vision_llm` is None (section absent in config)
                        or if the provider is not registered.
        """

        if settings.vision_llm is None:
            raise ValueError(
                "vision_llm section is missing from settings. "
                "Add a 'vision_llm' block to config/settings.yaml to enable vision support."
            )
        provider = settings.vision_llm.provider.strip().lower()
        vision_cls = cls._vision_registry.get(provider)
        if vision_cls is None:
            registered = ", ".join(cls.registered_vision_providers()) or "(none)"
            raise ValueError(
                f"Unknown Vision LLM provider: '{provider}'. "
                f"Registered vision providers: {registered}"
            )
        return vision_cls(settings.vision_llm)
