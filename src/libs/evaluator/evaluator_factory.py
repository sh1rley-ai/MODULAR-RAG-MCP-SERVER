"""Evaluator factory: routes `evaluation.provider` from settings to a registered implementation.

New backends (RagasEvaluator in Phase H) register themselves here via
`EvaluatorFactory.register`, so adding an evaluation backend requires no
change to calling code — only configuration.

`CustomEvaluator` is registered under "custom" at import time as the
built-in default. The `evaluation.enabled` flag is intentionally not
checked here: evaluation is an explicit offline action, so the caller
(EvalRunner, Phase H) decides whether to run it at all.
"""

from __future__ import annotations

from typing import Dict, Type

from core.settings import Settings

from libs.evaluator.base_evaluator import BaseEvaluator
from libs.evaluator.custom_evaluator import CustomEvaluator


class EvaluatorFactory:
    """Creates a `BaseEvaluator` instance based on `settings.evaluation`."""

    _registry: Dict[str, Type[BaseEvaluator]] = {}

    @classmethod
    def register(cls, provider: str, evaluator_cls: Type[BaseEvaluator]) -> None:
        """Register an evaluator implementation under a provider name (case-insensitive)."""

        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("provider name must be a non-empty string")
        if not (isinstance(evaluator_cls, type) and issubclass(evaluator_cls, BaseEvaluator)):
            raise TypeError(f"{evaluator_cls!r} must be a subclass of BaseEvaluator")
        cls._registry[provider.strip().lower()] = evaluator_cls

    @classmethod
    def unregister(cls, provider: str) -> None:
        """Remove a provider from the registry (no-op if absent)."""

        cls._registry.pop(provider.strip().lower(), None)

    @classmethod
    def registered_providers(cls) -> list[str]:
        return sorted(cls._registry)

    @classmethod
    def create(cls, settings: Settings) -> BaseEvaluator:
        """Create the evaluator instance configured by `settings.evaluation`."""

        provider = settings.evaluation.provider.strip().lower()
        evaluator_cls = cls._registry.get(provider)
        if evaluator_cls is None:
            registered = ", ".join(cls.registered_providers()) or "(none)"
            raise ValueError(
                f"Unknown evaluator provider: '{provider}'. Registered providers: {registered}"
            )
        return evaluator_cls(settings.evaluation)


EvaluatorFactory.register("custom", CustomEvaluator)
