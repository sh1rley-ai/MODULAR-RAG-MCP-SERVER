"""Evaluator abstraction base class.

All evaluator backends (CustomEvaluator here, RagasEvaluator in Phase H)
implement `BaseEvaluator` so upper layers depend only on this contract and
the backend can be swapped via `config/settings.yaml` (evaluation.provider).

Evaluation contract:
    evaluate(query, retrieved_ids, golden_ids) -> {metric_name: float}

`retrieved_ids` is the ranked list of chunk ids returned by retrieval
(most relevant first). `golden_ids` is the expected relevant set from the
golden test set. The returned mapping is deterministic for identical input.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class EvaluatorError(RuntimeError):
    """Raised when an evaluation call fails or its input is invalid."""


class BaseEvaluator(ABC):
    """Abstract base class for evaluator backends.

    Concrete backends are constructed by `EvaluatorFactory` and receive the
    `EvaluationSettings` section from the global settings as their first
    argument (metrics selection lives there).
    """

    @abstractmethod
    def evaluate(
        self,
        query: str,
        retrieved_ids: List[str],
        golden_ids: List[str],
        trace: Any | None = None,
    ) -> Dict[str, float]:
        """Return metric values for one query.

        `retrieved_ids` must be ranked (most relevant first). `trace` is an
        optional TraceContext (Phase F).
        """

    @staticmethod
    def validate_query(query: Any) -> None:
        """Validate the query shape; raise EvaluatorError with a readable message."""

        if not isinstance(query, str) or not query.strip():
            raise EvaluatorError(
                f"query must be a non-empty string, got {type(query).__name__}"
            )

    @staticmethod
    def validate_ids(ids: Any, field: str, allow_empty: bool = True) -> None:
        """Validate an id list shape; raise EvaluatorError with a readable message."""

        if not isinstance(ids, list):
            raise EvaluatorError(
                f"{field} must be a list of strings, got {type(ids).__name__}"
            )
        if not allow_empty and not ids:
            raise EvaluatorError(f"{field} must not be empty")
        for index, item in enumerate(ids):
            if not isinstance(item, str) or not item:
                raise EvaluatorError(f"{field}[{index}] must be a non-empty string")
