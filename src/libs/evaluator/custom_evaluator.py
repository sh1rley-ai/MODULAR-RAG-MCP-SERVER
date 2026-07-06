"""Custom lightweight retrieval metrics evaluator (hit_rate / mrr).

Default evaluation backend: computes rank-based metrics purely from
retrieved_ids vs golden_ids, so it needs no LLM or external service and is
fully deterministic. Which metrics are computed is driven by
`evaluation.metrics` in `config/settings.yaml`.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from libs.evaluator.base_evaluator import BaseEvaluator, EvaluatorError


def _hit_rate(retrieved_ids: List[str], golden_ids: List[str]) -> float:
    """1.0 if any golden id appears in the retrieved list, else 0.0."""

    golden = set(golden_ids)
    return 1.0 if any(chunk_id in golden for chunk_id in retrieved_ids) else 0.0


def _mrr(retrieved_ids: List[str], golden_ids: List[str]) -> float:
    """Reciprocal rank of the first golden id in the retrieved list (0.0 on miss)."""

    golden = set(golden_ids)
    for index, chunk_id in enumerate(retrieved_ids):
        if chunk_id in golden:
            return 1.0 / (index + 1)
    return 0.0


class CustomEvaluator(BaseEvaluator):
    """Computes the metrics listed in `evaluation.metrics` for one query.

    Supported metric names: "hit_rate", "mrr". Unknown names fail at
    construction time so a config typo is caught before any evaluation run.
    """

    _metric_fns: Dict[str, Callable[[List[str], List[str]], float]] = {
        "hit_rate": _hit_rate,
        "mrr": _mrr,
    }

    def __init__(self, evaluation_settings: Any) -> None:
        self.settings = evaluation_settings
        metrics = [str(name).strip().lower() for name in evaluation_settings.metrics]
        if not metrics:
            raise EvaluatorError(
                "evaluation.metrics must list at least one metric "
                f"(supported: {', '.join(sorted(self._metric_fns))})"
            )
        unknown = [name for name in metrics if name not in self._metric_fns]
        if unknown:
            raise EvaluatorError(
                f"Unknown metric(s) in evaluation.metrics: {', '.join(unknown)}. "
                f"Supported metrics: {', '.join(sorted(self._metric_fns))}"
            )
        self.metrics = metrics

    def evaluate(
        self,
        query: str,
        retrieved_ids: List[str],
        golden_ids: List[str],
        trace: Any | None = None,
    ) -> Dict[str, float]:
        self.validate_query(query)
        self.validate_ids(retrieved_ids, "retrieved_ids")
        self.validate_ids(golden_ids, "golden_ids", allow_empty=False)
        return {
            name: self._metric_fns[name](retrieved_ids, golden_ids)
            for name in self.metrics
        }
