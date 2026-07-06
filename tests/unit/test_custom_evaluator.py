"""B6 — Tests for BaseEvaluator abstraction, CustomEvaluator metrics and EvaluatorFactory routing.

All tests are pure unit tests: metrics are computed from id lists only, and
fake evaluators are registered in-test to verify factory routing, so no LLM
or external dependency is needed.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from core.settings import (
    EmbeddingSettings,
    EvaluationSettings,
    LLMSettings,
    ObservabilitySettings,
    RerankSettings,
    RetrievalSettings,
    Settings,
    VectorStoreSettings,
)
from libs.evaluator.base_evaluator import BaseEvaluator, EvaluatorError
from libs.evaluator.custom_evaluator import CustomEvaluator
from libs.evaluator.evaluator_factory import EvaluatorFactory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_evaluation_settings(
    provider: str = "custom",
    metrics: List[str] | None = None,
    enabled: bool = True,
) -> EvaluationSettings:
    return EvaluationSettings(
        enabled=enabled,
        provider=provider,
        metrics=metrics if metrics is not None else ["hit_rate", "mrr"],
    )


def make_settings(
    provider: str = "custom",
    metrics: List[str] | None = None,
    enabled: bool = True,
) -> Settings:
    """Build a full Settings object with the given evaluation section."""
    return Settings(
        llm=LLMSettings(provider="openai", model="gpt-4o-mini", temperature=0.0, max_tokens=256),
        embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small", dimensions=4),
        vector_store=VectorStoreSettings(provider="chroma", persist_directory="data/db/chroma", collection_name="default"),
        retrieval=RetrievalSettings(dense_top_k=20, sparse_top_k=20, fusion_top_k=10, rrf_k=60),
        rerank=RerankSettings(enabled=False, provider="none", model="none", top_k=5),
        evaluation=make_evaluation_settings(provider=provider, metrics=metrics, enabled=enabled),
        observability=ObservabilitySettings(
            log_level="INFO", trace_enabled=True, trace_file="logs/traces.jsonl", structured_logging=False
        ),
    )


class FakeEvaluator(BaseEvaluator):
    """In-test stub backend: returns a constant metric to verify routing."""

    def __init__(self, evaluation_settings: EvaluationSettings) -> None:
        self.settings = evaluation_settings

    def evaluate(
        self,
        query: str,
        retrieved_ids: List[str],
        golden_ids: List[str],
        trace: Any | None = None,
    ) -> Dict[str, float]:
        return {"fake_metric": 1.0}


class AnotherFakeEvaluator(FakeEvaluator):
    """Second stub backend to verify routing between multiple providers."""


@pytest.fixture(autouse=True)
def clean_registry():
    """Snapshot and restore the factory registry around each test."""
    snapshot = dict(EvaluatorFactory._registry)
    yield
    EvaluatorFactory._registry.clear()
    EvaluatorFactory._registry.update(snapshot)


# ---------------------------------------------------------------------------
# BaseEvaluator — abstractness and validation helpers
# ---------------------------------------------------------------------------

class TestBaseEvaluator:
    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            BaseEvaluator()  # type: ignore[abstract]

    def test_validate_query_rejects_non_string(self):
        with pytest.raises(EvaluatorError, match="query must be a non-empty string"):
            BaseEvaluator.validate_query(123)

    def test_validate_query_rejects_blank_string(self):
        with pytest.raises(EvaluatorError, match="query must be a non-empty string"):
            BaseEvaluator.validate_query("   ")

    def test_validate_ids_rejects_non_list(self):
        with pytest.raises(EvaluatorError, match="retrieved_ids must be a list"):
            BaseEvaluator.validate_ids("c1", "retrieved_ids")

    def test_validate_ids_rejects_non_string_item(self):
        with pytest.raises(EvaluatorError, match=r"retrieved_ids\[1\] must be a non-empty string"):
            BaseEvaluator.validate_ids(["c1", 2], "retrieved_ids")

    def test_validate_ids_rejects_empty_string_item(self):
        with pytest.raises(EvaluatorError, match=r"golden_ids\[0\] must be a non-empty string"):
            BaseEvaluator.validate_ids([""], "golden_ids")

    def test_validate_ids_allows_empty_list_by_default(self):
        BaseEvaluator.validate_ids([], "retrieved_ids")

    def test_validate_ids_rejects_empty_list_when_disallowed(self):
        with pytest.raises(EvaluatorError, match="golden_ids must not be empty"):
            BaseEvaluator.validate_ids([], "golden_ids", allow_empty=False)


# ---------------------------------------------------------------------------
# CustomEvaluator — hit_rate
# ---------------------------------------------------------------------------

class TestHitRate:
    def make(self, metrics: List[str] | None = None) -> CustomEvaluator:
        return CustomEvaluator(make_evaluation_settings(metrics=metrics or ["hit_rate"]))

    def test_hit_at_first_position(self):
        result = self.make().evaluate("q", ["g1", "c2", "c3"], ["g1"])
        assert result == {"hit_rate": 1.0}

    def test_hit_at_later_position(self):
        result = self.make().evaluate("q", ["c1", "c2", "g1"], ["g1"])
        assert result == {"hit_rate": 1.0}

    def test_miss(self):
        result = self.make().evaluate("q", ["c1", "c2", "c3"], ["g1"])
        assert result == {"hit_rate": 0.0}

    def test_empty_retrieved_is_a_miss(self):
        result = self.make().evaluate("q", [], ["g1"])
        assert result == {"hit_rate": 0.0}

    def test_multiple_golden_any_hit_counts(self):
        result = self.make().evaluate("q", ["c1", "g2"], ["g1", "g2"])
        assert result == {"hit_rate": 1.0}


# ---------------------------------------------------------------------------
# CustomEvaluator — mrr
# ---------------------------------------------------------------------------

class TestMRR:
    def make(self) -> CustomEvaluator:
        return CustomEvaluator(make_evaluation_settings(metrics=["mrr"]))

    def test_first_position_gives_1(self):
        assert self.make().evaluate("q", ["g1", "c2", "c3"], ["g1"]) == {"mrr": 1.0}

    def test_second_position_gives_half(self):
        assert self.make().evaluate("q", ["c1", "g1", "c3"], ["g1"]) == {"mrr": 0.5}

    def test_third_position_gives_one_third(self):
        result = self.make().evaluate("q", ["c1", "c2", "g1"], ["g1"])
        assert result["mrr"] == pytest.approx(1.0 / 3.0)

    def test_miss_gives_0(self):
        assert self.make().evaluate("q", ["c1", "c2"], ["g1"]) == {"mrr": 0.0}

    def test_empty_retrieved_gives_0(self):
        assert self.make().evaluate("q", [], ["g1"]) == {"mrr": 0.0}

    def test_uses_first_golden_hit_only(self):
        # g2 at rank 2 comes before g1 at rank 3 — MRR uses the earliest hit.
        assert self.make().evaluate("q", ["c1", "g2", "g1"], ["g1", "g2"]) == {"mrr": 0.5}


# ---------------------------------------------------------------------------
# CustomEvaluator — metrics selection, determinism, validation
# ---------------------------------------------------------------------------

class TestCustomEvaluator:
    def test_computes_all_configured_metrics(self):
        evaluator = CustomEvaluator(make_evaluation_settings(metrics=["hit_rate", "mrr"]))
        result = evaluator.evaluate("q", ["c1", "g1"], ["g1"])
        assert result == {"hit_rate": 1.0, "mrr": 0.5}

    def test_only_configured_metrics_are_returned(self):
        evaluator = CustomEvaluator(make_evaluation_settings(metrics=["hit_rate"]))
        result = evaluator.evaluate("q", ["g1"], ["g1"])
        assert "mrr" not in result

    def test_metric_names_are_case_insensitive(self):
        evaluator = CustomEvaluator(make_evaluation_settings(metrics=["Hit_Rate", "MRR"]))
        result = evaluator.evaluate("q", ["g1"], ["g1"])
        assert result == {"hit_rate": 1.0, "mrr": 1.0}

    def test_deterministic_for_identical_input(self):
        evaluator = CustomEvaluator(make_evaluation_settings())
        first = evaluator.evaluate("q", ["c1", "g1", "c3"], ["g1"])
        second = evaluator.evaluate("q", ["c1", "g1", "c3"], ["g1"])
        assert first == second

    def test_unknown_metric_fails_at_construction(self):
        with pytest.raises(EvaluatorError, match="Unknown metric.*ndcg"):
            CustomEvaluator(make_evaluation_settings(metrics=["hit_rate", "ndcg"]))

    def test_empty_metrics_list_fails_at_construction(self):
        with pytest.raises(EvaluatorError, match="at least one metric"):
            CustomEvaluator(make_evaluation_settings(metrics=[]))

    def test_rejects_empty_golden_ids(self):
        evaluator = CustomEvaluator(make_evaluation_settings())
        with pytest.raises(EvaluatorError, match="golden_ids must not be empty"):
            evaluator.evaluate("q", ["c1"], [])

    def test_rejects_invalid_query(self):
        evaluator = CustomEvaluator(make_evaluation_settings())
        with pytest.raises(EvaluatorError, match="query must be a non-empty string"):
            evaluator.evaluate("", ["c1"], ["g1"])

    def test_rejects_non_list_retrieved_ids(self):
        evaluator = CustomEvaluator(make_evaluation_settings())
        with pytest.raises(EvaluatorError, match="retrieved_ids must be a list"):
            evaluator.evaluate("q", "c1", ["g1"])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# EvaluatorFactory — registration
# ---------------------------------------------------------------------------

class TestFactoryRegistration:
    def test_custom_is_registered_by_default(self):
        assert "custom" in EvaluatorFactory.registered_providers()

    def test_register_and_list(self):
        EvaluatorFactory.register("fake", FakeEvaluator)
        assert "fake" in EvaluatorFactory.registered_providers()

    def test_register_normalizes_case_and_whitespace(self):
        EvaluatorFactory.register("  FaKe  ", FakeEvaluator)
        assert "fake" in EvaluatorFactory.registered_providers()

    def test_register_rejects_empty_name(self):
        with pytest.raises(ValueError, match="non-empty string"):
            EvaluatorFactory.register("   ", FakeEvaluator)

    def test_register_rejects_non_evaluator_class(self):
        with pytest.raises(TypeError, match="must be a subclass of BaseEvaluator"):
            EvaluatorFactory.register("bad", dict)  # type: ignore[arg-type]

    def test_unregister_removes_provider(self):
        EvaluatorFactory.register("fake", FakeEvaluator)
        EvaluatorFactory.unregister("fake")
        assert "fake" not in EvaluatorFactory.registered_providers()

    def test_unregister_missing_provider_is_noop(self):
        EvaluatorFactory.unregister("does-not-exist")


# ---------------------------------------------------------------------------
# EvaluatorFactory — creation and routing
# ---------------------------------------------------------------------------

class TestFactoryCreate:
    def test_creates_custom_evaluator(self):
        evaluator = EvaluatorFactory.create(make_settings(provider="custom"))
        assert isinstance(evaluator, CustomEvaluator)

    def test_created_evaluator_receives_evaluation_settings(self):
        settings = make_settings(provider="custom", metrics=["mrr"])
        evaluator = EvaluatorFactory.create(settings)
        assert evaluator.settings is settings.evaluation
        assert evaluator.metrics == ["mrr"]

    def test_routes_between_multiple_providers(self):
        EvaluatorFactory.register("fake", FakeEvaluator)
        EvaluatorFactory.register("another", AnotherFakeEvaluator)
        assert type(EvaluatorFactory.create(make_settings(provider="fake"))) is FakeEvaluator
        assert type(EvaluatorFactory.create(make_settings(provider="another"))) is AnotherFakeEvaluator

    def test_provider_lookup_is_case_insensitive(self):
        EvaluatorFactory.register("fake", FakeEvaluator)
        evaluator = EvaluatorFactory.create(make_settings(provider="  FAKE "))
        assert isinstance(evaluator, FakeEvaluator)

    def test_unknown_provider_raises_with_registered_list(self):
        with pytest.raises(ValueError, match="Unknown evaluator provider: 'ragas'.*custom"):
            EvaluatorFactory.create(make_settings(provider="ragas"))

    def test_enabled_flag_does_not_block_creation(self):
        # evaluation.enabled gates whether EvalRunner runs, not factory creation.
        evaluator = EvaluatorFactory.create(make_settings(enabled=False))
        assert isinstance(evaluator, CustomEvaluator)

    def test_end_to_end_metrics_via_factory(self):
        evaluator = EvaluatorFactory.create(make_settings(metrics=["hit_rate", "mrr"]))
        result = evaluator.evaluate("what is rrf", ["c1", "g1", "c3"], ["g1"])
        assert result == {"hit_rate": 1.0, "mrr": 0.5}
