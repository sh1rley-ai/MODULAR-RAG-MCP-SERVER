"""B5 — Tests for BaseReranker abstraction, NoneReranker fallback and RerankerFactory routing.

All tests are pure unit tests: fake rerankers are registered in-test to
verify factory routing. Fake rerankers reorder deterministically so behaviour
can be asserted without any model or external dependency.
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
from libs.reranker.base_reranker import BaseReranker, NoneReranker, RerankerError
from libs.reranker.reranker_factory import RerankerFactory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_settings(
    provider: str = "none",
    enabled: bool = True,
    model: str = "none",
    top_k: int = 5,
) -> Settings:
    """Build a full Settings object with the given rerank section."""
    return Settings(
        llm=LLMSettings(provider="openai", model="gpt-4o-mini", temperature=0.0, max_tokens=256),
        embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small", dimensions=4),
        vector_store=VectorStoreSettings(provider="chroma", persist_directory="data/db/chroma", collection_name="default"),
        retrieval=RetrievalSettings(dense_top_k=20, sparse_top_k=20, fusion_top_k=10, rrf_k=60),
        rerank=RerankSettings(enabled=enabled, provider=provider, model=model, top_k=top_k),
        evaluation=EvaluationSettings(enabled=False, provider="custom", metrics=["hit_rate"]),
        observability=ObservabilitySettings(
            log_level="INFO", trace_enabled=True, trace_file="logs/traces.jsonl", structured_logging=False
        ),
    )


def make_candidates(*ids: str) -> List[Dict[str, Any]]:
    """Build a candidate list with descending scores in the given id order."""
    return [
        {"id": candidate_id, "score": 1.0 - index * 0.1, "text": f"text-{candidate_id}", "metadata": {}}
        for index, candidate_id in enumerate(ids)
    ]


class ReverseFakeReranker(BaseReranker):
    """In-test stub backend: reverses the candidate order deterministically."""

    def __init__(self, rerank_settings: RerankSettings) -> None:
        self.settings = rerank_settings

    def rerank(
        self, query: str, candidates: List[Dict[str, Any]], trace: Any | None = None
    ) -> List[Dict[str, Any]]:
        self.validate_query(query)
        self.validate_candidates(candidates)
        return list(reversed(candidates))


class AnotherFakeReranker(ReverseFakeReranker):
    """Second stub backend to verify routing between multiple backends."""


@pytest.fixture(autouse=True)
def clean_registry():
    """Snapshot and restore the factory registry around each test.

    Yields the pre-test snapshot so tests can assert on module-level defaults.
    """
    snapshot = dict(RerankerFactory._registry)
    RerankerFactory._registry.clear()
    yield snapshot
    RerankerFactory._registry.clear()
    RerankerFactory._registry.update(snapshot)


# ---------------------------------------------------------------------------
# TestRerankerFactoryRouting
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRerankerFactoryRouting:
    """Factory routes settings.rerank.provider to the registered implementation."""

    def test_none_backend_registered_by_default(self, clean_registry) -> None:
        assert clean_registry.get("none") is NoneReranker

    def test_create_returns_registered_backend(self) -> None:
        RerankerFactory.register("fake", ReverseFakeReranker)
        reranker = RerankerFactory.create(make_settings(provider="fake"))
        assert isinstance(reranker, ReverseFakeReranker)

    def test_created_instance_receives_rerank_settings(self) -> None:
        RerankerFactory.register("fake", ReverseFakeReranker)
        reranker = RerankerFactory.create(make_settings(provider="fake", model="bge-reranker", top_k=3))
        assert reranker.settings.model == "bge-reranker"
        assert reranker.settings.top_k == 3

    def test_routes_between_multiple_backends(self) -> None:
        RerankerFactory.register("fake", ReverseFakeReranker)
        RerankerFactory.register("another", AnotherFakeReranker)
        assert isinstance(RerankerFactory.create(make_settings(provider="another")), AnotherFakeReranker)
        reranker = RerankerFactory.create(make_settings(provider="fake"))
        assert isinstance(reranker, ReverseFakeReranker)
        assert not isinstance(reranker, AnotherFakeReranker)

    def test_backend_name_is_case_insensitive(self) -> None:
        RerankerFactory.register("Fake", ReverseFakeReranker)
        reranker = RerankerFactory.create(make_settings(provider="FAKE"))
        assert isinstance(reranker, ReverseFakeReranker)

    def test_provider_none_returns_none_reranker(self) -> None:
        RerankerFactory.register("none", NoneReranker)
        reranker = RerankerFactory.create(make_settings(provider="none"))
        assert isinstance(reranker, NoneReranker)

    def test_disabled_returns_none_reranker_regardless_of_provider(self) -> None:
        RerankerFactory.register("fake", ReverseFakeReranker)
        reranker = RerankerFactory.create(make_settings(provider="fake", enabled=False))
        assert isinstance(reranker, NoneReranker)

    def test_disabled_works_with_empty_registry(self) -> None:
        reranker = RerankerFactory.create(make_settings(provider="whatever", enabled=False))
        assert isinstance(reranker, NoneReranker)

    def test_unknown_backend_raises_readable_error(self) -> None:
        RerankerFactory.register("fake", ReverseFakeReranker)
        with pytest.raises(ValueError, match="Unknown reranker backend: 'nonexistent'"):
            RerankerFactory.create(make_settings(provider="nonexistent"))

    def test_unknown_backend_error_lists_registered(self) -> None:
        RerankerFactory.register("fake", ReverseFakeReranker)
        with pytest.raises(ValueError, match="fake"):
            RerankerFactory.create(make_settings(provider="nonexistent"))

    def test_unregister_removes_backend(self) -> None:
        RerankerFactory.register("fake", ReverseFakeReranker)
        RerankerFactory.unregister("fake")
        with pytest.raises(ValueError, match="Unknown reranker backend"):
            RerankerFactory.create(make_settings(provider="fake"))


# ---------------------------------------------------------------------------
# TestRerankerFactoryRegistration
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRerankerFactoryRegistration:
    """register() validates its inputs."""

    def test_register_rejects_non_basereranker_class(self) -> None:
        class NotAReranker:
            pass

        with pytest.raises(TypeError, match="BaseReranker"):
            RerankerFactory.register("bad", NotAReranker)  # type: ignore[arg-type]

    def test_register_rejects_empty_backend_name(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            RerankerFactory.register("  ", ReverseFakeReranker)

    def test_registered_backends_lists_names(self) -> None:
        RerankerFactory.register("fake", ReverseFakeReranker)
        RerankerFactory.register("another", AnotherFakeReranker)
        assert RerankerFactory.registered_backends() == ["another", "fake"]


# ---------------------------------------------------------------------------
# TestBaseRerankerContract
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBaseRerankerContract:
    """BaseReranker abstract contract and input validation."""

    def test_base_reranker_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError):
            BaseReranker()  # type: ignore[abstract]

    def test_rerank_accepts_optional_trace(self) -> None:
        reranker = NoneReranker(make_settings().rerank)
        candidates = make_candidates("a", "b")
        assert reranker.rerank("query", candidates, trace=object()) == candidates

    def test_validate_query_rejects_non_string(self) -> None:
        with pytest.raises(RerankerError, match="query must be a non-empty string"):
            BaseReranker.validate_query(42)

    def test_validate_query_rejects_blank_string(self) -> None:
        with pytest.raises(RerankerError, match="query must be a non-empty string"):
            BaseReranker.validate_query("   ")

    def test_validate_candidates_rejects_non_list(self) -> None:
        with pytest.raises(RerankerError, match="candidates must be a list"):
            BaseReranker.validate_candidates("not a list")

    def test_validate_candidates_rejects_non_dict_element(self) -> None:
        with pytest.raises(RerankerError, match=r"candidates\[1\] must be a dict"):
            BaseReranker.validate_candidates([{"id": "a"}, "oops"])

    def test_validate_candidates_rejects_missing_id(self) -> None:
        with pytest.raises(RerankerError, match=r"candidates\[0\].id must be a non-empty string"):
            BaseReranker.validate_candidates([{"score": 0.5}])

    def test_validate_candidates_accepts_empty_list(self) -> None:
        BaseReranker.validate_candidates([])


# ---------------------------------------------------------------------------
# TestNoneRerankerFallback
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestNoneRerankerFallback:
    """NoneReranker keeps the original order untouched (backend=none acceptance)."""

    def test_preserves_original_order(self) -> None:
        reranker = NoneReranker(make_settings().rerank)
        candidates = make_candidates("a", "b", "c")
        ranked = reranker.rerank("query", candidates)
        assert [candidate["id"] for candidate in ranked] == ["a", "b", "c"]

    def test_does_not_drop_or_truncate_candidates(self) -> None:
        reranker = NoneReranker(make_settings(top_k=2).rerank)
        candidates = make_candidates("a", "b", "c", "d", "e")
        assert len(reranker.rerank("query", candidates)) == 5

    def test_returns_new_list_without_mutating_input(self) -> None:
        reranker = NoneReranker(make_settings().rerank)
        candidates = make_candidates("a", "b")
        ranked = reranker.rerank("query", candidates)
        assert ranked is not candidates
        assert ranked == candidates

    def test_is_deterministic(self) -> None:
        reranker = NoneReranker(make_settings().rerank)
        candidates = make_candidates("a", "b", "c")
        assert reranker.rerank("query", candidates) == reranker.rerank("query", candidates)

    def test_empty_candidates_returns_empty_list(self) -> None:
        reranker = NoneReranker(make_settings().rerank)
        assert reranker.rerank("query", []) == []

    def test_rejects_invalid_query(self) -> None:
        reranker = NoneReranker(make_settings().rerank)
        with pytest.raises(RerankerError, match="query must be a non-empty string"):
            reranker.rerank("", make_candidates("a"))

    def test_rejects_invalid_candidates(self) -> None:
        reranker = NoneReranker(make_settings().rerank)
        with pytest.raises(RerankerError, match="candidates must be a list"):
            reranker.rerank("query", None)  # type: ignore[arg-type]
