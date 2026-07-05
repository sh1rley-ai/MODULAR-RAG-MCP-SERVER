"""B4 — Contract tests for BaseVectorStore and VectorStoreFactory routing.

All tests are pure unit tests: an in-memory fake backend is registered
in-test to verify factory routing and to pin the input/output shape of the
upsert/query contract. No real database, no network.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

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
from libs.vector_store.base_vector_store import BaseVectorStore, VectorStoreError
from libs.vector_store.vector_store_factory import VectorStoreFactory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_settings(
    provider: str = "fake",
    persist_directory: str = "data/db/chroma",
    collection_name: str = "default",
) -> Settings:
    """Build a full Settings object with the given vector_store provider."""
    return Settings(
        llm=LLMSettings(provider="openai", model="gpt-4o-mini", temperature=0.0, max_tokens=256),
        embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small", dimensions=4),
        vector_store=VectorStoreSettings(
            provider=provider, persist_directory=persist_directory, collection_name=collection_name
        ),
        retrieval=RetrievalSettings(dense_top_k=20, sparse_top_k=20, fusion_top_k=10, rrf_k=60),
        rerank=RerankSettings(enabled=False, provider="none", model="none", top_k=5),
        evaluation=EvaluationSettings(enabled=False, provider="custom", metrics=["hit_rate"]),
        observability=ObservabilitySettings(
            log_level="INFO", trace_enabled=True, trace_file="logs/traces.jsonl", structured_logging=False
        ),
    )


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class FakeVectorStore(BaseVectorStore):
    """In-memory backend: cosine-similarity query over upserted records."""

    def __init__(self, vector_store_settings: VectorStoreSettings) -> None:
        self.settings = vector_store_settings
        self._records: Dict[str, Dict[str, Any]] = {}

    def upsert(self, records: List[Dict[str, Any]], trace: Any | None = None) -> None:
        self.validate_records(records)
        for record in records:
            self._records[record["id"]] = record

    def query(
        self,
        vector: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        trace: Any | None = None,
    ) -> List[Dict[str, Any]]:
        self.validate_query_args(vector, top_k, filters)
        candidates = list(self._records.values())
        if filters:
            candidates = [
                record
                for record in candidates
                if all(record.get("metadata", {}).get(key) == value for key, value in filters.items())
            ]
        scored = sorted(
            candidates,
            key=lambda record: cosine_similarity(vector, record["vector"]),
            reverse=True,
        )
        return [
            {
                "id": record["id"],
                "score": cosine_similarity(vector, record["vector"]),
                "text": record.get("text"),
                "metadata": record.get("metadata", {}),
            }
            for record in scored[:top_k]
        ]


class AnotherFakeVectorStore(FakeVectorStore):
    """Second stub backend to verify routing between multiple providers."""


@pytest.fixture(autouse=True)
def clean_registry():
    """Snapshot and restore the factory registry around each test."""
    snapshot = dict(VectorStoreFactory._registry)
    VectorStoreFactory._registry.clear()
    yield
    VectorStoreFactory._registry.clear()
    VectorStoreFactory._registry.update(snapshot)


@pytest.fixture
def store() -> FakeVectorStore:
    return FakeVectorStore(make_settings().vector_store)


SAMPLE_RECORDS = [
    {"id": "c1", "vector": [1.0, 0.0, 0.0, 0.0], "text": "alpha", "metadata": {"lang": "en"}},
    {"id": "c2", "vector": [0.0, 1.0, 0.0, 0.0], "text": "beta", "metadata": {"lang": "en"}},
    {"id": "c3", "vector": [0.0, 0.0, 1.0, 0.0], "text": "gamma", "metadata": {"lang": "zh"}},
]


# ---------------------------------------------------------------------------
# TestVectorStoreFactoryRouting
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestVectorStoreFactoryRouting:
    """Factory routes settings.vector_store.provider to the registered implementation."""

    def test_create_returns_registered_provider(self) -> None:
        VectorStoreFactory.register("fake", FakeVectorStore)
        store = VectorStoreFactory.create(make_settings(provider="fake"))
        assert isinstance(store, FakeVectorStore)

    def test_created_instance_receives_vector_store_settings(self) -> None:
        VectorStoreFactory.register("fake", FakeVectorStore)
        store = VectorStoreFactory.create(
            make_settings(provider="fake", persist_directory="tmp/db", collection_name="docs")
        )
        assert store.settings.persist_directory == "tmp/db"
        assert store.settings.collection_name == "docs"

    def test_routes_between_multiple_providers(self) -> None:
        VectorStoreFactory.register("fake", FakeVectorStore)
        VectorStoreFactory.register("another", AnotherFakeVectorStore)
        assert isinstance(VectorStoreFactory.create(make_settings(provider="another")), AnotherFakeVectorStore)
        store = VectorStoreFactory.create(make_settings(provider="fake"))
        assert isinstance(store, FakeVectorStore)
        assert not isinstance(store, AnotherFakeVectorStore)

    def test_provider_name_is_case_insensitive(self) -> None:
        VectorStoreFactory.register("Fake", FakeVectorStore)
        store = VectorStoreFactory.create(make_settings(provider="FAKE"))
        assert isinstance(store, FakeVectorStore)

    def test_unknown_provider_raises_readable_error(self) -> None:
        VectorStoreFactory.register("fake", FakeVectorStore)
        with pytest.raises(ValueError, match="Unknown vector store provider: 'nonexistent'"):
            VectorStoreFactory.create(make_settings(provider="nonexistent"))

    def test_unknown_provider_error_lists_registered(self) -> None:
        VectorStoreFactory.register("fake", FakeVectorStore)
        with pytest.raises(ValueError, match="fake"):
            VectorStoreFactory.create(make_settings(provider="nonexistent"))

    def test_unregister_removes_provider(self) -> None:
        VectorStoreFactory.register("fake", FakeVectorStore)
        VectorStoreFactory.unregister("fake")
        with pytest.raises(ValueError, match="Unknown vector store provider"):
            VectorStoreFactory.create(make_settings(provider="fake"))


# ---------------------------------------------------------------------------
# TestVectorStoreFactoryRegistration
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestVectorStoreFactoryRegistration:
    """register() validates its inputs."""

    def test_register_rejects_non_basevectorstore_class(self) -> None:
        class NotAStore:
            pass

        with pytest.raises(TypeError, match="BaseVectorStore"):
            VectorStoreFactory.register("bad", NotAStore)  # type: ignore[arg-type]

    def test_register_rejects_empty_provider_name(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            VectorStoreFactory.register("  ", FakeVectorStore)

    def test_registered_providers_lists_names(self) -> None:
        VectorStoreFactory.register("fake", FakeVectorStore)
        VectorStoreFactory.register("another", AnotherFakeVectorStore)
        assert VectorStoreFactory.registered_providers() == ["another", "fake"]


# ---------------------------------------------------------------------------
# TestUpsertContract
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestUpsertContract:
    """upsert() input shape and idempotency."""

    def test_base_vector_store_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError):
            BaseVectorStore()  # type: ignore[abstract]

    def test_upsert_accepts_valid_records(self, store: FakeVectorStore) -> None:
        store.upsert(SAMPLE_RECORDS)
        assert len(store._records) == 3

    def test_upsert_same_id_overwrites(self, store: FakeVectorStore) -> None:
        store.upsert(SAMPLE_RECORDS)
        store.upsert([{"id": "c1", "vector": [0.5, 0.5, 0.0, 0.0], "text": "alpha-v2"}])
        assert len(store._records) == 3
        assert store._records["c1"]["text"] == "alpha-v2"

    def test_upsert_accepts_optional_trace(self, store: FakeVectorStore) -> None:
        store.upsert(SAMPLE_RECORDS[:1], trace=object())
        assert "c1" in store._records

    def test_validate_records_rejects_empty_list(self) -> None:
        with pytest.raises(VectorStoreError, match="non-empty list"):
            BaseVectorStore.validate_records([])

    def test_validate_records_rejects_non_list(self) -> None:
        with pytest.raises(VectorStoreError, match="non-empty list"):
            BaseVectorStore.validate_records({"id": "c1"})

    def test_validate_records_rejects_non_dict_item(self) -> None:
        with pytest.raises(VectorStoreError, match=r"records\[0\] must be a dict"):
            BaseVectorStore.validate_records(["not-a-dict"])

    def test_validate_records_rejects_missing_id(self) -> None:
        with pytest.raises(VectorStoreError, match=r"records\[0\].id"):
            BaseVectorStore.validate_records([{"vector": [1.0]}])

    def test_validate_records_rejects_empty_vector(self) -> None:
        with pytest.raises(VectorStoreError, match=r"records\[0\].vector"):
            BaseVectorStore.validate_records([{"id": "c1", "vector": []}])

    def test_validate_records_rejects_non_numeric_vector(self) -> None:
        with pytest.raises(VectorStoreError, match=r"records\[1\].vector .*only numbers"):
            BaseVectorStore.validate_records(
                [{"id": "c1", "vector": [1.0]}, {"id": "c2", "vector": ["x"]}]
            )

    def test_validate_records_rejects_non_dict_metadata(self) -> None:
        with pytest.raises(VectorStoreError, match=r"records\[0\].metadata"):
            BaseVectorStore.validate_records([{"id": "c1", "vector": [1.0], "metadata": "tag"}])


# ---------------------------------------------------------------------------
# TestQueryContract
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestQueryContract:
    """query() input validation and output shape."""

    def test_query_returns_list_of_result_dicts(self, store: FakeVectorStore) -> None:
        store.upsert(SAMPLE_RECORDS)
        results = store.query([1.0, 0.0, 0.0, 0.0], top_k=2)
        assert isinstance(results, list)
        for result in results:
            assert set(result) >= {"id", "score", "metadata"}
            assert isinstance(result["id"], str)
            assert isinstance(result["score"], float)
            assert isinstance(result["metadata"], dict)

    def test_query_respects_top_k(self, store: FakeVectorStore) -> None:
        store.upsert(SAMPLE_RECORDS)
        assert len(store.query([1.0, 0.0, 0.0, 0.0], top_k=2)) == 2
        assert len(store.query([1.0, 0.0, 0.0, 0.0], top_k=10)) == 3

    def test_query_returns_most_similar_first(self, store: FakeVectorStore) -> None:
        store.upsert(SAMPLE_RECORDS)
        results = store.query([0.9, 0.1, 0.0, 0.0], top_k=3)
        assert results[0]["id"] == "c1"
        scores = [result["score"] for result in results]
        assert scores == sorted(scores, reverse=True)

    def test_query_applies_metadata_filters(self, store: FakeVectorStore) -> None:
        store.upsert(SAMPLE_RECORDS)
        results = store.query([1.0, 1.0, 1.0, 1.0], top_k=10, filters={"lang": "zh"})
        assert [result["id"] for result in results] == ["c3"]

    def test_query_accepts_optional_trace(self, store: FakeVectorStore) -> None:
        store.upsert(SAMPLE_RECORDS)
        results = store.query([1.0, 0.0, 0.0, 0.0], top_k=1, trace=object())
        assert len(results) == 1

    def test_validate_query_rejects_empty_vector(self) -> None:
        with pytest.raises(VectorStoreError, match="vector must be a non-empty list"):
            BaseVectorStore.validate_query_args([], 5)

    def test_validate_query_rejects_non_numeric_vector(self) -> None:
        with pytest.raises(VectorStoreError, match="only numbers"):
            BaseVectorStore.validate_query_args(["x"], 5)

    def test_validate_query_rejects_non_positive_top_k(self) -> None:
        with pytest.raises(VectorStoreError, match="top_k must be a positive integer"):
            BaseVectorStore.validate_query_args([1.0], 0)
        with pytest.raises(VectorStoreError, match="top_k must be a positive integer"):
            BaseVectorStore.validate_query_args([1.0], -3)

    def test_validate_query_rejects_bool_top_k(self) -> None:
        with pytest.raises(VectorStoreError, match="top_k must be a positive integer"):
            BaseVectorStore.validate_query_args([1.0], True)

    def test_validate_query_rejects_non_dict_filters(self) -> None:
        with pytest.raises(VectorStoreError, match="filters must be a dict"):
            BaseVectorStore.validate_query_args([1.0], 5, filters=["lang"])
