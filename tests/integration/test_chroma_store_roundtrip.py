"""B7.6 — Integration tests for ChromaStore (ChromaDB-backed VectorStore).

All tests use real ChromaDB (no mocks). The ephemeral client is used for
speed; persist_directory=":memory:" triggers EphemeralClient so no
temp-dir cleanup is needed.  One class also exercises PersistentClient
with pytest's tmp_path fixture to verify on-disk storage.

Each test gets a unique collection name via the `col` fixture to prevent
cross-test data leakage (EphemeralClient shares process-wide state in
ChromaDB 1.5+).
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List

import pytest

from core.settings import VectorStoreSettings
from libs.vector_store.base_vector_store import VectorStoreError
from libs.vector_store.chroma_store import ChromaStore
from libs.vector_store.vector_store_factory import VectorStoreFactory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 4


@pytest.fixture
def col() -> str:
    """Unique collection name per test — prevents cross-test data leakage."""
    return f"t_{uuid.uuid4().hex}"


def make_settings(persist_directory: str = ":memory:", collection_name: str = "default") -> VectorStoreSettings:
    return VectorStoreSettings(
        provider="chroma",
        persist_directory=persist_directory,
        collection_name=collection_name,
    )


def fresh_store(col_name: str, persist_directory: str = ":memory:") -> ChromaStore:
    return ChromaStore(make_settings(persist_directory, col_name))


def vec(seed: float, dim: int = DIM) -> List[float]:
    return [seed] * dim


def record(
    record_id: str,
    seed: float,
    text: str = "",
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "id": record_id,
        "vector": vec(seed),
        "text": text,
        "metadata": metadata or {},
    }


# ---------------------------------------------------------------------------
# TestFactoryRouting
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestFactoryRouting:
    def test_chroma_registered_by_default(self) -> None:
        assert "chroma" in VectorStoreFactory.registered_providers()

    def test_factory_creates_chroma_store(self, col: str) -> None:
        from core.settings import (
            EmbeddingSettings,
            EvaluationSettings,
            LLMSettings,
            ObservabilitySettings,
            RerankSettings,
            RetrievalSettings,
            Settings,
        )

        settings = Settings(
            llm=LLMSettings(provider="openai", model="gpt-4o-mini", temperature=0.0, max_tokens=256),
            embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small", dimensions=4),
            vector_store=make_settings(collection_name=col),
            retrieval=RetrievalSettings(dense_top_k=20, sparse_top_k=20, fusion_top_k=10, rrf_k=60),
            rerank=RerankSettings(enabled=False, provider="none", model="none", top_k=5),
            evaluation=EvaluationSettings(enabled=False, provider="custom", metrics=["hit_rate"]),
            observability=ObservabilitySettings(
                log_level="INFO", trace_enabled=True, trace_file="logs/traces.jsonl", structured_logging=False
            ),
        )
        store = VectorStoreFactory.create(settings)
        assert isinstance(store, ChromaStore)


# ---------------------------------------------------------------------------
# TestBasicUpsertQuery
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestBasicUpsertQuery:
    def test_upsert_and_query_returns_result(self, col: str) -> None:
        store = fresh_store(col)
        store.upsert([record("a", 1.0, "hello")])
        results = store.query(vec(1.0), top_k=1)
        assert len(results) == 1
        assert results[0]["id"] == "a"

    def test_query_result_has_expected_fields(self, col: str) -> None:
        store = fresh_store(col)
        store.upsert([record("r1", 0.5, "some text")])
        result = store.query(vec(0.5), top_k=1)[0]
        assert "id" in result
        assert "score" in result
        assert "text" in result
        assert "metadata" in result

    def test_query_score_is_between_zero_and_one(self, col: str) -> None:
        store = fresh_store(col)
        store.upsert([record("r1", 1.0)])
        result = store.query(vec(1.0), top_k=1)[0]
        assert 0.0 < result["score"] <= 1.0

    def test_exact_match_has_highest_score(self, col: str) -> None:
        store = fresh_store(col)
        store.upsert([
            record("close", 0.5, "close"),
            record("exact", 1.0, "exact"),
        ])
        results = store.query(vec(1.0), top_k=2)
        assert results[0]["id"] == "exact"
        assert results[0]["score"] > results[1]["score"]

    def test_text_is_preserved_in_results(self, col: str) -> None:
        store = fresh_store(col)
        store.upsert([record("doc1", 0.3, "retrieved text content")])
        result = store.query(vec(0.3), top_k=1)[0]
        assert result["text"] == "retrieved text content"

    def test_upsert_multiple_records(self, col: str) -> None:
        store = fresh_store(col)
        records = [record(f"r{i}", float(i) * 0.1, f"doc {i}") for i in range(5)]
        store.upsert(records)
        results = store.query(vec(0.4), top_k=5)
        assert len(results) == 5

    def test_results_ordered_most_similar_first(self, col: str) -> None:
        store = fresh_store(col)
        store.upsert([
            record("far", 0.0, "far"),
            record("near", 1.0, "near"),
        ])
        results = store.query(vec(1.0), top_k=2)
        assert results[0]["id"] == "near"
        assert results[1]["id"] == "far"


# ---------------------------------------------------------------------------
# TestTopK
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestTopK:
    def test_top_k_limits_number_of_results(self, col: str) -> None:
        store = fresh_store(col)
        store.upsert([record(f"r{i}", float(i) * 0.1) for i in range(10)])
        results = store.query(vec(0.5), top_k=3)
        assert len(results) == 3

    def test_top_k_one_returns_single_result(self, col: str) -> None:
        store = fresh_store(col)
        store.upsert([record("a", 1.0), record("b", 0.5)])
        results = store.query(vec(1.0), top_k=1)
        assert len(results) == 1

    def test_top_k_larger_than_collection_returns_all(self, col: str) -> None:
        store = fresh_store(col)
        store.upsert([record("a", 1.0), record("b", 0.5)])
        results = store.query(vec(1.0), top_k=100)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# TestMetadataFilters
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestMetadataFilters:
    def test_filter_returns_only_matching_records(self, col: str) -> None:
        store = fresh_store(col)
        store.upsert([
            record("match", 1.0, "match", {"type": "A"}),
            record("no_match", 0.5, "no match", {"type": "B"}),
        ])
        results = store.query(vec(1.0), top_k=5, filters={"type": "A"})
        ids = [r["id"] for r in results]
        assert "match" in ids
        assert "no_match" not in ids

    def test_filter_excludes_non_matching_records(self, col: str) -> None:
        store = fresh_store(col)
        store.upsert([
            record("x", 1.0, "x", {"category": "tech"}),
            record("y", 0.7, "y", {"category": "science"}),
            record("z", 0.9, "z", {"category": "tech"}),
        ])
        results = store.query(vec(1.0), top_k=5, filters={"category": "tech"})
        ids = [r["id"] for r in results]
        assert "x" in ids
        assert "z" in ids
        assert "y" not in ids

    def test_no_filter_returns_all_results(self, col: str) -> None:
        store = fresh_store(col)
        store.upsert([
            record("a", 1.0, "a", {"tag": "x"}),
            record("b", 0.5, "b", {"tag": "y"}),
        ])
        results = store.query(vec(1.0), top_k=5)
        assert len(results) == 2

    def test_metadata_preserved_in_query_results(self, col: str) -> None:
        store = fresh_store(col)
        store.upsert([record("r1", 1.0, "text", {"source": "doc.pdf", "page": "3"})])
        result = store.query(vec(1.0), top_k=1)[0]
        assert result["metadata"].get("source") == "doc.pdf"


# ---------------------------------------------------------------------------
# TestIdempotentUpsert
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestIdempotentUpsert:
    def test_upsert_same_id_overwrites(self, col: str) -> None:
        store = fresh_store(col)
        store.upsert([record("dup", 1.0, "original")])
        store.upsert([record("dup", 1.0, "updated")])
        results = store.query(vec(1.0), top_k=5)
        assert len(results) == 1
        assert results[0]["text"] == "updated"

    def test_upsert_twice_does_not_duplicate(self, col: str) -> None:
        store = fresh_store(col)
        r = record("single", 0.5, "text")
        store.upsert([r])
        store.upsert([r])
        results = store.query(vec(0.5), top_k=100)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# TestPersistentClient
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestPersistentClient:
    def test_data_survives_new_client_instance(self, tmp_path: Path, col: str) -> None:
        persist_dir = str(tmp_path / "chroma_db")
        store1 = fresh_store(col, persist_directory=persist_dir)
        store1.upsert([record("persisted", 1.0, "stored text")])

        store2 = fresh_store(col, persist_directory=persist_dir)
        results = store2.query(vec(1.0), top_k=1)
        assert results[0]["id"] == "persisted"
        assert results[0]["text"] == "stored text"

    def test_different_collections_are_isolated(self, tmp_path: Path) -> None:
        persist_dir = str(tmp_path / "chroma_db")
        col_a = f"col_a_{uuid.uuid4().hex}"
        col_b = f"col_b_{uuid.uuid4().hex}"
        store_a = fresh_store(col_a, persist_directory=persist_dir)
        store_b = fresh_store(col_b, persist_directory=persist_dir)

        store_a.upsert([record("only_in_a", 1.0, "a")])
        results_b = store_b.query(vec(1.0), top_k=10)
        ids_b = [r["id"] for r in results_b]
        assert "only_in_a" not in ids_b


# ---------------------------------------------------------------------------
# TestValidation
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestValidation:
    def test_upsert_empty_list_raises_vector_store_error(self, col: str) -> None:
        store = fresh_store(col)
        with pytest.raises(VectorStoreError, match="non-empty list"):
            store.upsert([])

    def test_upsert_missing_id_raises_readable_error(self, col: str) -> None:
        store = fresh_store(col)
        with pytest.raises(VectorStoreError, match="id must be"):
            store.upsert([{"id": "", "vector": vec(1.0)}])

    def test_query_empty_vector_raises_readable_error(self, col: str) -> None:
        store = fresh_store(col)
        with pytest.raises(VectorStoreError, match="non-empty list"):
            store.query([], top_k=5)

    def test_query_invalid_top_k_raises_readable_error(self, col: str) -> None:
        store = fresh_store(col)
        with pytest.raises(VectorStoreError, match="positive integer"):
            store.query(vec(1.0), top_k=0)
