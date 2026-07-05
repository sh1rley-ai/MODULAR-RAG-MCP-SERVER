"""B2 — Tests for BaseEmbedding abstraction and EmbeddingFactory routing.

All tests are pure unit tests: fake providers are registered in-test to
verify factory routing. Fake embeddings return stable vectors so batch
behaviour can be asserted deterministically. No network, no external services.
"""

from __future__ import annotations

import hashlib
from typing import Any, List

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
from libs.embedding.base_embedding import BaseEmbedding, EmbeddingError
from libs.embedding.embedding_factory import EmbeddingFactory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_settings(provider: str = "fake", model: str = "fake-model", dimensions: int = 4) -> Settings:
    """Build a full Settings object with the given embedding provider."""
    return Settings(
        llm=LLMSettings(provider="openai", model="gpt-4o-mini", temperature=0.0, max_tokens=256),
        embedding=EmbeddingSettings(provider=provider, model=model, dimensions=dimensions),
        vector_store=VectorStoreSettings(provider="chroma", persist_directory="data/db/chroma", collection_name="default"),
        retrieval=RetrievalSettings(dense_top_k=20, sparse_top_k=20, fusion_top_k=10, rrf_k=60),
        rerank=RerankSettings(enabled=False, provider="none", model="none", top_k=5),
        evaluation=EvaluationSettings(enabled=False, provider="custom", metrics=["hit_rate"]),
        observability=ObservabilitySettings(
            log_level="INFO", trace_enabled=True, trace_file="logs/traces.jsonl", structured_logging=False
        ),
    )


class FakeEmbedding(BaseEmbedding):
    """In-test stub provider: returns a stable vector derived from each text."""

    def __init__(self, embedding_settings: EmbeddingSettings) -> None:
        self.settings = embedding_settings

    def embed(self, texts: List[str], trace: Any | None = None) -> List[List[float]]:
        self.validate_texts(texts)
        dimensions = self.settings.dimensions
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vectors.append([digest[i] / 255.0 for i in range(dimensions)])
        return vectors


class AnotherFakeEmbedding(FakeEmbedding):
    """Second stub provider to verify routing between multiple providers."""


@pytest.fixture(autouse=True)
def clean_registry():
    """Snapshot and restore the factory registry around each test."""
    snapshot = dict(EmbeddingFactory._registry)
    EmbeddingFactory._registry.clear()
    yield
    EmbeddingFactory._registry.clear()
    EmbeddingFactory._registry.update(snapshot)


# ---------------------------------------------------------------------------
# TestEmbeddingFactoryRouting
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEmbeddingFactoryRouting:
    """Factory routes settings.embedding.provider to the registered implementation."""

    def test_create_returns_registered_provider(self) -> None:
        EmbeddingFactory.register("fake", FakeEmbedding)
        embedding = EmbeddingFactory.create(make_settings(provider="fake"))
        assert isinstance(embedding, FakeEmbedding)

    def test_created_instance_receives_embedding_settings(self) -> None:
        EmbeddingFactory.register("fake", FakeEmbedding)
        embedding = EmbeddingFactory.create(make_settings(provider="fake", model="my-model", dimensions=8))
        assert embedding.settings.model == "my-model"
        assert embedding.settings.dimensions == 8

    def test_routes_between_multiple_providers(self) -> None:
        EmbeddingFactory.register("fake", FakeEmbedding)
        EmbeddingFactory.register("another", AnotherFakeEmbedding)
        assert isinstance(EmbeddingFactory.create(make_settings(provider="another")), AnotherFakeEmbedding)
        embedding = EmbeddingFactory.create(make_settings(provider="fake"))
        assert isinstance(embedding, FakeEmbedding)
        assert not isinstance(embedding, AnotherFakeEmbedding)

    def test_provider_name_is_case_insensitive(self) -> None:
        EmbeddingFactory.register("Fake", FakeEmbedding)
        embedding = EmbeddingFactory.create(make_settings(provider="FAKE"))
        assert isinstance(embedding, FakeEmbedding)

    def test_unknown_provider_raises_readable_error(self) -> None:
        EmbeddingFactory.register("fake", FakeEmbedding)
        with pytest.raises(ValueError, match="Unknown embedding provider: 'nonexistent'"):
            EmbeddingFactory.create(make_settings(provider="nonexistent"))

    def test_unknown_provider_error_lists_registered(self) -> None:
        EmbeddingFactory.register("fake", FakeEmbedding)
        with pytest.raises(ValueError, match="fake"):
            EmbeddingFactory.create(make_settings(provider="nonexistent"))

    def test_unregister_removes_provider(self) -> None:
        EmbeddingFactory.register("fake", FakeEmbedding)
        EmbeddingFactory.unregister("fake")
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            EmbeddingFactory.create(make_settings(provider="fake"))


# ---------------------------------------------------------------------------
# TestEmbeddingFactoryRegistration
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEmbeddingFactoryRegistration:
    """register() validates its inputs."""

    def test_register_rejects_non_baseembedding_class(self) -> None:
        class NotAnEmbedding:
            pass

        with pytest.raises(TypeError, match="BaseEmbedding"):
            EmbeddingFactory.register("bad", NotAnEmbedding)  # type: ignore[arg-type]

    def test_register_rejects_empty_provider_name(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            EmbeddingFactory.register("  ", FakeEmbedding)

    def test_registered_providers_lists_names(self) -> None:
        EmbeddingFactory.register("fake", FakeEmbedding)
        EmbeddingFactory.register("another", AnotherFakeEmbedding)
        assert EmbeddingFactory.registered_providers() == ["another", "fake"]


# ---------------------------------------------------------------------------
# TestBaseEmbeddingContract
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBaseEmbeddingContract:
    """BaseEmbedding abstract contract and batch input validation."""

    def test_base_embedding_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError):
            BaseEmbedding()  # type: ignore[abstract]

    def test_embed_returns_one_vector_per_text(self) -> None:
        embedding = FakeEmbedding(make_settings().embedding)
        vectors = embedding.embed(["hello", "world", "foo"])
        assert len(vectors) == 3

    def test_embed_vectors_have_configured_dimensions(self) -> None:
        embedding = FakeEmbedding(make_settings(dimensions=4).embedding)
        vectors = embedding.embed(["hello", "world"])
        assert all(len(vector) == 4 for vector in vectors)

    def test_embed_is_deterministic(self) -> None:
        embedding = FakeEmbedding(make_settings().embedding)
        assert embedding.embed(["same text"]) == embedding.embed(["same text"])

    def test_embed_preserves_input_order(self) -> None:
        embedding = FakeEmbedding(make_settings().embedding)
        batch = embedding.embed(["a", "b"])
        assert batch[0] == embedding.embed(["a"])[0]
        assert batch[1] == embedding.embed(["b"])[0]

    def test_embed_accepts_optional_trace(self) -> None:
        embedding = FakeEmbedding(make_settings().embedding)
        vectors = embedding.embed(["hello"], trace=object())
        assert len(vectors) == 1

    def test_validate_texts_rejects_empty_list(self) -> None:
        with pytest.raises(EmbeddingError, match="non-empty list"):
            BaseEmbedding.validate_texts([])

    def test_validate_texts_rejects_non_list(self) -> None:
        with pytest.raises(EmbeddingError, match="non-empty list"):
            BaseEmbedding.validate_texts("hello")

    def test_validate_texts_rejects_non_string_items(self) -> None:
        with pytest.raises(EmbeddingError, match=r"texts\[1\] must be a string"):
            BaseEmbedding.validate_texts(["ok", 42])
