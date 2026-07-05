"""B3 — Tests for BaseSplitter abstraction and SplitterFactory routing.

All tests are pure unit tests: fake splitters are registered in-test to
verify factory routing. Fake splitters produce deterministic fragments so
behaviour can be asserted without LangChain or any external dependency.
"""

from __future__ import annotations

from typing import Any, List

import pytest

from core.settings import (
    EmbeddingSettings,
    EvaluationSettings,
    IngestionSettings,
    LLMSettings,
    ObservabilitySettings,
    RerankSettings,
    RetrievalSettings,
    Settings,
    VectorStoreSettings,
)
from libs.splitter.base_splitter import BaseSplitter, SplitterError
from libs.splitter.splitter_factory import SplitterFactory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_settings(
    splitter: str = "fake",
    chunk_size: int = 100,
    chunk_overlap: int = 20,
    with_ingestion: bool = True,
) -> Settings:
    """Build a full Settings object with the given ingestion.splitter strategy."""
    return Settings(
        llm=LLMSettings(provider="openai", model="gpt-4o-mini", temperature=0.0, max_tokens=256),
        embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small", dimensions=4),
        vector_store=VectorStoreSettings(provider="chroma", persist_directory="data/db/chroma", collection_name="default"),
        retrieval=RetrievalSettings(dense_top_k=20, sparse_top_k=20, fusion_top_k=10, rrf_k=60),
        rerank=RerankSettings(enabled=False, provider="none", model="none", top_k=5),
        evaluation=EvaluationSettings(enabled=False, provider="custom", metrics=["hit_rate"]),
        observability=ObservabilitySettings(
            log_level="INFO", trace_enabled=True, trace_file="logs/traces.jsonl", structured_logging=False
        ),
        ingestion=IngestionSettings(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, splitter=splitter, batch_size=16
        )
        if with_ingestion
        else None,
    )


class FakeSplitter(BaseSplitter):
    """In-test stub strategy: splits text into fixed-size pieces of chunk_size chars."""

    def __init__(self, ingestion_settings: IngestionSettings) -> None:
        self.settings = ingestion_settings

    def split_text(self, text: str, trace: Any | None = None) -> List[str]:
        self.validate_text(text)
        size = self.settings.chunk_size
        return [text[i : i + size] for i in range(0, len(text), size)]


class AnotherFakeSplitter(FakeSplitter):
    """Second stub strategy to verify routing between multiple strategies."""


@pytest.fixture(autouse=True)
def clean_registry():
    """Snapshot and restore the factory registry around each test."""
    snapshot = dict(SplitterFactory._registry)
    SplitterFactory._registry.clear()
    yield
    SplitterFactory._registry.clear()
    SplitterFactory._registry.update(snapshot)


# ---------------------------------------------------------------------------
# TestSplitterFactoryRouting
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSplitterFactoryRouting:
    """Factory routes settings.ingestion.splitter to the registered implementation."""

    def test_create_returns_registered_strategy(self) -> None:
        SplitterFactory.register("fake", FakeSplitter)
        splitter = SplitterFactory.create(make_settings(splitter="fake"))
        assert isinstance(splitter, FakeSplitter)

    def test_created_instance_receives_ingestion_settings(self) -> None:
        SplitterFactory.register("fake", FakeSplitter)
        splitter = SplitterFactory.create(make_settings(splitter="fake", chunk_size=42, chunk_overlap=7))
        assert splitter.settings.chunk_size == 42
        assert splitter.settings.chunk_overlap == 7

    def test_routes_between_multiple_strategies(self) -> None:
        SplitterFactory.register("fake", FakeSplitter)
        SplitterFactory.register("another", AnotherFakeSplitter)
        assert isinstance(SplitterFactory.create(make_settings(splitter="another")), AnotherFakeSplitter)
        splitter = SplitterFactory.create(make_settings(splitter="fake"))
        assert isinstance(splitter, FakeSplitter)
        assert not isinstance(splitter, AnotherFakeSplitter)

    def test_strategy_name_is_case_insensitive(self) -> None:
        SplitterFactory.register("Fake", FakeSplitter)
        splitter = SplitterFactory.create(make_settings(splitter="FAKE"))
        assert isinstance(splitter, FakeSplitter)

    def test_unknown_strategy_raises_readable_error(self) -> None:
        SplitterFactory.register("fake", FakeSplitter)
        with pytest.raises(ValueError, match="Unknown splitter strategy: 'nonexistent'"):
            SplitterFactory.create(make_settings(splitter="nonexistent"))

    def test_unknown_strategy_error_lists_registered(self) -> None:
        SplitterFactory.register("fake", FakeSplitter)
        with pytest.raises(ValueError, match="fake"):
            SplitterFactory.create(make_settings(splitter="nonexistent"))

    def test_missing_ingestion_section_raises_readable_error(self) -> None:
        SplitterFactory.register("fake", FakeSplitter)
        with pytest.raises(ValueError, match="settings.ingestion is missing"):
            SplitterFactory.create(make_settings(with_ingestion=False))

    def test_unregister_removes_strategy(self) -> None:
        SplitterFactory.register("fake", FakeSplitter)
        SplitterFactory.unregister("fake")
        with pytest.raises(ValueError, match="Unknown splitter strategy"):
            SplitterFactory.create(make_settings(splitter="fake"))


# ---------------------------------------------------------------------------
# TestSplitterFactoryRegistration
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSplitterFactoryRegistration:
    """register() validates its inputs."""

    def test_register_rejects_non_basesplitter_class(self) -> None:
        class NotASplitter:
            pass

        with pytest.raises(TypeError, match="BaseSplitter"):
            SplitterFactory.register("bad", NotASplitter)  # type: ignore[arg-type]

    def test_register_rejects_empty_strategy_name(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            SplitterFactory.register("  ", FakeSplitter)

    def test_registered_strategies_lists_names(self) -> None:
        SplitterFactory.register("fake", FakeSplitter)
        SplitterFactory.register("another", AnotherFakeSplitter)
        assert SplitterFactory.registered_strategies() == ["another", "fake"]


# ---------------------------------------------------------------------------
# TestBaseSplitterContract
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBaseSplitterContract:
    """BaseSplitter abstract contract and input validation."""

    def test_base_splitter_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError):
            BaseSplitter()  # type: ignore[abstract]

    def test_split_text_returns_list_of_strings(self) -> None:
        splitter = FakeSplitter(make_settings(chunk_size=5).ingestion)
        fragments = splitter.split_text("hello world")
        assert isinstance(fragments, list)
        assert all(isinstance(fragment, str) for fragment in fragments)

    def test_split_text_respects_configured_chunk_size(self) -> None:
        splitter = FakeSplitter(make_settings(chunk_size=4).ingestion)
        fragments = splitter.split_text("abcdefgh")
        assert fragments == ["abcd", "efgh"]

    def test_chunk_size_change_changes_output(self) -> None:
        text = "abcdefghij"
        small = FakeSplitter(make_settings(chunk_size=2).ingestion).split_text(text)
        large = FakeSplitter(make_settings(chunk_size=5).ingestion).split_text(text)
        assert len(small) == 5
        assert len(large) == 2

    def test_split_text_preserves_order(self) -> None:
        splitter = FakeSplitter(make_settings(chunk_size=3).ingestion)
        fragments = splitter.split_text("abcdefghi")
        assert "".join(fragments) == "abcdefghi"

    def test_split_text_is_deterministic(self) -> None:
        splitter = FakeSplitter(make_settings(chunk_size=3).ingestion)
        assert splitter.split_text("same text") == splitter.split_text("same text")

    def test_split_text_accepts_optional_trace(self) -> None:
        splitter = FakeSplitter(make_settings(chunk_size=5).ingestion)
        fragments = splitter.split_text("hello", trace=object())
        assert fragments == ["hello"]

    def test_validate_text_rejects_non_string(self) -> None:
        with pytest.raises(SplitterError, match="text must be a string"):
            BaseSplitter.validate_text(42)

    def test_validate_text_accepts_empty_string(self) -> None:
        BaseSplitter.validate_text("")
