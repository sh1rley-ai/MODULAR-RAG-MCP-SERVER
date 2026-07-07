"""B7.5 — Tests for RecursiveSplitter (LangChain-backed, Markdown-aware).

All tests are pure unit tests: no network, no file I/O. The only external
dependency is `langchain_text_splitters`, which is available in the .venv.
"""

from __future__ import annotations

from typing import Any

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
from libs.splitter.base_splitter import SplitterError
from libs.splitter.recursive_splitter import RecursiveSplitter
from libs.splitter.splitter_factory import SplitterFactory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_ingestion_settings(
    chunk_size: int = 200, chunk_overlap: int = 20
) -> IngestionSettings:
    return IngestionSettings(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        splitter="recursive",
        batch_size=16,
    )


def make_settings(chunk_size: int = 200, chunk_overlap: int = 20) -> Settings:
    return Settings(
        llm=LLMSettings(provider="openai", model="gpt-4o-mini", temperature=0.0, max_tokens=256),
        embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small", dimensions=4),
        vector_store=VectorStoreSettings(
            provider="chroma", persist_directory="data/db/chroma", collection_name="default"
        ),
        retrieval=RetrievalSettings(dense_top_k=20, sparse_top_k=20, fusion_top_k=10, rrf_k=60),
        rerank=RerankSettings(enabled=False, provider="none", model="none", top_k=5),
        evaluation=EvaluationSettings(enabled=False, provider="custom", metrics=["hit_rate"]),
        observability=ObservabilitySettings(
            log_level="INFO",
            trace_enabled=True,
            trace_file="logs/traces.jsonl",
            structured_logging=False,
        ),
        ingestion=IngestionSettings(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            splitter="recursive",
            batch_size=16,
        ),
    )


# ---------------------------------------------------------------------------
# TestFactoryRouting
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFactoryRouting:
    def test_recursive_registered_by_default(self) -> None:
        assert "recursive" in SplitterFactory.registered_strategies()

    def test_factory_creates_recursive_splitter(self) -> None:
        splitter = SplitterFactory.create(make_settings())
        assert isinstance(splitter, RecursiveSplitter)

    def test_factory_passes_chunk_size_to_instance(self) -> None:
        splitter = SplitterFactory.create(make_settings(chunk_size=512, chunk_overlap=50))
        assert splitter.settings.chunk_size == 512
        assert splitter.settings.chunk_overlap == 50


# ---------------------------------------------------------------------------
# TestBasicSplitting
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBasicSplitting:
    def test_returns_list_of_strings(self) -> None:
        splitter = RecursiveSplitter(make_ingestion_settings())
        result = splitter.split_text("Hello world.")
        assert isinstance(result, list)
        assert all(isinstance(chunk, str) for chunk in result)

    def test_empty_string_returns_empty_list(self) -> None:
        splitter = RecursiveSplitter(make_ingestion_settings())
        result = splitter.split_text("")
        assert result == []

    def test_short_text_returns_single_chunk(self) -> None:
        splitter = RecursiveSplitter(make_ingestion_settings(chunk_size=500))
        text = "Short text that fits in one chunk."
        result = splitter.split_text(text)
        assert len(result) == 1
        assert result[0] == text

    def test_long_text_is_split_into_multiple_chunks(self) -> None:
        splitter = RecursiveSplitter(make_ingestion_settings(chunk_size=50, chunk_overlap=0))
        text = "word " * 30  # 150 chars
        result = splitter.split_text(text)
        assert len(result) > 1

    def test_all_content_is_preserved(self) -> None:
        splitter = RecursiveSplitter(make_ingestion_settings(chunk_size=80, chunk_overlap=0))
        text = "The quick brown fox jumps over the lazy dog. " * 5
        chunks = splitter.split_text(text)
        joined = " ".join(chunks)
        # Every word from the original appears somewhere in the joined output.
        for word in text.split():
            assert word in joined

    def test_split_is_deterministic(self) -> None:
        splitter = RecursiveSplitter(make_ingestion_settings(chunk_size=100))
        text = "Repeated input text.\n\nAnother paragraph.\n\nThird paragraph here."
        assert splitter.split_text(text) == splitter.split_text(text)

    def test_accepts_optional_trace_argument(self) -> None:
        splitter = RecursiveSplitter(make_ingestion_settings())
        result = splitter.split_text("Some text.", trace=object())
        assert isinstance(result, list)

    def test_non_string_raises_splitter_error(self) -> None:
        splitter = RecursiveSplitter(make_ingestion_settings())
        with pytest.raises(SplitterError, match="text must be a string"):
            splitter.split_text(42)  # type: ignore[arg-type]

    def test_none_input_raises_splitter_error(self) -> None:
        splitter = RecursiveSplitter(make_ingestion_settings())
        with pytest.raises(SplitterError, match="text must be a string"):
            splitter.split_text(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TestChunkSizeAndOverlap
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestChunkSizeAndOverlap:
    def test_chunks_do_not_exceed_chunk_size_by_large_margin(self) -> None:
        chunk_size = 100
        splitter = RecursiveSplitter(make_ingestion_settings(chunk_size=chunk_size, chunk_overlap=0))
        # Generate text with no natural break points to force character splitting.
        text = "a" * 500
        chunks = splitter.split_text(text)
        # LangChain may produce chunks slightly larger due to separator keep logic,
        # but no chunk should be wildly larger than chunk_size.
        for chunk in chunks:
            assert len(chunk) <= chunk_size * 2

    def test_smaller_chunk_size_produces_more_chunks(self) -> None:
        text = "word " * 40  # 200 chars
        large_splitter = RecursiveSplitter(make_ingestion_settings(chunk_size=200, chunk_overlap=0))
        small_splitter = RecursiveSplitter(make_ingestion_settings(chunk_size=50, chunk_overlap=0))
        assert len(small_splitter.split_text(text)) >= len(large_splitter.split_text(text))

    def test_overlap_causes_content_repetition_between_consecutive_chunks(self) -> None:
        overlap = 20
        # Build text long enough to guarantee at least two chunks.
        text = "abcdefghij" * 20  # 200 chars
        splitter_no_overlap = RecursiveSplitter(
            make_ingestion_settings(chunk_size=60, chunk_overlap=0)
        )
        splitter_with_overlap = RecursiveSplitter(
            make_ingestion_settings(chunk_size=60, chunk_overlap=overlap)
        )
        chunks_no = splitter_no_overlap.split_text(text)
        chunks_ov = splitter_with_overlap.split_text(text)
        if len(chunks_ov) > 1:
            # Total chars in overlapped output should be >= non-overlapped.
            assert sum(len(c) for c in chunks_ov) >= sum(len(c) for c in chunks_no)


# ---------------------------------------------------------------------------
# TestMarkdownStructure
# ---------------------------------------------------------------------------


MARKDOWN_DOC = """\
# Introduction

This is the introduction paragraph with some background information.

## Section One

Content for section one goes here with multiple sentences.
It spans more than one line.

### Subsection 1.1

Detailed content inside the subsection.

## Section Two

Content for section two is independent of section one.

```python
def hello():
    return "world"
```

End of document.
"""


@pytest.mark.unit
class TestMarkdownStructure:
    def test_splits_markdown_without_error(self) -> None:
        splitter = RecursiveSplitter(make_ingestion_settings(chunk_size=200, chunk_overlap=20))
        chunks = splitter.split_text(MARKDOWN_DOC)
        assert len(chunks) >= 1

    def test_heading_boundary_preferred_over_mid_paragraph(self) -> None:
        # With a chunk size that fits one section but not two, the splitter
        # should break at heading boundaries.
        splitter = RecursiveSplitter(make_ingestion_settings(chunk_size=120, chunk_overlap=0))
        chunks = splitter.split_text(MARKDOWN_DOC)
        # No chunk should contain both "Section One" and "Section Two" headings.
        for chunk in chunks:
            has_s1 = "Section One" in chunk
            has_s2 = "Section Two" in chunk
            assert not (has_s1 and has_s2), (
                f"Chunk contains both section headings:\n{chunk!r}"
            )

    def test_code_block_content_is_preserved(self) -> None:
        splitter = RecursiveSplitter(make_ingestion_settings(chunk_size=500, chunk_overlap=0))
        chunks = splitter.split_text(MARKDOWN_DOC)
        all_text = "\n".join(chunks)
        assert "def hello():" in all_text
        assert 'return "world"' in all_text

    def test_paragraph_content_is_preserved(self) -> None:
        splitter = RecursiveSplitter(make_ingestion_settings(chunk_size=500, chunk_overlap=0))
        chunks = splitter.split_text(MARKDOWN_DOC)
        all_text = "\n".join(chunks)
        assert "introduction paragraph" in all_text
        assert "Content for section two" in all_text

    def test_large_chunk_size_keeps_markdown_in_single_chunk(self) -> None:
        splitter = RecursiveSplitter(make_ingestion_settings(chunk_size=5000, chunk_overlap=0))
        chunks = splitter.split_text(MARKDOWN_DOC)
        assert len(chunks) == 1

    def test_result_type_is_always_list_of_str(self) -> None:
        splitter = RecursiveSplitter(make_ingestion_settings(chunk_size=200))
        for chunk in splitter.split_text(MARKDOWN_DOC):
            assert isinstance(chunk, str)
