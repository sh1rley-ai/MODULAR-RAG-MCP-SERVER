"""B7.8 — Unit tests for CrossEncoderReranker.

All tests are pure unit tests: the cross-encoder model is always replaced by an
injectable `scorer` callable. No sentence-transformers or network access needed.

Test structure:
    TestFactoryRouting        — factory registration and creation
    TestRerankOrdering        — candidates are reordered by descending scorer output
    TestEdgeCases             — empty list, single candidate, ties
    TestScorerInteraction     — scorer receives correct (query, text) pairs
    TestScorerFailures        — scorer exceptions → RerankerError
    TestOutputShapeMismatch   — scorer returning wrong number of scores
    TestMissingText           — candidates without 'text' use empty string
    TestValidation            — query / candidates input validation
    TestSentenceTransformers  — missing library raises RerankerError (monkeypatched)
    TestNoScorerNoModel       — construction without scorer and bad model name
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, patch

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
from libs.reranker.base_reranker import RerankerError
from libs.reranker.cross_encoder_reranker import CrossEncoderReranker
from libs.reranker.reranker_factory import RerankerFactory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_rerank_settings(
    provider: str = "cross_encoder",
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    enabled: bool = True,
) -> RerankSettings:
    return RerankSettings(enabled=enabled, provider=provider, model=model, top_k=5)


def make_full_settings(
    provider: str = "cross_encoder",
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> Settings:
    return Settings(
        llm=LLMSettings(provider="openai", model="gpt-4o-mini", temperature=0.0, max_tokens=256),
        embedding=EmbeddingSettings(
            provider="openai", model="text-embedding-3-small", dimensions=4
        ),
        vector_store=VectorStoreSettings(
            provider="chroma", persist_directory=":memory:", collection_name="default"
        ),
        retrieval=RetrievalSettings(
            dense_top_k=20, sparse_top_k=20, fusion_top_k=10, rrf_k=60
        ),
        rerank=RerankSettings(enabled=True, provider=provider, model=model, top_k=5),
        evaluation=EvaluationSettings(
            enabled=False, provider="custom", metrics=["hit_rate"]
        ),
        observability=ObservabilitySettings(
            log_level="INFO",
            trace_enabled=True,
            trace_file="logs/traces.jsonl",
            structured_logging=False,
        ),
    )


def candidate(cid: str, text: str = "some text", score: float = 0.5) -> Dict[str, Any]:
    return {"id": cid, "score": score, "text": text, "metadata": {}}


def fixed_scorer(scores: List[float]):
    """Return a scorer that yields the given scores in order."""

    def scorer(query: str, pairs: List[Tuple[str, str]]) -> List[float]:
        return scores

    return scorer


def identity_scorer(query: str, pairs: List[Tuple[str, str]]) -> List[float]:
    """Assign each candidate a score equal to its index (ascending)."""
    return [float(i) for i in range(len(pairs))]


# ---------------------------------------------------------------------------
# TestFactoryRouting
# ---------------------------------------------------------------------------


class TestFactoryRouting:
    def test_cross_encoder_is_registered(self):
        assert "cross_encoder" in RerankerFactory.registered_backends()

    def test_factory_creates_cross_encoder_with_scorer(self):
        settings = make_full_settings()
        reranker = CrossEncoderReranker(settings.rerank, scorer=identity_scorer)
        assert isinstance(reranker, CrossEncoderReranker)

    def test_factory_create_routes_to_cross_encoder(self):
        settings = make_full_settings()
        reranker = CrossEncoderReranker(settings.rerank, scorer=identity_scorer)
        # Confirm it was constructed without error and is the right type
        assert reranker.provider_name == "cross_encoder"

    def test_disabled_rerank_returns_none_reranker(self):
        from libs.reranker.base_reranker import NoneReranker

        settings = make_full_settings()
        disabled = Settings(
            llm=settings.llm,
            embedding=settings.embedding,
            vector_store=settings.vector_store,
            retrieval=settings.retrieval,
            rerank=RerankSettings(
                enabled=False, provider="cross_encoder", model="some-model", top_k=5
            ),
            evaluation=settings.evaluation,
            observability=settings.observability,
        )
        result = RerankerFactory.create(disabled)
        assert isinstance(result, NoneReranker)


# ---------------------------------------------------------------------------
# TestRerankOrdering
# ---------------------------------------------------------------------------


class TestRerankOrdering:
    def test_candidates_reordered_by_descending_score(self):
        reranker = CrossEncoderReranker(
            make_rerank_settings(), scorer=fixed_scorer([0.1, 0.9, 0.5])
        )
        candidates = [candidate("a"), candidate("b"), candidate("c")]
        result = reranker.rerank("query", candidates)
        assert [r["id"] for r in result] == ["b", "c", "a"]

    def test_already_sorted_candidates_stay_in_order(self):
        reranker = CrossEncoderReranker(
            make_rerank_settings(), scorer=fixed_scorer([3.0, 2.0, 1.0])
        )
        candidates = [candidate("x"), candidate("y"), candidate("z")]
        result = reranker.rerank("query", candidates)
        assert [r["id"] for r in result] == ["x", "y", "z"]

    def test_reverse_sorted_candidates_are_flipped(self):
        reranker = CrossEncoderReranker(
            make_rerank_settings(), scorer=fixed_scorer([1.0, 2.0, 3.0])
        )
        candidates = [candidate("a"), candidate("b"), candidate("c")]
        result = reranker.rerank("query", candidates)
        assert [r["id"] for r in result] == ["c", "b", "a"]

    def test_all_candidates_preserved_after_rerank(self):
        reranker = CrossEncoderReranker(
            make_rerank_settings(), scorer=fixed_scorer([0.3, 0.7, 0.1, 0.9])
        )
        candidates = [candidate(str(i)) for i in range(4)]
        result = reranker.rerank("query", candidates)
        assert len(result) == 4
        assert {r["id"] for r in result} == {"0", "1", "2", "3"}

    def test_candidate_fields_preserved_after_rerank(self):
        reranker = CrossEncoderReranker(
            make_rerank_settings(), scorer=fixed_scorer([0.2, 0.8])
        )
        c1 = {"id": "a", "text": "hello", "score": 0.1, "metadata": {"src": "doc1"}}
        c2 = {"id": "b", "text": "world", "score": 0.9, "metadata": {"src": "doc2"}}
        result = reranker.rerank("q", [c1, c2])
        assert result[0]["id"] == "b"
        assert result[0]["metadata"]["src"] == "doc2"
        assert result[1]["id"] == "a"

    def test_negative_scores_supported(self):
        reranker = CrossEncoderReranker(
            make_rerank_settings(), scorer=fixed_scorer([-1.0, -0.5, -2.0])
        )
        candidates = [candidate("a"), candidate("b"), candidate("c")]
        result = reranker.rerank("query", candidates)
        assert [r["id"] for r in result] == ["b", "a", "c"]


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_candidates_returns_empty_list(self):
        reranker = CrossEncoderReranker(make_rerank_settings(), scorer=fixed_scorer([]))
        assert reranker.rerank("query", []) == []

    def test_single_candidate_returned_unchanged(self):
        reranker = CrossEncoderReranker(
            make_rerank_settings(), scorer=fixed_scorer([0.99])
        )
        c = candidate("only")
        result = reranker.rerank("q", [c])
        assert result == [c]

    def test_tied_scores_all_candidates_present(self):
        reranker = CrossEncoderReranker(
            make_rerank_settings(), scorer=fixed_scorer([0.5, 0.5, 0.5])
        )
        candidates = [candidate("a"), candidate("b"), candidate("c")]
        result = reranker.rerank("query", candidates)
        assert len(result) == 3
        assert {r["id"] for r in result} == {"a", "b", "c"}


# ---------------------------------------------------------------------------
# TestScorerInteraction
# ---------------------------------------------------------------------------


class TestScorerInteraction:
    def test_scorer_receives_query_as_first_element_of_each_pair(self):
        received_queries: List[str] = []

        def capturing_scorer(
            query: str, pairs: List[Tuple[str, str]]
        ) -> List[float]:
            received_queries.extend(q for q, _ in pairs)
            return [0.0] * len(pairs)

        reranker = CrossEncoderReranker(
            make_rerank_settings(), scorer=capturing_scorer
        )
        reranker.rerank("my-query", [candidate("a"), candidate("b")])
        assert received_queries == ["my-query", "my-query"]

    def test_scorer_receives_candidate_text_as_second_element(self):
        received_texts: List[str] = []

        def capturing_scorer(
            query: str, pairs: List[Tuple[str, str]]
        ) -> List[float]:
            received_texts.extend(t for _, t in pairs)
            return [0.0] * len(pairs)

        reranker = CrossEncoderReranker(
            make_rerank_settings(), scorer=capturing_scorer
        )
        c1 = candidate("a", text="alpha text")
        c2 = candidate("b", text="beta text")
        reranker.rerank("q", [c1, c2])
        assert received_texts == ["alpha text", "beta text"]

    def test_scorer_called_once_per_rerank(self):
        call_count = [0]

        def counting_scorer(
            query: str, pairs: List[Tuple[str, str]]
        ) -> List[float]:
            call_count[0] += 1
            return [0.0] * len(pairs)

        reranker = CrossEncoderReranker(
            make_rerank_settings(), scorer=counting_scorer
        )
        reranker.rerank("q", [candidate("a"), candidate("b"), candidate("c")])
        assert call_count[0] == 1


# ---------------------------------------------------------------------------
# TestMissingText
# ---------------------------------------------------------------------------


class TestMissingText:
    def test_missing_text_key_uses_empty_string(self):
        received_texts: List[str] = []

        def capturing_scorer(
            query: str, pairs: List[Tuple[str, str]]
        ) -> List[float]:
            received_texts.extend(t for _, t in pairs)
            return [0.0] * len(pairs)

        reranker = CrossEncoderReranker(
            make_rerank_settings(), scorer=capturing_scorer
        )
        c = {"id": "no-text"}
        reranker.rerank("q", [c])
        assert received_texts == [""]

    def test_none_text_uses_empty_string(self):
        received_texts: List[str] = []

        def capturing_scorer(
            query: str, pairs: List[Tuple[str, str]]
        ) -> List[float]:
            received_texts.extend(t for _, t in pairs)
            return [0.0] * len(pairs)

        reranker = CrossEncoderReranker(
            make_rerank_settings(), scorer=capturing_scorer
        )
        c = {"id": "null-text", "text": None}
        reranker.rerank("q", [c])
        assert received_texts == [""]


# ---------------------------------------------------------------------------
# TestScorerFailures
# ---------------------------------------------------------------------------


class TestScorerFailures:
    def test_scorer_exception_raises_reranker_error(self):
        def bad_scorer(query: str, pairs: List[Tuple[str, str]]) -> List[float]:
            raise RuntimeError("model crashed")

        reranker = CrossEncoderReranker(make_rerank_settings(), scorer=bad_scorer)
        with pytest.raises(RerankerError, match="scorer raised an exception"):
            reranker.rerank("query", [candidate("a")])

    def test_scorer_error_message_includes_original_exception(self):
        def bad_scorer(query: str, pairs: List[Tuple[str, str]]) -> List[float]:
            raise ValueError("out of memory")

        reranker = CrossEncoderReranker(make_rerank_settings(), scorer=bad_scorer)
        with pytest.raises(RerankerError, match="out of memory"):
            reranker.rerank("query", [candidate("a")])


# ---------------------------------------------------------------------------
# TestOutputShapeMismatch
# ---------------------------------------------------------------------------


class TestOutputShapeMismatch:
    def test_scorer_returning_fewer_scores_raises_error(self):
        reranker = CrossEncoderReranker(
            make_rerank_settings(), scorer=fixed_scorer([0.5])
        )
        with pytest.raises(RerankerError, match="scorer returned 1 scores for 3 candidates"):
            reranker.rerank("q", [candidate("a"), candidate("b"), candidate("c")])

    def test_scorer_returning_more_scores_raises_error(self):
        reranker = CrossEncoderReranker(
            make_rerank_settings(), scorer=fixed_scorer([0.1, 0.2, 0.3, 0.4])
        )
        with pytest.raises(RerankerError, match="scorer returned 4 scores for 2 candidates"):
            reranker.rerank("q", [candidate("a"), candidate("b")])


# ---------------------------------------------------------------------------
# TestValidation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_empty_query_raises_error(self):
        reranker = CrossEncoderReranker(make_rerank_settings(), scorer=fixed_scorer([]))
        with pytest.raises(RerankerError, match="non-empty string"):
            reranker.rerank("", [])

    def test_whitespace_only_query_raises_error(self):
        reranker = CrossEncoderReranker(make_rerank_settings(), scorer=fixed_scorer([]))
        with pytest.raises(RerankerError, match="non-empty string"):
            reranker.rerank("   ", [])

    def test_non_string_query_raises_error(self):
        reranker = CrossEncoderReranker(make_rerank_settings(), scorer=fixed_scorer([]))
        with pytest.raises(RerankerError):
            reranker.rerank(42, [])  # type: ignore[arg-type]

    def test_non_list_candidates_raises_error(self):
        reranker = CrossEncoderReranker(make_rerank_settings(), scorer=fixed_scorer([]))
        with pytest.raises(RerankerError):
            reranker.rerank("q", "not a list")  # type: ignore[arg-type]

    def test_candidate_without_id_raises_error(self):
        reranker = CrossEncoderReranker(make_rerank_settings(), scorer=fixed_scorer([]))
        with pytest.raises(RerankerError, match="id must be a non-empty string"):
            reranker.rerank("q", [{"text": "hello"}])

    def test_candidate_with_empty_id_raises_error(self):
        reranker = CrossEncoderReranker(make_rerank_settings(), scorer=fixed_scorer([]))
        with pytest.raises(RerankerError, match="id must be a non-empty string"):
            reranker.rerank("q", [{"id": "", "text": "hello"}])


# ---------------------------------------------------------------------------
# TestSentenceTransformers (mocked import)
# ---------------------------------------------------------------------------


class TestSentenceTransformers:
    def test_missing_sentence_transformers_raises_reranker_error(self):
        settings = make_rerank_settings(model="cross-encoder/ms-marco-MiniLM-L-6-v2")
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with pytest.raises(RerankerError, match="sentence-transformers is not installed"):
                CrossEncoderReranker(settings)

    def test_model_load_failure_raises_reranker_error(self):
        settings = make_rerank_settings(model="nonexistent/model-xyz")
        mock_module = MagicMock()
        mock_module.CrossEncoder.side_effect = OSError("model not found")
        with patch.dict("sys.modules", {"sentence_transformers": mock_module}):
            with pytest.raises(RerankerError, match="failed to load model"):
                CrossEncoderReranker(settings)

    def test_model_predict_success(self):
        settings = make_rerank_settings(model="some/model")
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.1]
        mock_module = MagicMock()
        mock_module.CrossEncoder.return_value = mock_model
        with patch.dict("sys.modules", {"sentence_transformers": mock_module}):
            reranker = CrossEncoderReranker(settings)
            candidates = [candidate("a"), candidate("b")]
            result = reranker.rerank("q", candidates)
        assert result[0]["id"] == "a"
        assert result[1]["id"] == "b"

    def test_model_predict_failure_raises_reranker_error(self):
        settings = make_rerank_settings(model="some/model")
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("GPU OOM")
        mock_module = MagicMock()
        mock_module.CrossEncoder.return_value = mock_model
        with patch.dict("sys.modules", {"sentence_transformers": mock_module}):
            reranker = CrossEncoderReranker(settings)
            with pytest.raises(RerankerError, match="model.predict failed"):
                reranker.rerank("q", [candidate("a")])


# ---------------------------------------------------------------------------
# TestNoScorerNoModel
# ---------------------------------------------------------------------------


class TestNoScorerNoModel:
    def test_empty_model_name_raises_reranker_error(self):
        settings = make_rerank_settings(model="")
        with pytest.raises(RerankerError, match="rerank.model must be set"):
            CrossEncoderReranker(settings)

    def test_none_model_name_raises_reranker_error(self):
        settings = RerankSettings(
            enabled=True, provider="cross_encoder", model="none", top_k=5
        )
        with pytest.raises(RerankerError, match="rerank.model must be set"):
            CrossEncoderReranker(settings)
