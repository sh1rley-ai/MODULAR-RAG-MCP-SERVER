"""B7.7 — Unit tests for LLMReranker.

All tests are pure unit tests: the LLM is always mocked (no network, no real
API calls). The mock is a simple callable that returns a ChatResponse with a
pre-configured JSON body.

Test structure:
    TestFactoryRouting    — factory registration and creation
    TestRerankOrdering    — candidates are reordered by descending LLM score
    TestPromptLoading     — prompt path / text injection and missing-file error
    TestLLMInteraction    — message shape sent to the LLM, LLM error handling
    TestOutputParsing     — JSON parsing and schema validation
    TestValidation        — query / candidates input validation
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

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
from libs.llm.base_llm import BaseLLM, ChatResponse
from libs.reranker.base_reranker import RerankerError
from libs.reranker.llm_reranker import LLMReranker
from libs.reranker.reranker_factory import RerankerFactory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SIMPLE_PROMPT = "Score passages for relevance. Return JSON array."


def make_rerank_settings(enabled: bool = True) -> RerankSettings:
    return RerankSettings(enabled=enabled, provider="llm", model="gpt-4o-mini", top_k=5)


def make_full_settings(provider: str = "llm") -> Settings:
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
        rerank=RerankSettings(enabled=True, provider=provider, model="gpt-4o-mini", top_k=5),
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


def candidate(cid: str, text: str = "", score: float = 0.5) -> Dict[str, Any]:
    return {"id": cid, "score": score, "text": text, "metadata": {}}


def mock_llm(json_response: Any) -> BaseLLM:
    """Return a mock BaseLLM whose .chat() yields the given JSON as response content."""
    llm = MagicMock(spec=BaseLLM)
    llm.chat.return_value = ChatResponse(
        content=json.dumps(json_response), model="mock"
    )
    return llm


def scored_response(
    candidates: List[Dict[str, Any]], scores: List[int]
) -> List[Dict[str, Any]]:
    """Build a valid LLM response list from candidates + scores."""
    return [
        {"passage_id": c["id"], "score": s, "reasoning": "mock"}
        for c, s in zip(candidates, scores)
    ]


# ---------------------------------------------------------------------------
# TestFactoryRouting
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFactoryRouting:
    def test_llm_backend_registered_by_default(self) -> None:
        assert "llm" in RerankerFactory.registered_backends()

    def test_factory_creates_llm_reranker(self) -> None:
        reranker = RerankerFactory.create(make_full_settings(provider="llm"))
        assert isinstance(reranker, LLMReranker)

    def test_factory_stores_settings(self) -> None:
        reranker = RerankerFactory.create(make_full_settings(provider="llm"))
        assert reranker.settings.provider == "llm"


# ---------------------------------------------------------------------------
# TestRerankOrdering
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRerankOrdering:
    def test_higher_score_candidate_ranked_first(self) -> None:
        candidates = [candidate("low", "low text"), candidate("high", "high text")]
        llm = mock_llm(
            [
                {"passage_id": "low", "score": 1, "reasoning": ""},
                {"passage_id": "high", "score": 3, "reasoning": ""},
            ]
        )
        reranker = LLMReranker(make_rerank_settings(), llm=llm, prompt_text=_SIMPLE_PROMPT)
        result = reranker.rerank("query", candidates)
        assert result[0]["id"] == "high"
        assert result[1]["id"] == "low"

    def test_all_candidates_preserved_in_output(self) -> None:
        candidates = [candidate(f"c{i}") for i in range(4)]
        llm = mock_llm(scored_response(candidates, [2, 0, 3, 1]))
        reranker = LLMReranker(make_rerank_settings(), llm=llm, prompt_text=_SIMPLE_PROMPT)
        result = reranker.rerank("q", candidates)
        assert len(result) == 4
        assert {r["id"] for r in result} == {"c0", "c1", "c2", "c3"}

    def test_output_order_matches_score_descending(self) -> None:
        candidates = [candidate("a"), candidate("b"), candidate("c")]
        llm = mock_llm(
            [
                {"passage_id": "a", "score": 1, "reasoning": ""},
                {"passage_id": "b", "score": 3, "reasoning": ""},
                {"passage_id": "c", "score": 2, "reasoning": ""},
            ]
        )
        reranker = LLMReranker(make_rerank_settings(), llm=llm, prompt_text=_SIMPLE_PROMPT)
        result = reranker.rerank("q", candidates)
        assert [r["id"] for r in result] == ["b", "c", "a"]

    def test_single_candidate_returned_unchanged(self) -> None:
        candidates = [candidate("only", "text")]
        llm = mock_llm([{"passage_id": "only", "score": 2, "reasoning": ""}])
        reranker = LLMReranker(make_rerank_settings(), llm=llm, prompt_text=_SIMPLE_PROMPT)
        result = reranker.rerank("q", candidates)
        assert len(result) == 1
        assert result[0]["id"] == "only"

    def test_empty_candidates_returns_empty_list(self) -> None:
        reranker = LLMReranker(make_rerank_settings(), llm=MagicMock(spec=BaseLLM), prompt_text=_SIMPLE_PROMPT)
        result = reranker.rerank("q", [])
        assert result == []

    def test_accepts_optional_trace_argument(self) -> None:
        candidates = [candidate("x")]
        llm = mock_llm([{"passage_id": "x", "score": 2, "reasoning": ""}])
        reranker = LLMReranker(make_rerank_settings(), llm=llm, prompt_text=_SIMPLE_PROMPT)
        result = reranker.rerank("q", candidates, trace=object())
        assert len(result) == 1

    def test_tie_score_preserves_all_candidates(self) -> None:
        candidates = [candidate("x"), candidate("y")]
        llm = mock_llm(
            [
                {"passage_id": "x", "score": 2, "reasoning": ""},
                {"passage_id": "y", "score": 2, "reasoning": ""},
            ]
        )
        reranker = LLMReranker(make_rerank_settings(), llm=llm, prompt_text=_SIMPLE_PROMPT)
        result = reranker.rerank("q", candidates)
        assert len(result) == 2
        assert {r["id"] for r in result} == {"x", "y"}


# ---------------------------------------------------------------------------
# TestPromptLoading
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPromptLoading:
    def test_prompt_text_injection_bypasses_file(self) -> None:
        injected = "Custom prompt text — no file needed."
        reranker = LLMReranker(make_rerank_settings(), prompt_text=injected)
        assert reranker._prompt_template == injected

    def test_prompt_loaded_from_custom_path(self, tmp_path: Path) -> None:
        prompt_file = tmp_path / "custom_rerank.txt"
        prompt_file.write_text("custom file content", encoding="utf-8")
        reranker = LLMReranker(make_rerank_settings(), prompt_path=str(prompt_file))
        assert reranker._prompt_template == "custom file content"

    def test_missing_prompt_path_raises_reranker_error(self, tmp_path: Path) -> None:
        missing = str(tmp_path / "nonexistent.txt")
        with pytest.raises(RerankerError, match="failed to load rerank prompt"):
            LLMReranker(make_rerank_settings(), prompt_path=missing)

    def test_default_prompt_path_loads_rerank_txt(self) -> None:
        reranker = LLMReranker(make_rerank_settings())
        assert len(reranker._prompt_template) > 20


# ---------------------------------------------------------------------------
# TestLLMInteraction
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLLMInteraction:
    def test_llm_called_with_system_and_user_messages(self) -> None:
        candidates = [candidate("doc1", "some text")]
        llm = mock_llm([{"passage_id": "doc1", "score": 2, "reasoning": ""}])
        reranker = LLMReranker(make_rerank_settings(), llm=llm, prompt_text=_SIMPLE_PROMPT)
        reranker.rerank("test query", candidates)
        messages = llm.chat.call_args[0][0]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == _SIMPLE_PROMPT
        assert messages[1]["role"] == "user"

    def test_user_message_contains_query_and_candidate_ids(self) -> None:
        candidates = [candidate("myid", "mytext")]
        llm = mock_llm([{"passage_id": "myid", "score": 1, "reasoning": ""}])
        reranker = LLMReranker(make_rerank_settings(), llm=llm, prompt_text=_SIMPLE_PROMPT)
        reranker.rerank("the_query", candidates)
        user_msg = llm.chat.call_args[0][0][1]["content"]
        assert "the_query" in user_msg
        assert "myid" in user_msg
        assert "mytext" in user_msg

    def test_no_llm_raises_reranker_error(self) -> None:
        reranker = LLMReranker(make_rerank_settings(), prompt_text=_SIMPLE_PROMPT)
        with pytest.raises(RerankerError, match="requires an injected BaseLLM"):
            reranker.rerank("q", [candidate("c1")])

    def test_llm_exception_wrapped_as_reranker_error(self) -> None:
        llm = MagicMock(spec=BaseLLM)
        llm.chat.side_effect = RuntimeError("network error")
        reranker = LLMReranker(make_rerank_settings(), llm=llm, prompt_text=_SIMPLE_PROMPT)
        with pytest.raises(RerankerError, match="LLM call failed"):
            reranker.rerank("q", [candidate("c1")])


# ---------------------------------------------------------------------------
# TestOutputParsing
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOutputParsing:
    def test_valid_json_response_parsed_correctly(self) -> None:
        candidates = [candidate("a"), candidate("b")]
        llm = mock_llm(
            [
                {"passage_id": "a", "score": 3, "reasoning": "good"},
                {"passage_id": "b", "score": 1, "reasoning": "weak"},
            ]
        )
        reranker = LLMReranker(make_rerank_settings(), llm=llm, prompt_text=_SIMPLE_PROMPT)
        result = reranker.rerank("q", candidates)
        assert result[0]["id"] == "a"

    def test_json_embedded_in_prose_is_extracted(self) -> None:
        candidates = [candidate("x")]
        prose_response = (
            'Here is the ranking:\n'
            '[{"passage_id": "x", "score": 2, "reasoning": "ok"}]\n'
            'End of response.'
        )
        llm = MagicMock(spec=BaseLLM)
        llm.chat.return_value = ChatResponse(content=prose_response, model="mock")
        reranker = LLMReranker(make_rerank_settings(), llm=llm, prompt_text=_SIMPLE_PROMPT)
        result = reranker.rerank("q", candidates)
        assert len(result) == 1

    def test_non_json_response_raises_reranker_error(self) -> None:
        llm = MagicMock(spec=BaseLLM)
        llm.chat.return_value = ChatResponse(content="I cannot rank these.", model="mock")
        reranker = LLMReranker(make_rerank_settings(), llm=llm, prompt_text=_SIMPLE_PROMPT)
        with pytest.raises(RerankerError, match="no JSON array"):
            reranker.rerank("q", [candidate("c1")])

    def test_unknown_passage_id_raises_reranker_error(self) -> None:
        candidates = [candidate("real_id")]
        llm = mock_llm([{"passage_id": "ghost_id", "score": 2, "reasoning": ""}])
        reranker = LLMReranker(make_rerank_settings(), llm=llm, prompt_text=_SIMPLE_PROMPT)
        with pytest.raises(RerankerError, match="not in candidates"):
            reranker.rerank("q", candidates)

    def test_missing_candidate_in_response_raises_reranker_error(self) -> None:
        candidates = [candidate("a"), candidate("b")]
        # Only returns "a", missing "b"
        llm = mock_llm([{"passage_id": "a", "score": 2, "reasoning": ""}])
        reranker = LLMReranker(make_rerank_settings(), llm=llm, prompt_text=_SIMPLE_PROMPT)
        with pytest.raises(RerankerError, match="missing passage_ids"):
            reranker.rerank("q", candidates)

    def test_non_numeric_score_raises_reranker_error(self) -> None:
        candidates = [candidate("x")]
        llm = mock_llm([{"passage_id": "x", "score": "high", "reasoning": ""}])
        reranker = LLMReranker(make_rerank_settings(), llm=llm, prompt_text=_SIMPLE_PROMPT)
        with pytest.raises(RerankerError, match="must be numeric"):
            reranker.rerank("q", candidates)

    def test_missing_passage_id_field_raises_reranker_error(self) -> None:
        candidates = [candidate("x")]
        llm = mock_llm([{"score": 2, "reasoning": "no id here"}])
        reranker = LLMReranker(make_rerank_settings(), llm=llm, prompt_text=_SIMPLE_PROMPT)
        with pytest.raises(RerankerError, match="non-empty string 'passage_id'"):
            reranker.rerank("q", candidates)


# ---------------------------------------------------------------------------
# TestValidation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidation:
    def test_empty_query_raises_reranker_error(self) -> None:
        reranker = LLMReranker(make_rerank_settings(), prompt_text=_SIMPLE_PROMPT)
        with pytest.raises(RerankerError, match="non-empty string"):
            reranker.rerank("", [candidate("c")])

    def test_whitespace_only_query_raises_reranker_error(self) -> None:
        reranker = LLMReranker(make_rerank_settings(), prompt_text=_SIMPLE_PROMPT)
        with pytest.raises(RerankerError, match="non-empty string"):
            reranker.rerank("   ", [candidate("c")])

    def test_non_list_candidates_raises_reranker_error(self) -> None:
        reranker = LLMReranker(make_rerank_settings(), prompt_text=_SIMPLE_PROMPT)
        with pytest.raises(RerankerError, match="list of dicts"):
            reranker.rerank("q", "not a list")  # type: ignore[arg-type]

    def test_candidate_missing_id_raises_reranker_error(self) -> None:
        reranker = LLMReranker(make_rerank_settings(), prompt_text=_SIMPLE_PROMPT)
        with pytest.raises(RerankerError, match="non-empty string"):
            reranker.rerank("q", [{"id": "", "text": "x"}])
