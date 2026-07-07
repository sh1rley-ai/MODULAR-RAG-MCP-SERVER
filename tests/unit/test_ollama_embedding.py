"""B7.4 — Tests for OllamaEmbedding (local backend, /api/embed protocol).

All tests are pure unit tests: HTTP is faked with `httpx.MockTransport`
(injected as a ready-made `httpx.Client`), so no network is touched.
"""

from __future__ import annotations

import json
from typing import Any, Callable

import httpx
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
from libs.embedding.base_embedding import EmbeddingError
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.embedding.ollama_embedding import DEFAULT_BASE_URL, OllamaEmbedding

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 4  # small vector dimension for tests


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)


def make_embedding_settings(**overrides: Any) -> EmbeddingSettings:
    defaults: dict[str, Any] = {
        "provider": "ollama",
        "model": "nomic-embed-text",
        "dimensions": DIM,
    }
    defaults.update(overrides)
    return EmbeddingSettings(**defaults)


def make_settings(**overrides: Any) -> Settings:
    return Settings(
        llm=LLMSettings(provider="openai", model="gpt-4o-mini", temperature=0.0, max_tokens=256),
        embedding=make_embedding_settings(**overrides),
        vector_store=VectorStoreSettings(provider="chroma", persist_directory="data/db/chroma", collection_name="default"),
        retrieval=RetrievalSettings(dense_top_k=20, sparse_top_k=20, fusion_top_k=10, rrf_k=60),
        rerank=RerankSettings(enabled=False, provider="none", model="none", top_k=5),
        evaluation=EvaluationSettings(enabled=False, provider="custom", metrics=["hit_rate"]),
        observability=ObservabilitySettings(
            log_level="INFO", trace_enabled=True, trace_file="logs/traces.jsonl", structured_logging=False
        ),
    )


def vector_for(seed: int, dim: int = DIM) -> list[float]:
    return [float(seed)] * dim


def ollama_embed_response(count: int, dim: int = DIM, model: str = "nomic-embed-text") -> dict:
    """Build a dict shaped like an Ollama /api/embed response."""

    return {
        "model": model,
        "embeddings": [vector_for(i, dim) for i in range(count)],
    }


def make_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://mock-ollama")


def make_embedding(handler: Callable[[httpx.Request], httpx.Response], **overrides: Any) -> OllamaEmbedding:
    return OllamaEmbedding(make_embedding_settings(**overrides), client=make_client(handler))


# ---------------------------------------------------------------------------
# TestFactoryRouting
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFactoryRouting:
    def test_ollama_registered_by_default(self) -> None:
        assert "ollama" in EmbeddingFactory.registered_providers()

    def test_factory_creates_ollama_without_api_key(self) -> None:
        embedding = EmbeddingFactory.create(make_settings())
        assert isinstance(embedding, OllamaEmbedding)


# ---------------------------------------------------------------------------
# TestOllamaEmbed
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOllamaEmbed:
    def test_embed_returns_one_vector_per_text(self) -> None:
        embedding = make_embedding(lambda r: httpx.Response(200, json=ollama_embed_response(3)))
        vectors = embedding.embed(["a", "b", "c"])
        assert vectors == [vector_for(0), vector_for(1), vector_for(2)]

    def test_embed_posts_to_api_embed_endpoint(self) -> None:
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, json=ollama_embed_response(2))

        texts = ["hello", "world"]
        make_embedding(handler, model="mxbai-embed-large").embed(texts)

        assert captured[0].url.path == "/api/embed"
        payload = json.loads(captured[0].content)
        assert payload["model"] == "mxbai-embed-large"
        assert payload["input"] == texts

    def test_embed_single_text(self) -> None:
        embedding = make_embedding(lambda r: httpx.Response(200, json=ollama_embed_response(1)))
        vectors = embedding.embed(["single text"])
        assert len(vectors) == 1
        assert vectors[0] == vector_for(0)

    def test_embed_validates_messages_before_http(self) -> None:
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(200, json=ollama_embed_response(1))

        embedding = make_embedding(handler)
        with pytest.raises(EmbeddingError, match="non-empty list"):
            embedding.embed([])
        assert calls == []

    def test_empty_string_raises_before_http(self) -> None:
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(200, json=ollama_embed_response(1))

        embedding = make_embedding(handler)
        with pytest.raises(EmbeddingError, match=r"texts\[0\] is empty"):
            embedding.embed(["   "])
        assert calls == []

    def test_non_string_item_raises_readable_error(self) -> None:
        embedding = make_embedding(lambda r: httpx.Response(200, json=ollama_embed_response(2)))
        with pytest.raises(EmbeddingError, match=r"texts\[1\]"):
            embedding.embed(["ok", 42])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# TestOversizePolicy
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOversizePolicy:
    def test_no_limit_passes_long_text_through(self) -> None:
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, json=ollama_embed_response(1))

        long_text = "x" * 100_000
        make_embedding(handler).embed([long_text])
        assert json.loads(captured[0].content)["input"] == [long_text]

    def test_oversize_raises_by_default(self) -> None:
        embedding = make_embedding(
            lambda r: httpx.Response(200, json=ollama_embed_response(1)),
            max_input_chars=10,
        )
        with pytest.raises(EmbeddingError, match=r"texts\[0\].*max_input_chars=10"):
            embedding.embed(["x" * 11])

    def test_oversize_truncated_when_configured(self) -> None:
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, json=ollama_embed_response(1))

        make_embedding(handler, max_input_chars=5, truncate_oversize=True).embed(["hello world"])
        assert json.loads(captured[0].content)["input"] == ["hello"]


# ---------------------------------------------------------------------------
# TestOllamaErrors
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOllamaErrors:
    def test_connect_error_raises_readable_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        with pytest.raises(EmbeddingError, match=r"\[ollama\] cannot connect.*ollama serve"):
            make_embedding(handler).embed(["hi"])

    def test_timeout_raises_readable_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out", request=request)

        with pytest.raises(EmbeddingError, match=r"\[ollama\] request to .* timed out"):
            make_embedding(handler).embed(["hi"])

    def test_http_404_raises_readable_error(self) -> None:
        with pytest.raises(EmbeddingError, match=r"\[ollama\] HTTP 404"):
            make_embedding(lambda r: httpx.Response(404, text="model not found")).embed(["hi"])

    def test_http_500_raises_readable_error(self) -> None:
        with pytest.raises(EmbeddingError, match=r"\[ollama\] HTTP 500"):
            make_embedding(lambda r: httpx.Response(500, text="internal error")).embed(["hi"])

    def test_non_dict_response_raises_readable_error(self) -> None:
        with pytest.raises(EmbeddingError, match=r"\[ollama\] unexpected response shape"):
            make_embedding(lambda r: httpx.Response(200, json=["unexpected"])).embed(["hi"])

    def test_missing_embeddings_field_raises_readable_error(self) -> None:
        with pytest.raises(EmbeddingError, match=r"\[ollama\] response missing 'embeddings' field"):
            make_embedding(lambda r: httpx.Response(200, json={"model": "x"})).embed(["hi"])

    def test_count_mismatch_raises_readable_error(self) -> None:
        with pytest.raises(EmbeddingError, match="expected 2 embeddings, got 1"):
            make_embedding(lambda r: httpx.Response(200, json=ollama_embed_response(1))).embed(["a", "b"])

    def test_invalid_json_wrapped_as_embedding_error(self) -> None:
        with pytest.raises(EmbeddingError, match=r"\[ollama\] embed call failed"):
            make_embedding(lambda r: httpx.Response(200, content=b"not json")).embed(["hi"])


# ---------------------------------------------------------------------------
# TestBaseUrlResolution
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBaseUrlResolution:
    def test_default_base_url_when_nothing_configured(self) -> None:
        embedding = OllamaEmbedding(make_embedding_settings())
        assert embedding.base_url == DEFAULT_BASE_URL

    def test_settings_base_url_takes_precedence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://from-env:11434")
        embedding = OllamaEmbedding(make_embedding_settings(base_url="http://from-settings:11434"))
        assert embedding.base_url == "http://from-settings:11434"

    def test_env_var_used_when_settings_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://gpu-box:11434")
        embedding = OllamaEmbedding(make_embedding_settings())
        assert embedding.base_url == "http://gpu-box:11434"

    def test_trailing_slash_stripped(self) -> None:
        embedding = OllamaEmbedding(make_embedding_settings(base_url="http://localhost:11434/"))
        assert embedding.base_url == "http://localhost:11434"
