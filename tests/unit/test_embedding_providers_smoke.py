"""B7.3 — Smoke tests for OpenAI & Azure embedding providers.

All tests are pure unit tests: SDK clients are replaced by in-test fakes
(injected via the `client` constructor argument, or by monkeypatching the SDK
constructors). No network is touched.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

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
from libs.embedding.azure_embedding import DEFAULT_API_VERSION, AzureEmbedding
from libs.embedding.base_embedding import EmbeddingError
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.embedding.openai_embedding import OpenAICompatibleEmbedding, OpenAIEmbedding

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROVIDER_ENV_VARS = ("OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT")

DIM = 4  # small vector dimension for tests


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch):
    """Remove real provider credentials so tests are deterministic."""

    for name in PROVIDER_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def make_embedding_settings(provider: str = "openai", **overrides: Any) -> EmbeddingSettings:
    defaults: dict[str, Any] = {
        "provider": provider,
        "model": "text-embedding-3-small",
        "dimensions": DIM,
    }
    defaults.update(overrides)
    return EmbeddingSettings(**defaults)


def make_settings(provider: str, **overrides: Any) -> Settings:
    """Build a full Settings object with the given embedding provider."""

    return Settings(
        llm=LLMSettings(provider="openai", model="gpt-4o-mini", temperature=0.0, max_tokens=256),
        embedding=make_embedding_settings(provider=provider, **overrides),
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


def make_sdk_response(count: int, dim: int = DIM, shuffle: bool = False) -> SimpleNamespace:
    """Build an object shaped like an openai SDK embeddings response."""

    items = [SimpleNamespace(embedding=vector_for(i, dim), index=i) for i in range(count)]
    if shuffle:
        items = list(reversed(items))
    return SimpleNamespace(data=items, model="text-embedding-3-small")


class FakeEmbeddings:
    """Records create() calls and returns a canned response or raises."""

    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeClient:
    """Minimal stand-in for the openai SDK client (embeddings.create)."""

    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.embeddings = FakeEmbeddings(response, error)


class RecordingConstructor:
    """Monkeypatch target for OpenAI/AzureOpenAI: records constructor kwargs."""

    last_kwargs: dict = {}

    def __init__(self, **kwargs: Any) -> None:
        type(self).last_kwargs = kwargs
        self.embeddings = FakeEmbeddings(make_sdk_response(1))


# ---------------------------------------------------------------------------
# TestFactoryRegistration
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFactoryRegistration:
    """Built-in providers register with EmbeddingFactory at import time."""

    def test_default_providers_registered(self) -> None:
        registered = EmbeddingFactory.registered_providers()
        assert "openai" in registered
        assert "azure" in registered

    def test_factory_routes_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        embedding = EmbeddingFactory.create(make_settings("openai"))
        assert isinstance(embedding, OpenAIEmbedding)

    def test_factory_routes_azure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
        embedding = EmbeddingFactory.create(make_settings("azure"))
        assert isinstance(embedding, AzureEmbedding)


# ---------------------------------------------------------------------------
# TestOpenAIEmbedding
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOpenAIEmbedding:
    def test_embed_returns_one_vector_per_text(self) -> None:
        client = FakeClient(make_sdk_response(3))
        embedding = OpenAIEmbedding(make_embedding_settings(), client=client)
        vectors = embedding.embed(["a", "b", "c"])
        assert vectors == [vector_for(0), vector_for(1), vector_for(2)]

    def test_embed_passes_model_and_input(self) -> None:
        client = FakeClient(make_sdk_response(2))
        embedding = OpenAIEmbedding(make_embedding_settings(model="text-embedding-3-large"), client=client)
        embedding.embed(["hello", "world"])
        call = client.embeddings.calls[0]
        assert call["model"] == "text-embedding-3-large"
        assert call["input"] == ["hello", "world"]

    def test_embed_restores_input_order_from_index(self) -> None:
        client = FakeClient(make_sdk_response(3, shuffle=True))
        embedding = OpenAIEmbedding(make_embedding_settings(), client=client)
        vectors = embedding.embed(["a", "b", "c"])
        assert vectors == [vector_for(0), vector_for(1), vector_for(2)]

    def test_empty_list_raises_readable_error(self) -> None:
        embedding = OpenAIEmbedding(make_embedding_settings(), client=FakeClient())
        with pytest.raises(EmbeddingError, match="non-empty list"):
            embedding.embed([])

    def test_non_string_item_raises_readable_error(self) -> None:
        embedding = OpenAIEmbedding(make_embedding_settings(), client=FakeClient())
        with pytest.raises(EmbeddingError, match=r"texts\[1\]"):
            embedding.embed(["ok", 42])  # type: ignore[list-item]

    def test_empty_string_item_raises_readable_error(self) -> None:
        client = FakeClient(make_sdk_response(2))
        embedding = OpenAIEmbedding(make_embedding_settings(), client=client)
        with pytest.raises(EmbeddingError, match=r"texts\[1\] is empty"):
            embedding.embed(["ok", "   "])
        assert client.embeddings.calls == []

    def test_api_error_wrapped_with_provider_and_type(self) -> None:
        client = FakeClient(error=RuntimeError("quota exceeded"))
        embedding = OpenAIEmbedding(make_embedding_settings(api_key="sk-secret-value"), client=client)
        with pytest.raises(EmbeddingError, match=r"\[openai\] embedding call failed \(RuntimeError\)") as exc_info:
            embedding.embed(["hi"])
        assert "sk-secret-value" not in str(exc_info.value)

    def test_count_mismatch_raises_readable_error(self) -> None:
        client = FakeClient(make_sdk_response(1))
        embedding = OpenAIEmbedding(make_embedding_settings(), client=client)
        with pytest.raises(EmbeddingError, match="expected 2 embeddings, got 1"):
            embedding.embed(["a", "b"])

    def test_dimension_mismatch_raises_readable_error(self) -> None:
        client = FakeClient(make_sdk_response(1, dim=8))
        embedding = OpenAIEmbedding(make_embedding_settings(dimensions=4), client=client)
        with pytest.raises(EmbeddingError, match=r"8-dim vectors but embedding\.dimensions=4"):
            embedding.embed(["hi"])

    def test_missing_api_key_raises_readable_error(self) -> None:
        with pytest.raises(EmbeddingError, match=r"\[openai\].*OPENAI_API_KEY"):
            OpenAIEmbedding(make_embedding_settings())

    def test_api_key_and_base_url_from_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("libs.embedding.openai_embedding.OpenAI", RecordingConstructor)
        OpenAIEmbedding(make_embedding_settings(api_key="sk-from-settings", base_url="https://gateway.example/v1"))
        assert RecordingConstructor.last_kwargs["api_key"] == "sk-from-settings"
        assert RecordingConstructor.last_kwargs["base_url"] == "https://gateway.example/v1"

    def test_api_key_env_reference_expanded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_EMBED_KEY", "sk-from-env-ref")
        monkeypatch.setattr("libs.embedding.openai_embedding.OpenAI", RecordingConstructor)
        OpenAIEmbedding(make_embedding_settings(api_key="${MY_EMBED_KEY}"))
        assert RecordingConstructor.last_kwargs["api_key"] == "sk-from-env-ref"


# ---------------------------------------------------------------------------
# TestOversizePolicy
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOversizePolicy:
    def test_no_limit_passes_long_text_through(self) -> None:
        client = FakeClient(make_sdk_response(1))
        embedding = OpenAIEmbedding(make_embedding_settings(), client=client)
        long_text = "x" * 100_000
        embedding.embed([long_text])
        assert client.embeddings.calls[0]["input"] == [long_text]

    def test_oversize_raises_by_default(self) -> None:
        client = FakeClient(make_sdk_response(1))
        embedding = OpenAIEmbedding(make_embedding_settings(max_input_chars=10), client=client)
        with pytest.raises(EmbeddingError, match=r"texts\[0\].*max_input_chars=10.*truncate_oversize"):
            embedding.embed(["x" * 11])
        assert client.embeddings.calls == []

    def test_oversize_truncated_when_configured(self) -> None:
        client = FakeClient(make_sdk_response(1))
        embedding = OpenAIEmbedding(
            make_embedding_settings(max_input_chars=10, truncate_oversize=True), client=client
        )
        embedding.embed(["x" * 11])
        assert client.embeddings.calls[0]["input"] == ["x" * 10]

    def test_text_within_limit_untouched(self) -> None:
        client = FakeClient(make_sdk_response(1))
        embedding = OpenAIEmbedding(
            make_embedding_settings(max_input_chars=10, truncate_oversize=True), client=client
        )
        embedding.embed(["short"])
        assert client.embeddings.calls[0]["input"] == ["short"]


# ---------------------------------------------------------------------------
# TestAzureEmbedding
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAzureEmbedding:
    def test_missing_endpoint_raises_readable_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
        with pytest.raises(EmbeddingError, match=r"\[azure\].*azure_endpoint.*AZURE_OPENAI_ENDPOINT"):
            AzureEmbedding(make_embedding_settings(provider="azure"))

    def test_missing_api_key_names_azure_env_var(self) -> None:
        with pytest.raises(EmbeddingError, match=r"\[azure\].*AZURE_OPENAI_API_KEY"):
            AzureEmbedding(make_embedding_settings(provider="azure", azure_endpoint="https://x.openai.azure.com"))

    def test_client_built_from_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("libs.embedding.azure_embedding.AzureOpenAI", RecordingConstructor)
        AzureEmbedding(
            make_embedding_settings(
                provider="azure",
                api_key="azure-key",
                azure_endpoint="https://x.openai.azure.com",
                api_version="2024-10-21",
            )
        )
        assert RecordingConstructor.last_kwargs == {
            "api_key": "azure-key",
            "azure_endpoint": "https://x.openai.azure.com",
            "api_version": "2024-10-21",
        }

    def test_api_version_defaults_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("libs.embedding.azure_embedding.AzureOpenAI", RecordingConstructor)
        AzureEmbedding(
            make_embedding_settings(provider="azure", api_key="k", azure_endpoint="https://x.openai.azure.com")
        )
        assert RecordingConstructor.last_kwargs["api_version"] == DEFAULT_API_VERSION

    def test_deployment_name_used_as_model(self) -> None:
        client = FakeClient(make_sdk_response(1))
        embedding = AzureEmbedding(
            make_embedding_settings(provider="azure", deployment_name="my-ada-deployment"), client=client
        )
        embedding.embed(["hi"])
        assert client.embeddings.calls[0]["model"] == "my-ada-deployment"

    def test_model_used_when_no_deployment_name(self) -> None:
        client = FakeClient(make_sdk_response(1))
        embedding = AzureEmbedding(
            make_embedding_settings(provider="azure", model="text-embedding-ada-002"), client=client
        )
        embedding.embed(["hi"])
        assert client.embeddings.calls[0]["model"] == "text-embedding-ada-002"

    def test_azure_reuses_openai_core_logic(self) -> None:
        """Azure shares validation/oversize/order logic via the shared base class."""

        assert issubclass(AzureEmbedding, OpenAICompatibleEmbedding)
        client = FakeClient(make_sdk_response(3, shuffle=True))
        embedding = AzureEmbedding(make_embedding_settings(provider="azure"), client=client)
        assert embedding.embed(["a", "b", "c"]) == [vector_for(0), vector_for(1), vector_for(2)]

    def test_error_wrapped_with_azure_provider_name(self) -> None:
        client = FakeClient(error=TimeoutError("timed out"))
        embedding = AzureEmbedding(make_embedding_settings(provider="azure"), client=client)
        with pytest.raises(EmbeddingError, match=r"\[azure\] embedding call failed \(TimeoutError\)"):
            embedding.embed(["hi"])


# ---------------------------------------------------------------------------
# TestEmbeddingSettingsCompat
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEmbeddingSettingsCompat:
    def test_optional_fields_default_sensibly(self) -> None:
        settings = EmbeddingSettings(provider="openai", model="m", dimensions=4)
        assert settings.api_key is None
        assert settings.base_url is None
        assert settings.azure_endpoint is None
        assert settings.api_version is None
        assert settings.deployment_name is None
        assert settings.max_input_chars is None
        assert settings.truncate_oversize is False
