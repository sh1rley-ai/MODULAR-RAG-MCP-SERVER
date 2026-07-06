"""B7.1 — Smoke tests for OpenAI-compatible LLM providers (OpenAI/Azure/DeepSeek).

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
from libs.llm.azure_llm import DEFAULT_API_VERSION, AzureLLM
from libs.llm.base_llm import ChatResponse, LLMError
from libs.llm.deepseek_llm import DeepSeekLLM
from libs.llm.llm_factory import LLMFactory
from libs.llm.openai_llm import OpenAICompatibleLLM, OpenAILLM, resolve_secret

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROVIDER_ENV_VARS = (
    "OPENAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch):
    """Remove real provider credentials so tests are deterministic."""

    for name in PROVIDER_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def make_llm_settings(provider: str = "openai", **overrides: Any) -> LLMSettings:
    defaults: dict[str, Any] = {
        "provider": provider,
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "max_tokens": 256,
    }
    defaults.update(overrides)
    return LLMSettings(**defaults)


def make_settings(provider: str, **overrides: Any) -> Settings:
    """Build a full Settings object with the given llm provider."""

    return Settings(
        llm=make_llm_settings(provider=provider, **overrides),
        embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small", dimensions=1536),
        vector_store=VectorStoreSettings(provider="chroma", persist_directory="data/db/chroma", collection_name="default"),
        retrieval=RetrievalSettings(dense_top_k=20, sparse_top_k=20, fusion_top_k=10, rrf_k=60),
        rerank=RerankSettings(enabled=False, provider="none", model="none", top_k=5),
        evaluation=EvaluationSettings(enabled=False, provider="custom", metrics=["hit_rate"]),
        observability=ObservabilitySettings(
            log_level="INFO", trace_enabled=True, trace_file="logs/traces.jsonl", structured_logging=False
        ),
    )


def make_sdk_response(content: str = "hello", model: str = "gpt-4o-mini", with_usage: bool = True) -> SimpleNamespace:
    """Build an object shaped like an openai SDK ChatCompletion response."""

    usage = SimpleNamespace(prompt_tokens=5, completion_tokens=7, total_tokens=12) if with_usage else None
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        model=model,
        usage=usage,
    )


class FakeCompletions:
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
    """Minimal stand-in for the openai SDK client (chat.completions.create)."""

    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.completions = FakeCompletions(response, error)
        self.chat = SimpleNamespace(completions=self.completions)


class RecordingConstructor:
    """Monkeypatch target for OpenAI/AzureOpenAI: records constructor kwargs."""

    last_kwargs: dict = {}

    def __init__(self, **kwargs: Any) -> None:
        type(self).last_kwargs = kwargs
        self.chat = SimpleNamespace(completions=FakeCompletions(make_sdk_response()))


# ---------------------------------------------------------------------------
# TestFactoryRegistration
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFactoryRegistration:
    """Built-in providers register with LLMFactory at import time."""

    def test_default_providers_registered(self) -> None:
        registered = LLMFactory.registered_providers()
        for provider in ("openai", "azure", "deepseek"):
            assert provider in registered

    def test_factory_routes_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        llm = LLMFactory.create(make_settings("openai"))
        assert isinstance(llm, OpenAILLM)

    def test_factory_routes_deepseek(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        llm = LLMFactory.create(make_settings("deepseek"))
        assert isinstance(llm, DeepSeekLLM)

    def test_factory_routes_azure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
        llm = LLMFactory.create(make_settings("azure"))
        assert isinstance(llm, AzureLLM)


# ---------------------------------------------------------------------------
# TestOpenAILLM
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOpenAILLM:
    def test_chat_returns_chat_response(self) -> None:
        client = FakeClient(make_sdk_response(content="hi there", model="gpt-4o-mini"))
        llm = OpenAILLM(make_llm_settings(), client=client)
        response = llm.chat([{"role": "user", "content": "hello"}])
        assert isinstance(response, ChatResponse)
        assert response.content == "hi there"
        assert response.model == "gpt-4o-mini"
        assert response.usage == {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}

    def test_chat_passes_settings_to_api(self) -> None:
        client = FakeClient(make_sdk_response())
        llm = OpenAILLM(make_llm_settings(model="my-model", temperature=0.7, max_tokens=99), client=client)
        messages = [{"role": "user", "content": "hello"}]
        llm.chat(messages)
        call = client.completions.calls[0]
        assert call["model"] == "my-model"
        assert call["messages"] == messages
        assert call["temperature"] == 0.7
        assert call["max_tokens"] == 99

    def test_chat_validates_messages_before_api_call(self) -> None:
        client = FakeClient(make_sdk_response())
        llm = OpenAILLM(make_llm_settings(), client=client)
        with pytest.raises(LLMError, match="role"):
            llm.chat([{"role": "robot", "content": "hi"}])
        assert client.completions.calls == []

    def test_api_error_wrapped_with_provider_and_type(self) -> None:
        client = FakeClient(error=RuntimeError("connection refused"))
        llm = OpenAILLM(make_llm_settings(api_key="sk-secret-value"), client=client)
        with pytest.raises(LLMError, match=r"\[openai\] chat call failed \(RuntimeError\)") as exc_info:
            llm.chat([{"role": "user", "content": "hi"}])
        assert "sk-secret-value" not in str(exc_info.value)

    def test_empty_choices_raises_readable_error(self) -> None:
        client = FakeClient(SimpleNamespace(choices=[], model="m", usage=None))
        llm = OpenAILLM(make_llm_settings(), client=client)
        with pytest.raises(LLMError, match=r"\[openai\] response contained no choices"):
            llm.chat([{"role": "user", "content": "hi"}])

    def test_missing_content_raises_readable_error(self) -> None:
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None))], model="m", usage=None
        )
        llm = OpenAILLM(make_llm_settings(), client=FakeClient(response))
        with pytest.raises(LLMError, match="content is missing"):
            llm.chat([{"role": "user", "content": "hi"}])

    def test_response_without_usage_yields_empty_usage(self) -> None:
        client = FakeClient(make_sdk_response(with_usage=False))
        llm = OpenAILLM(make_llm_settings(), client=client)
        assert llm.chat([{"role": "user", "content": "hi"}]).usage == {}

    def test_missing_api_key_raises_readable_error(self) -> None:
        with pytest.raises(LLMError, match=r"\[openai\].*OPENAI_API_KEY"):
            OpenAILLM(make_llm_settings())

    def test_api_key_and_base_url_from_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("libs.llm.openai_llm.OpenAI", RecordingConstructor)
        OpenAILLM(make_llm_settings(api_key="sk-from-settings", base_url="https://gateway.example/v1"))
        assert RecordingConstructor.last_kwargs["api_key"] == "sk-from-settings"
        assert RecordingConstructor.last_kwargs["base_url"] == "https://gateway.example/v1"

    def test_api_key_env_reference_expanded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_CUSTOM_KEY", "sk-from-env-ref")
        monkeypatch.setattr("libs.llm.openai_llm.OpenAI", RecordingConstructor)
        OpenAILLM(make_llm_settings(api_key="${MY_CUSTOM_KEY}"))
        assert RecordingConstructor.last_kwargs["api_key"] == "sk-from-env-ref"

    def test_api_key_falls_back_to_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        monkeypatch.setattr("libs.llm.openai_llm.OpenAI", RecordingConstructor)
        OpenAILLM(make_llm_settings())
        assert RecordingConstructor.last_kwargs["api_key"] == "sk-from-env"


# ---------------------------------------------------------------------------
# TestDeepSeekLLM
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDeepSeekLLM:
    def test_uses_deepseek_default_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ds")
        monkeypatch.setattr("libs.llm.openai_llm.OpenAI", RecordingConstructor)
        DeepSeekLLM(make_llm_settings(provider="deepseek", model="deepseek-chat"))
        assert RecordingConstructor.last_kwargs["base_url"] == "https://api.deepseek.com/v1"

    def test_settings_base_url_overrides_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ds")
        monkeypatch.setattr("libs.llm.openai_llm.OpenAI", RecordingConstructor)
        DeepSeekLLM(make_llm_settings(provider="deepseek", base_url="https://proxy.example/v1"))
        assert RecordingConstructor.last_kwargs["base_url"] == "https://proxy.example/v1"

    def test_missing_api_key_names_deepseek_env_var(self) -> None:
        with pytest.raises(LLMError, match=r"\[deepseek\].*DEEPSEEK_API_KEY"):
            DeepSeekLLM(make_llm_settings(provider="deepseek"))

    def test_error_wrapped_with_deepseek_provider_name(self) -> None:
        client = FakeClient(error=TimeoutError("timed out"))
        llm = DeepSeekLLM(make_llm_settings(provider="deepseek"), client=client)
        with pytest.raises(LLMError, match=r"\[deepseek\] chat call failed \(TimeoutError\)"):
            llm.chat([{"role": "user", "content": "hi"}])


# ---------------------------------------------------------------------------
# TestAzureLLM
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAzureLLM:
    def test_missing_endpoint_raises_readable_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
        with pytest.raises(LLMError, match=r"\[azure\].*azure_endpoint.*AZURE_OPENAI_ENDPOINT"):
            AzureLLM(make_llm_settings(provider="azure"))

    def test_missing_api_key_names_azure_env_var(self) -> None:
        with pytest.raises(LLMError, match=r"\[azure\].*AZURE_OPENAI_API_KEY"):
            AzureLLM(make_llm_settings(provider="azure", azure_endpoint="https://x.openai.azure.com"))

    def test_client_built_from_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("libs.llm.azure_llm.AzureOpenAI", RecordingConstructor)
        AzureLLM(
            make_llm_settings(
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
        monkeypatch.setattr("libs.llm.azure_llm.AzureOpenAI", RecordingConstructor)
        AzureLLM(make_llm_settings(provider="azure", api_key="k", azure_endpoint="https://x.openai.azure.com"))
        assert RecordingConstructor.last_kwargs["api_version"] == DEFAULT_API_VERSION

    def test_deployment_name_used_as_model(self) -> None:
        client = FakeClient(make_sdk_response())
        llm = AzureLLM(
            make_llm_settings(provider="azure", deployment_name="my-gpt4o-deployment"), client=client
        )
        llm.chat([{"role": "user", "content": "hi"}])
        assert client.completions.calls[0]["model"] == "my-gpt4o-deployment"

    def test_model_used_when_no_deployment_name(self) -> None:
        client = FakeClient(make_sdk_response())
        llm = AzureLLM(make_llm_settings(provider="azure", model="gpt-4o"), client=client)
        llm.chat([{"role": "user", "content": "hi"}])
        assert client.completions.calls[0]["model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# TestResolveSecret
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestResolveSecret:
    def test_literal_value_returned_as_is(self) -> None:
        assert resolve_secret("sk-literal", "OPENAI_API_KEY") == "sk-literal"

    def test_env_reference_resolved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SOME_VAR", "resolved")
        assert resolve_secret("${SOME_VAR}", "OPENAI_API_KEY") == "resolved"

    def test_unset_value_falls_back_to_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        assert resolve_secret(None, "OPENAI_API_KEY") == "sk-env"

    def test_returns_none_when_nothing_set(self) -> None:
        assert resolve_secret(None, "OPENAI_API_KEY") is None

    def test_env_reference_to_unset_var_returns_none(self) -> None:
        assert resolve_secret("${UNSET_VAR_XYZ}", "OPENAI_API_KEY") is None


# ---------------------------------------------------------------------------
# TestLLMSettingsCompat
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestLLMSettingsCompat:
    def test_optional_connection_fields_default_to_none(self) -> None:
        settings = LLMSettings(provider="openai", model="m", temperature=0.0, max_tokens=1)
        assert settings.api_key is None
        assert settings.base_url is None
        assert settings.azure_endpoint is None
        assert settings.api_version is None
        assert settings.deployment_name is None

    def test_subclass_provider_metadata(self) -> None:
        assert OpenAICompatibleLLM.provider_name == "openai"
        assert DeepSeekLLM.provider_name == "deepseek"
        assert AzureLLM.provider_name == "azure"
