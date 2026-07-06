"""B7.2 — Tests for OllamaLLM (local backend, native /api/chat protocol).

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
from libs.llm.base_llm import ChatResponse, LLMError
from libs.llm.llm_factory import LLMFactory
from libs.llm.ollama_llm import DEFAULT_BASE_URL, OllamaLLM

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch):
    """Remove real Ollama config so tests are deterministic."""

    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)


def make_llm_settings(**overrides: Any) -> LLMSettings:
    defaults: dict[str, Any] = {
        "provider": "ollama",
        "model": "llama3.1",
        "temperature": 0.0,
        "max_tokens": 256,
    }
    defaults.update(overrides)
    return LLMSettings(**defaults)


def make_settings(**overrides: Any) -> Settings:
    return Settings(
        llm=make_llm_settings(**overrides),
        embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small", dimensions=1536),
        vector_store=VectorStoreSettings(provider="chroma", persist_directory="data/db/chroma", collection_name="default"),
        retrieval=RetrievalSettings(dense_top_k=20, sparse_top_k=20, fusion_top_k=10, rrf_k=60),
        rerank=RerankSettings(enabled=False, provider="none", model="none", top_k=5),
        evaluation=EvaluationSettings(enabled=False, provider="custom", metrics=["hit_rate"]),
        observability=ObservabilitySettings(
            log_level="INFO", trace_enabled=True, trace_file="logs/traces.jsonl", structured_logging=False
        ),
    )


def ollama_response_body(content: str = "hello", model: str = "llama3.1", with_usage: bool = True) -> dict:
    """Build a dict shaped like an Ollama /api/chat non-streaming response."""

    body: dict[str, Any] = {
        "model": model,
        "message": {"role": "assistant", "content": content},
        "done": True,
    }
    if with_usage:
        body["prompt_eval_count"] = 5
        body["eval_count"] = 7
    return body


def make_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://mock-ollama")


def make_llm(handler: Callable[[httpx.Request], httpx.Response], **overrides: Any) -> OllamaLLM:
    return OllamaLLM(make_llm_settings(**overrides), client=make_client(handler))


# ---------------------------------------------------------------------------
# TestFactoryRouting
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFactoryRouting:
    def test_ollama_registered_by_default(self) -> None:
        assert "ollama" in LLMFactory.registered_providers()

    def test_factory_creates_ollama_without_api_key(self) -> None:
        llm = LLMFactory.create(make_settings())
        assert isinstance(llm, OllamaLLM)


# ---------------------------------------------------------------------------
# TestOllamaChat
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOllamaChat:
    def test_chat_returns_chat_response(self) -> None:
        llm = make_llm(lambda request: httpx.Response(200, json=ollama_response_body("hi there")))
        response = llm.chat([{"role": "user", "content": "hello"}])
        assert isinstance(response, ChatResponse)
        assert response.content == "hi there"
        assert response.model == "llama3.1"
        assert response.usage == {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}

    def test_chat_posts_expected_payload(self) -> None:
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, json=ollama_response_body())

        messages = [{"role": "user", "content": "hello"}]
        make_llm(handler, model="qwen2.5", temperature=0.3, max_tokens=64).chat(messages)

        assert captured[0].url.path == "/api/chat"
        payload = json.loads(captured[0].content)
        assert payload["model"] == "qwen2.5"
        assert payload["messages"] == messages
        assert payload["stream"] is False
        assert payload["options"] == {"temperature": 0.3, "num_predict": 64}

    def test_chat_validates_messages_before_http(self) -> None:
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(200, json=ollama_response_body())

        llm = make_llm(handler)
        with pytest.raises(LLMError, match="role"):
            llm.chat([{"role": "robot", "content": "hi"}])
        assert calls == []

    def test_response_without_usage_yields_empty_usage(self) -> None:
        llm = make_llm(lambda request: httpx.Response(200, json=ollama_response_body(with_usage=False)))
        assert llm.chat([{"role": "user", "content": "hi"}]).usage == {}

    def test_missing_content_raises_readable_error(self) -> None:
        llm = make_llm(lambda request: httpx.Response(200, json={"model": "llama3.1", "done": True}))
        with pytest.raises(LLMError, match=r"\[ollama\] response message content is missing"):
            llm.chat([{"role": "user", "content": "hi"}])


# ---------------------------------------------------------------------------
# TestOllamaErrors
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOllamaErrors:
    def test_connect_error_raises_readable_llm_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        llm = make_llm(handler)
        with pytest.raises(LLMError, match=r"\[ollama\] cannot connect .*ollama serve"):
            llm.chat([{"role": "user", "content": "hi"}])

    def test_timeout_raises_readable_llm_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out", request=request)

        llm = make_llm(handler)
        with pytest.raises(LLMError, match=r"\[ollama\] request to .* timed out"):
            llm.chat([{"role": "user", "content": "hi"}])

    def test_http_error_includes_status_code(self) -> None:
        llm = make_llm(lambda request: httpx.Response(404, text='{"error": "model not found"}'))
        with pytest.raises(LLMError, match=r"\[ollama\] HTTP 404 .*model not found"):
            llm.chat([{"role": "user", "content": "hi"}])

    def test_server_error_includes_status_code(self) -> None:
        llm = make_llm(lambda request: httpx.Response(500, text="internal error"))
        with pytest.raises(LLMError, match=r"\[ollama\] HTTP 500"):
            llm.chat([{"role": "user", "content": "hi"}])

    def test_invalid_json_wrapped_as_llm_error(self) -> None:
        llm = make_llm(lambda request: httpx.Response(200, content=b"not json"))
        with pytest.raises(LLMError, match=r"\[ollama\] chat call failed"):
            llm.chat([{"role": "user", "content": "hi"}])

    def test_non_dict_json_raises_readable_error(self) -> None:
        llm = make_llm(lambda request: httpx.Response(200, json=["unexpected"]))
        with pytest.raises(LLMError, match=r"\[ollama\] unexpected response shape"):
            llm.chat([{"role": "user", "content": "hi"}])


# ---------------------------------------------------------------------------
# TestBaseUrlResolution
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBaseUrlResolution:
    def test_default_base_url_when_nothing_configured(self) -> None:
        llm = OllamaLLM(make_llm_settings())
        assert llm.base_url == DEFAULT_BASE_URL

    def test_settings_base_url_takes_precedence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://from-env:11434")
        llm = OllamaLLM(make_llm_settings(base_url="http://from-settings:11434"))
        assert llm.base_url == "http://from-settings:11434"

    def test_env_var_used_when_settings_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://gpu-box:11434")
        llm = OllamaLLM(make_llm_settings())
        assert llm.base_url == "http://gpu-box:11434"

    def test_trailing_slash_stripped(self) -> None:
        llm = OllamaLLM(make_llm_settings(base_url="http://localhost:11434/"))
        assert llm.base_url == "http://localhost:11434"
