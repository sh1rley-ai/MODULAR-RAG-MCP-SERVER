"""B1 — Tests for BaseLLM abstraction and LLMFactory routing.

All tests are pure unit tests: fake providers are registered in-test to
verify factory routing. No network, no external services.
"""

from __future__ import annotations

from typing import List

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
from libs.llm.base_llm import BaseLLM, ChatResponse, LLMError, Message
from libs.llm.llm_factory import LLMFactory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_settings(provider: str = "fake", model: str = "fake-model") -> Settings:
    """Build a full Settings object with the given llm provider."""
    return Settings(
        llm=LLMSettings(provider=provider, model=model, temperature=0.0, max_tokens=256),
        embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small", dimensions=1536),
        vector_store=VectorStoreSettings(provider="chroma", persist_directory="data/db/chroma", collection_name="default"),
        retrieval=RetrievalSettings(dense_top_k=20, sparse_top_k=20, fusion_top_k=10, rrf_k=60),
        rerank=RerankSettings(enabled=False, provider="none", model="none", top_k=5),
        evaluation=EvaluationSettings(enabled=False, provider="custom", metrics=["hit_rate"]),
        observability=ObservabilitySettings(
            log_level="INFO", trace_enabled=True, trace_file="logs/traces.jsonl", structured_logging=False
        ),
    )


class FakeLLM(BaseLLM):
    """In-test stub provider: echoes the last user message."""

    def __init__(self, llm_settings: LLMSettings) -> None:
        self.settings = llm_settings

    def chat(self, messages: List[Message]) -> ChatResponse:
        self.validate_messages(messages)
        return ChatResponse(content=f"echo: {messages[-1]['content']}", model=self.settings.model)


class AnotherFakeLLM(FakeLLM):
    """Second stub provider to verify routing between multiple providers."""


@pytest.fixture(autouse=True)
def clean_registry():
    """Snapshot and restore the factory registry around each test."""
    snapshot = dict(LLMFactory._registry)
    LLMFactory._registry.clear()
    yield
    LLMFactory._registry.clear()
    LLMFactory._registry.update(snapshot)


# ---------------------------------------------------------------------------
# TestLLMFactoryRouting
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestLLMFactoryRouting:
    """Factory routes settings.llm.provider to the registered implementation."""

    def test_create_returns_registered_provider(self) -> None:
        LLMFactory.register("fake", FakeLLM)
        llm = LLMFactory.create(make_settings(provider="fake"))
        assert isinstance(llm, FakeLLM)

    def test_created_instance_receives_llm_settings(self) -> None:
        LLMFactory.register("fake", FakeLLM)
        llm = LLMFactory.create(make_settings(provider="fake", model="my-model"))
        assert llm.settings.model == "my-model"
        assert llm.settings.max_tokens == 256

    def test_routes_between_multiple_providers(self) -> None:
        LLMFactory.register("fake", FakeLLM)
        LLMFactory.register("another", AnotherFakeLLM)
        assert isinstance(LLMFactory.create(make_settings(provider="another")), AnotherFakeLLM)
        llm = LLMFactory.create(make_settings(provider="fake"))
        assert isinstance(llm, FakeLLM)
        assert not isinstance(llm, AnotherFakeLLM)

    def test_provider_name_is_case_insensitive(self) -> None:
        LLMFactory.register("Fake", FakeLLM)
        llm = LLMFactory.create(make_settings(provider="FAKE"))
        assert isinstance(llm, FakeLLM)

    def test_unknown_provider_raises_readable_error(self) -> None:
        LLMFactory.register("fake", FakeLLM)
        with pytest.raises(ValueError, match="Unknown LLM provider: 'nonexistent'"):
            LLMFactory.create(make_settings(provider="nonexistent"))

    def test_unknown_provider_error_lists_registered(self) -> None:
        LLMFactory.register("fake", FakeLLM)
        with pytest.raises(ValueError, match="fake"):
            LLMFactory.create(make_settings(provider="nonexistent"))

    def test_unregister_removes_provider(self) -> None:
        LLMFactory.register("fake", FakeLLM)
        LLMFactory.unregister("fake")
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            LLMFactory.create(make_settings(provider="fake"))


# ---------------------------------------------------------------------------
# TestLLMFactoryRegistration
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestLLMFactoryRegistration:
    """register() validates its inputs."""

    def test_register_rejects_non_basellm_class(self) -> None:
        class NotAnLLM:
            pass

        with pytest.raises(TypeError, match="BaseLLM"):
            LLMFactory.register("bad", NotAnLLM)  # type: ignore[arg-type]

    def test_register_rejects_empty_provider_name(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            LLMFactory.register("  ", FakeLLM)

    def test_registered_providers_lists_names(self) -> None:
        LLMFactory.register("fake", FakeLLM)
        LLMFactory.register("another", AnotherFakeLLM)
        assert LLMFactory.registered_providers() == ["another", "fake"]


# ---------------------------------------------------------------------------
# TestBaseLLMContract
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBaseLLMContract:
    """BaseLLM abstract contract and message validation."""

    def test_base_llm_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError):
            BaseLLM()  # type: ignore[abstract]

    def test_chat_returns_chat_response(self) -> None:
        llm = FakeLLM(make_settings().llm)
        response = llm.chat([{"role": "user", "content": "hello"}])
        assert isinstance(response, ChatResponse)
        assert response.content == "echo: hello"
        assert response.model == "fake-model"

    def test_chat_response_serializable(self) -> None:
        response = ChatResponse(content="hi", model="m", usage={"total_tokens": 3})
        assert response.to_dict() == {"content": "hi", "model": "m", "usage": {"total_tokens": 3}}

    def test_validate_messages_rejects_empty_list(self) -> None:
        with pytest.raises(LLMError, match="non-empty list"):
            BaseLLM.validate_messages([])

    def test_validate_messages_rejects_non_list(self) -> None:
        with pytest.raises(LLMError, match="non-empty list"):
            BaseLLM.validate_messages("hello")

    def test_validate_messages_rejects_bad_role(self) -> None:
        with pytest.raises(LLMError, match="role"):
            BaseLLM.validate_messages([{"role": "robot", "content": "hi"}])

    def test_validate_messages_rejects_missing_content(self) -> None:
        with pytest.raises(LLMError, match="content"):
            BaseLLM.validate_messages([{"role": "user"}])

    def test_validate_messages_accepts_valid_conversation(self) -> None:
        BaseLLM.validate_messages(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        )  # no exception
