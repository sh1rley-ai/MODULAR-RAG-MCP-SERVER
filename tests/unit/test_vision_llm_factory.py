"""B8 — Unit tests for BaseVisionLLM and LLMFactory vision extension.

All tests are pure unit tests using a FakeVisionLLM stub — no real API calls,
no file I/O beyond small in-memory bytes.

Test structure:
    TestVisionRegistry        — register / unregister / list vision providers
    TestCreateVisionLLM       — factory routing to registered providers
    TestCreateVisionLLMErrors — missing settings, unknown provider errors
    TestBaseVisionLLMContract — abstract interface and validation helpers
    TestPreprocessImage       — default preprocess_image (bytes passthrough + file read)
    TestSettingsParsing       — VisionLLMSettings parsed from dict via Settings.from_dict
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Union
from unittest.mock import patch

import pytest

from core.settings import (
    EmbeddingSettings,
    EvaluationSettings,
    LLMSettings,
    ObservabilitySettings,
    RerankSettings,
    RetrievalSettings,
    Settings,
    VisionLLMSettings,
    VectorStoreSettings,
)
from libs.llm.base_llm import ChatResponse
from libs.llm.base_vision_llm import BaseVisionLLM, VisionLLMError
from libs.llm.llm_factory import LLMFactory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeVisionLLM(BaseVisionLLM):
    """Minimal concrete implementation for testing."""

    provider_name = "fake_vision"

    def __init__(self, vision_settings: Any) -> None:
        self.settings = vision_settings
        self.calls: list[dict] = []

    def chat_with_image(
        self,
        text: str,
        image_path: Union[str, bytes],
        trace: Any | None = None,
    ) -> ChatResponse:
        self.validate_text(text)
        self.validate_image_path(image_path)
        self.calls.append({"text": text, "image_path": image_path})
        return ChatResponse(content="fake caption", model="fake-vision-model")


def make_vision_settings(provider: str = "fake_vision", model: str = "gpt-4o") -> VisionLLMSettings:
    return VisionLLMSettings(
        provider=provider,
        model=model,
        max_image_size=2048,
    )


def make_base_settings(vision_provider: str = "fake_vision") -> Settings:
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
        rerank=RerankSettings(enabled=False, provider="none", model="none", top_k=5),
        evaluation=EvaluationSettings(
            enabled=False, provider="custom", metrics=["hit_rate"]
        ),
        observability=ObservabilitySettings(
            log_level="INFO",
            trace_enabled=False,
            trace_file="logs/traces.jsonl",
            structured_logging=False,
        ),
        vision_llm=make_vision_settings(provider=vision_provider),
    )


def make_settings_no_vision() -> Settings:
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
        rerank=RerankSettings(enabled=False, provider="none", model="none", top_k=5),
        evaluation=EvaluationSettings(
            enabled=False, provider="custom", metrics=["hit_rate"]
        ),
        observability=ObservabilitySettings(
            log_level="INFO",
            trace_enabled=False,
            trace_file="logs/traces.jsonl",
            structured_logging=False,
        ),
        vision_llm=None,
    )


# ---------------------------------------------------------------------------
# TestVisionRegistry
# ---------------------------------------------------------------------------


class TestVisionRegistry:
    def setup_method(self):
        LLMFactory.unregister_vision("fake_vision")
        LLMFactory.unregister_vision("another_vision")

    def teardown_method(self):
        LLMFactory.unregister_vision("fake_vision")
        LLMFactory.unregister_vision("another_vision")

    def test_register_vision_provider(self):
        LLMFactory.register_vision("fake_vision", FakeVisionLLM)
        assert "fake_vision" in LLMFactory.registered_vision_providers()

    def test_unregister_vision_provider(self):
        LLMFactory.register_vision("fake_vision", FakeVisionLLM)
        LLMFactory.unregister_vision("fake_vision")
        assert "fake_vision" not in LLMFactory.registered_vision_providers()

    def test_unregister_nonexistent_provider_is_noop(self):
        LLMFactory.unregister_vision("nonexistent")  # should not raise

    def test_registered_vision_providers_sorted(self):
        LLMFactory.register_vision("another_vision", FakeVisionLLM)
        LLMFactory.register_vision("fake_vision", FakeVisionLLM)
        providers = LLMFactory.registered_vision_providers()
        assert providers == sorted(providers)

    def test_register_requires_non_empty_provider_name(self):
        with pytest.raises(ValueError, match="non-empty string"):
            LLMFactory.register_vision("", FakeVisionLLM)

    def test_register_requires_base_vision_llm_subclass(self):
        class NotAVisionLLM:
            pass

        with pytest.raises(TypeError, match="subclass of BaseVisionLLM"):
            LLMFactory.register_vision("bad", NotAVisionLLM)  # type: ignore[arg-type]

    def test_register_is_case_insensitive(self):
        LLMFactory.register_vision("FAKE_VISION", FakeVisionLLM)
        assert "fake_vision" in LLMFactory.registered_vision_providers()

    def test_vision_registry_independent_from_text_registry(self):
        # Registering a vision provider should not affect the text registry
        before = set(LLMFactory.registered_providers())
        LLMFactory.register_vision("fake_vision", FakeVisionLLM)
        after = set(LLMFactory.registered_providers())
        assert before == after


# ---------------------------------------------------------------------------
# TestCreateVisionLLM
# ---------------------------------------------------------------------------


class TestCreateVisionLLM:
    def setup_method(self):
        LLMFactory.register_vision("fake_vision", FakeVisionLLM)

    def teardown_method(self):
        LLMFactory.unregister_vision("fake_vision")

    def test_create_vision_llm_returns_correct_type(self):
        settings = make_base_settings(vision_provider="fake_vision")
        llm = LLMFactory.create_vision_llm(settings)
        assert isinstance(llm, FakeVisionLLM)

    def test_created_instance_receives_vision_settings(self):
        settings = make_base_settings(vision_provider="fake_vision")
        llm = LLMFactory.create_vision_llm(settings)
        assert isinstance(llm, FakeVisionLLM)
        assert llm.settings is settings.vision_llm

    def test_create_vision_llm_provider_lookup_is_case_insensitive(self):
        vs = VisionLLMSettings(
            provider="FAKE_VISION", model="gpt-4o", max_image_size=1024
        )
        settings = make_base_settings()
        settings_upper = Settings(
            llm=settings.llm,
            embedding=settings.embedding,
            vector_store=settings.vector_store,
            retrieval=settings.retrieval,
            rerank=settings.rerank,
            evaluation=settings.evaluation,
            observability=settings.observability,
            vision_llm=vs,
        )
        llm = LLMFactory.create_vision_llm(settings_upper)
        assert isinstance(llm, FakeVisionLLM)

    def test_created_vision_llm_can_chat_with_image(self):
        settings = make_base_settings(vision_provider="fake_vision")
        llm = LLMFactory.create_vision_llm(settings)
        response = llm.chat_with_image("describe this", b"\xff\xd8\xff")
        assert response.content == "fake caption"


# ---------------------------------------------------------------------------
# TestCreateVisionLLMErrors
# ---------------------------------------------------------------------------


class TestCreateVisionLLMErrors:
    def teardown_method(self):
        LLMFactory.unregister_vision("fake_vision")

    def test_missing_vision_llm_settings_raises_value_error(self):
        settings = make_settings_no_vision()
        with pytest.raises(ValueError, match="vision_llm section is missing"):
            LLMFactory.create_vision_llm(settings)

    def test_unknown_provider_raises_value_error(self):
        LLMFactory.unregister_vision("fake_vision")
        settings = make_base_settings(vision_provider="fake_vision")
        with pytest.raises(ValueError, match="Unknown Vision LLM provider"):
            LLMFactory.create_vision_llm(settings)

    def test_error_message_lists_registered_providers(self):
        LLMFactory.register_vision("fake_vision", FakeVisionLLM)
        settings = make_base_settings(vision_provider="unknown_provider")
        with pytest.raises(ValueError, match="fake_vision"):
            LLMFactory.create_vision_llm(settings)


# ---------------------------------------------------------------------------
# TestBaseVisionLLMContract
# ---------------------------------------------------------------------------


class TestBaseVisionLLMContract:
    def setup_method(self):
        LLMFactory.register_vision("fake_vision", FakeVisionLLM)
        self.llm = FakeVisionLLM(make_vision_settings())

    def teardown_method(self):
        LLMFactory.unregister_vision("fake_vision")

    def test_chat_with_image_returns_chat_response(self):
        result = self.llm.chat_with_image("describe", b"\x00\x01")
        assert isinstance(result, ChatResponse)

    def test_validate_text_rejects_empty_string(self):
        with pytest.raises(VisionLLMError, match="non-empty string"):
            BaseVisionLLM.validate_text("")

    def test_validate_text_rejects_whitespace(self):
        with pytest.raises(VisionLLMError, match="non-empty string"):
            BaseVisionLLM.validate_text("   ")

    def test_validate_text_rejects_non_string(self):
        with pytest.raises(VisionLLMError):
            BaseVisionLLM.validate_text(123)  # type: ignore[arg-type]

    def test_validate_text_accepts_valid_string(self):
        BaseVisionLLM.validate_text("hello world")  # should not raise

    def test_validate_image_path_accepts_bytes(self):
        BaseVisionLLM.validate_image_path(b"\xff\xd8\xff")  # should not raise

    def test_validate_image_path_accepts_string_path(self):
        BaseVisionLLM.validate_image_path("/some/path/image.png")  # should not raise

    def test_validate_image_path_rejects_empty_string(self):
        with pytest.raises(VisionLLMError, match="empty"):
            BaseVisionLLM.validate_image_path("")

    def test_validate_image_path_rejects_empty_bytes(self):
        with pytest.raises(VisionLLMError, match="empty"):
            BaseVisionLLM.validate_image_path(b"")

    def test_validate_image_path_rejects_wrong_type(self):
        with pytest.raises(VisionLLMError, match="str.*bytes"):
            BaseVisionLLM.validate_image_path(42)  # type: ignore[arg-type]

    def test_fake_llm_records_calls(self):
        self.llm.chat_with_image("first call", b"\x01")
        self.llm.chat_with_image("second call", b"\x02")
        assert len(self.llm.calls) == 2
        assert self.llm.calls[0]["text"] == "first call"
        assert self.llm.calls[1]["text"] == "second call"


# ---------------------------------------------------------------------------
# TestPreprocessImage
# ---------------------------------------------------------------------------


class TestPreprocessImage:
    def setup_method(self):
        self.llm = FakeVisionLLM(make_vision_settings())

    def test_preprocess_bytes_returns_bytes_unchanged(self):
        raw = b"\xff\xd8\xff\xe0\x00\x10JFIF"
        result = self.llm.preprocess_image(raw)
        assert result == raw

    def test_preprocess_file_path_reads_file_content(self):
        content = b"\x89PNG\r\n\x1a\n"
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        result = self.llm.preprocess_image(tmp_path)
        Path(tmp_path).unlink()
        assert result == content

    def test_preprocess_nonexistent_path_raises_vision_llm_error(self):
        with pytest.raises(VisionLLMError, match="Failed to read image file"):
            self.llm.preprocess_image("/nonexistent/path/image.png")

    def test_preprocess_accepts_max_size_param(self):
        raw = b"\x00\x01\x02"
        result = self.llm.preprocess_image(raw, max_size=512)
        assert result == raw


# ---------------------------------------------------------------------------
# TestSettingsParsing
# ---------------------------------------------------------------------------


class TestSettingsParsing:
    """Test that VisionLLMSettings is parsed correctly from Settings.from_dict."""

    BASE_YAML_DATA: dict = {
        "llm": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "temperature": 0.0,
            "max_tokens": 256,
        },
        "embedding": {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "dimensions": 4,
        },
        "vector_store": {
            "provider": "chroma",
            "persist_directory": ":memory:",
            "collection_name": "default",
        },
        "retrieval": {
            "dense_top_k": 20,
            "sparse_top_k": 20,
            "fusion_top_k": 10,
            "rrf_k": 60,
        },
        "rerank": {"enabled": False, "provider": "none", "model": "none", "top_k": 5},
        "evaluation": {"enabled": False, "provider": "custom", "metrics": ["hit_rate"]},
        "observability": {
            "log_level": "INFO",
            "trace_enabled": False,
            "trace_file": "logs/traces.jsonl",
            "structured_logging": False,
        },
    }

    def test_vision_llm_absent_gives_none(self):
        settings = Settings.from_dict(self.BASE_YAML_DATA)
        assert settings.vision_llm is None

    def test_vision_llm_present_parsed_correctly(self):
        data = {
            **self.BASE_YAML_DATA,
            "vision_llm": {
                "provider": "azure",
                "model": "gpt-4o",
                "max_image_size": 1024,
                "azure_endpoint": "https://my.openai.azure.com",
                "api_version": "2024-02-01",
                "deployment_name": "gpt-4o-vision",
                "api_key": "test-key",
            },
        }
        settings = Settings.from_dict(data)
        assert settings.vision_llm is not None
        assert settings.vision_llm.provider == "azure"
        assert settings.vision_llm.model == "gpt-4o"
        assert settings.vision_llm.max_image_size == 1024
        assert settings.vision_llm.azure_endpoint == "https://my.openai.azure.com"
        assert settings.vision_llm.api_version == "2024-02-01"
        assert settings.vision_llm.deployment_name == "gpt-4o-vision"
        assert settings.vision_llm.api_key == "test-key"

    def test_vision_llm_default_max_image_size_is_2048(self):
        data = {
            **self.BASE_YAML_DATA,
            "vision_llm": {"provider": "azure", "model": "gpt-4o"},
        }
        settings = Settings.from_dict(data)
        assert settings.vision_llm is not None
        assert settings.vision_llm.max_image_size == 2048

    def test_vision_llm_optional_fields_default_to_none(self):
        data = {
            **self.BASE_YAML_DATA,
            "vision_llm": {"provider": "azure", "model": "gpt-4o"},
        }
        settings = Settings.from_dict(data)
        assert settings.vision_llm is not None
        assert settings.vision_llm.api_key is None
        assert settings.vision_llm.azure_endpoint is None
        assert settings.vision_llm.api_version is None
        assert settings.vision_llm.deployment_name is None

    def test_vision_llm_missing_provider_raises_settings_error(self):
        from core.settings import SettingsError

        data = {
            **self.BASE_YAML_DATA,
            "vision_llm": {"model": "gpt-4o"},
        }
        with pytest.raises(SettingsError, match="vision_llm.provider"):
            Settings.from_dict(data)

    def test_vision_llm_missing_model_raises_settings_error(self):
        from core.settings import SettingsError

        data = {
            **self.BASE_YAML_DATA,
            "vision_llm": {"provider": "azure"},
        }
        with pytest.raises(SettingsError, match="vision_llm.model"):
            Settings.from_dict(data)
