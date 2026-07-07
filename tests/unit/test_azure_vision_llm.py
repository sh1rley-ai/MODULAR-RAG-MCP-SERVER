"""B9 — Unit tests for AzureVisionLLM.

All tests are pure unit tests: the Azure OpenAI client and Pillow are replaced
by in-test fakes (injected via the `client` constructor argument or patches).
No network is touched; no real credentials are needed.
"""

from __future__ import annotations

import base64
import io
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from core.settings import VisionLLMSettings
from libs.llm.azure_vision_llm import DEFAULT_API_VERSION, AzureVisionLLM
from libs.llm.base_llm import ChatResponse
from libs.llm.base_vision_llm import VisionLLMError
from libs.llm.llm_factory import LLMFactory


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


def make_settings(**overrides: Any) -> VisionLLMSettings:
    defaults: dict[str, Any] = {
        "provider": "azure",
        "model": "gpt-4o",
        "max_image_size": 2048,
        "api_key": "test-key",
        "azure_endpoint": "https://test.openai.azure.com/",
        "api_version": "2024-02-15-preview",
        "deployment_name": "gpt-4o-deploy",
    }
    defaults.update(overrides)
    return VisionLLMSettings(**defaults)


def make_fake_response(content: str = "image describes a cat") -> Any:
    """Build a minimal SDK-like response object."""
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return SimpleNamespace(choices=[choice], usage=usage, model="gpt-4o")


def make_client(content: str = "image describes a cat") -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.return_value = make_fake_response(content)
    return client


TINY_JPEG = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c"
    b"\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c"
    b"\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\x1eC\x15\x15CCCCCCCCCCCCCCC"
    b"CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC\xff\xc0\x00\x0b\x08\x00\x01\x00\x01"
    b"\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01"
    b"\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08"
    b"\x09\x0a\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05"
    b"\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13"
    b'Qa\x07"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n'
    b"\x16\x17\x18\x19\x1a%&'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz"
    b"\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a"
    b"\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9"
    b"\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8"
    b"\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5"
    b"\xf6\xf7\xf8\xf9\xfa\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\xd2\x8a"
    b"\x28\x03\xff\xd9"
)


# ---------------------------------------------------------------------------
# Factory registration
# ---------------------------------------------------------------------------


class TestVisionFactoryRegistration:
    def test_azure_registered_as_vision_provider(self):
        assert "azure" in LLMFactory.registered_vision_providers()

    def test_factory_creates_azure_vision_llm(self):
        settings = make_settings()
        # patch _build_client so no real network call is made
        with patch.object(AzureVisionLLM, "_build_client", return_value=MagicMock()):
            instance = LLMFactory.create_vision_llm(_make_full_settings(settings))
        assert isinstance(instance, AzureVisionLLM)

    def test_factory_raises_for_unknown_vision_provider(self):
        settings = make_settings(provider="nonexistent_vision")
        with pytest.raises(ValueError, match="Unknown Vision LLM provider"):
            LLMFactory.create_vision_llm(_make_full_settings(settings))


def _make_full_settings(vision_settings: VisionLLMSettings):
    """Wrap VisionLLMSettings into a minimal Settings object for factory tests."""
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

    return Settings(
        llm=LLMSettings(provider="openai", model="gpt-4o-mini", temperature=0.0, max_tokens=256),
        embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small", dimensions=1536),
        vector_store=VectorStoreSettings(provider="chroma", persist_directory="data/db/chroma", collection_name="test"),
        retrieval=RetrievalSettings(dense_top_k=5, sparse_top_k=5, fusion_top_k=5, rrf_k=60),
        rerank=RerankSettings(enabled=False, provider="none", model="", top_k=5),
        evaluation=EvaluationSettings(enabled=False, provider="custom", metrics=["hit_rate"]),
        observability=ObservabilitySettings(log_level="INFO", trace_enabled=False, trace_file="trace.jsonl", structured_logging=False),
        vision_llm=vision_settings,
    )


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------


class TestClientConstruction:
    def test_uses_injected_client(self):
        fake_client = MagicMock()
        llm = AzureVisionLLM(make_settings(), client=fake_client)
        assert llm._client is fake_client

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        settings = make_settings(api_key=None)
        with pytest.raises(VisionLLMError, match="missing API key"):
            AzureVisionLLM(settings)

    def test_missing_endpoint_raises(self, monkeypatch):
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        settings = make_settings(azure_endpoint=None)
        with pytest.raises(VisionLLMError, match="missing endpoint"):
            AzureVisionLLM(settings)

    def test_env_var_api_key(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://env.openai.azure.com/")
        settings = make_settings(api_key=None, azure_endpoint=None)
        with patch("libs.llm.azure_vision_llm.AzureOpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            llm = AzureVisionLLM(settings)
        mock_cls.assert_called_once_with(
            api_key="env-key",
            azure_endpoint="https://env.openai.azure.com/",
            api_version="2024-02-15-preview",
        )

    def test_default_api_version_used_when_unset(self, monkeypatch):
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        settings = make_settings(api_version=None)
        with patch("libs.llm.azure_vision_llm.AzureOpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            AzureVisionLLM(settings)
        _, kwargs = mock_cls.call_args
        assert kwargs["api_version"] == DEFAULT_API_VERSION

    def test_deployment_name_takes_priority_over_model(self):
        client = make_client()
        llm = AzureVisionLLM(make_settings(deployment_name="my-deploy", model="gpt-4o"), client=client)
        assert llm._deployment == "my-deploy"

    def test_model_used_when_no_deployment_name(self):
        client = make_client()
        llm = AzureVisionLLM(make_settings(deployment_name=None, model="gpt-4o"), client=client)
        assert llm._deployment == "gpt-4o"


# ---------------------------------------------------------------------------
# Normal chat_with_image call
# ---------------------------------------------------------------------------


class TestChatWithImage:
    def test_returns_chat_response_with_bytes_input(self):
        client = make_client("a dog on a couch")
        llm = AzureVisionLLM(make_settings(), client=client)
        result = llm.chat_with_image("What is in the image?", TINY_JPEG)
        assert isinstance(result, ChatResponse)
        assert result.content == "a dog on a couch"

    def test_returns_chat_response_with_path_input(self, tmp_path):
        img_path = tmp_path / "test.jpg"
        img_path.write_bytes(TINY_JPEG)
        client = make_client("a flower")
        llm = AzureVisionLLM(make_settings(), client=client)
        result = llm.chat_with_image("Describe this.", str(img_path))
        assert result.content == "a flower"

    def test_data_uri_passed_in_message(self):
        client = make_client()
        llm = AzureVisionLLM(make_settings(), client=client)
        llm.chat_with_image("describe", TINY_JPEG)
        call_args = client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages") or call_args.args[0] if call_args.args else call_args.kwargs["messages"]
        content = messages[0]["content"]
        image_part = next(p for p in content if p["type"] == "image_url")
        assert image_part["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_text_included_in_message(self):
        client = make_client()
        llm = AzureVisionLLM(make_settings(), client=client)
        llm.chat_with_image("What do you see?", TINY_JPEG)
        call_args = client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages") or call_args.kwargs["messages"]
        content = messages[0]["content"]
        text_part = next(p for p in content if p["type"] == "text")
        assert text_part["text"] == "What do you see?"

    def test_deployment_name_used_in_create_call(self):
        client = make_client()
        llm = AzureVisionLLM(make_settings(deployment_name="prod-deploy"), client=client)
        llm.chat_with_image("test", TINY_JPEG)
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "prod-deploy"

    def test_usage_included_in_response(self):
        client = make_client()
        llm = AzureVisionLLM(make_settings(), client=client)
        result = llm.chat_with_image("test", TINY_JPEG)
        assert result.usage.get("total_tokens") == 15

    def test_model_from_response_propagated(self):
        client = make_client()
        llm = AzureVisionLLM(make_settings(), client=client)
        result = llm.chat_with_image("test", TINY_JPEG)
        assert result.model == "gpt-4o"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_empty_text_raises(self):
        client = make_client()
        llm = AzureVisionLLM(make_settings(), client=client)
        with pytest.raises(VisionLLMError, match="non-empty string"):
            llm.chat_with_image("   ", TINY_JPEG)

    def test_non_string_text_raises(self):
        client = make_client()
        llm = AzureVisionLLM(make_settings(), client=client)
        with pytest.raises(VisionLLMError):
            llm.chat_with_image(42, TINY_JPEG)  # type: ignore[arg-type]

    def test_invalid_image_path_type_raises(self):
        client = make_client()
        llm = AzureVisionLLM(make_settings(), client=client)
        with pytest.raises(VisionLLMError):
            llm.chat_with_image("text", 12345)  # type: ignore[arg-type]

    def test_empty_string_path_raises(self):
        client = make_client()
        llm = AzureVisionLLM(make_settings(), client=client)
        with pytest.raises(VisionLLMError):
            llm.chat_with_image("text", "   ")

    def test_empty_bytes_raises(self):
        client = make_client()
        llm = AzureVisionLLM(make_settings(), client=client)
        with pytest.raises(VisionLLMError):
            llm.chat_with_image("text", b"")

    def test_nonexistent_file_path_raises(self):
        client = make_client()
        llm = AzureVisionLLM(make_settings(), client=client)
        with pytest.raises(VisionLLMError, match="failed to read image"):
            llm.chat_with_image("text", "/nonexistent/path/image.jpg")


# ---------------------------------------------------------------------------
# Image compression
# ---------------------------------------------------------------------------


class TestImageCompression:
    def _make_png_bytes(self, width: int, height: int) -> bytes:
        """Create a real PNG image using Pillow."""
        from PIL import Image as PILImage

        img = PILImage.new("RGB", (width, height), color=(128, 64, 32))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def test_small_image_not_compressed(self):
        """Image within max_size is returned unchanged."""
        raw = self._make_png_bytes(100, 100)
        client = make_client()
        llm = AzureVisionLLM(make_settings(max_image_size=512), client=client)
        result = llm.preprocess_image(raw, max_size=512)
        assert result == raw  # identical bytes returned

    def test_large_image_compressed(self):
        """Image exceeding max_size is shrunk."""
        raw = self._make_png_bytes(4000, 3000)
        client = make_client()
        llm = AzureVisionLLM(make_settings(max_image_size=512), client=client)
        result = llm.preprocess_image(raw, max_size=512)

        from PIL import Image as PILImage

        img = PILImage.open(io.BytesIO(result))
        assert max(img.width, img.height) <= 512

    def test_compression_uses_settings_max_image_size(self):
        """chat_with_image honours settings.max_image_size."""
        raw = self._make_png_bytes(3000, 3000)
        client = make_client()
        llm = AzureVisionLLM(make_settings(max_image_size=256), client=client)
        llm.chat_with_image("describe", raw)

        call_kwargs = client.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        content = messages[0]["content"]
        image_part = next(p for p in content if p["type"] == "image_url")
        img_bytes = base64.b64decode(image_part["image_url"]["url"].split(",", 1)[1])

        from PIL import Image as PILImage

        img = PILImage.open(io.BytesIO(img_bytes))
        assert max(img.width, img.height) <= 256

    def test_image_loaded_from_path_before_compression(self, tmp_path):
        """File-path input is loaded then compression is applied."""
        raw = self._make_png_bytes(3000, 2000)
        p = tmp_path / "big.png"
        p.write_bytes(raw)

        client = make_client()
        llm = AzureVisionLLM(make_settings(max_image_size=512), client=client)
        result = llm.preprocess_image(str(p), max_size=512)

        from PIL import Image as PILImage

        img = PILImage.open(io.BytesIO(result))
        assert max(img.width, img.height) <= 512


# ---------------------------------------------------------------------------
# API error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_api_exception_wrapped_in_vision_llm_error(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("network down")
        llm = AzureVisionLLM(make_settings(), client=client)
        with pytest.raises(VisionLLMError, match="API call failed"):
            llm.chat_with_image("test", TINY_JPEG)

    def test_azure_error_code_included_in_message(self):
        client = MagicMock()
        err = RuntimeError("auth failed")
        err.code = "AuthenticationFailed"  # type: ignore[attr-defined]
        client.chat.completions.create.side_effect = err
        llm = AzureVisionLLM(make_settings(), client=client)
        with pytest.raises(VisionLLMError, match="AuthenticationFailed"):
            llm.chat_with_image("test", TINY_JPEG)

    def test_timeout_exception_wrapped(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = TimeoutError("request timed out")
        llm = AzureVisionLLM(make_settings(), client=client)
        with pytest.raises(VisionLLMError, match="API call failed"):
            llm.chat_with_image("test", TINY_JPEG)

    def test_empty_choices_raises(self):
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(choices=[], usage=None, model="")
        llm = AzureVisionLLM(make_settings(), client=client)
        with pytest.raises(VisionLLMError, match="no choices"):
            llm.chat_with_image("test", TINY_JPEG)

    def test_missing_content_raises(self):
        client = MagicMock()
        choice = SimpleNamespace(message=SimpleNamespace(content=None))
        client.chat.completions.create.return_value = SimpleNamespace(choices=[choice], usage=None, model="")
        llm = AzureVisionLLM(make_settings(), client=client)
        with pytest.raises(VisionLLMError, match="content is missing"):
            llm.chat_with_image("test", TINY_JPEG)

    def test_status_code_included_in_message_when_present(self):
        client = MagicMock()
        err = RuntimeError("rate limit")
        err.status_code = 429  # type: ignore[attr-defined]
        client.chat.completions.create.side_effect = err
        llm = AzureVisionLLM(make_settings(), client=client)
        with pytest.raises(VisionLLMError, match="429"):
            llm.chat_with_image("test", TINY_JPEG)
