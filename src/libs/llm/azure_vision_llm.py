"""Azure OpenAI Vision LLM implementation (B9).

Sends text + image to Azure's GPT-4o / GPT-4-Vision-Preview via the
chat.completions multimodal API (image_url with a data-URI).

Image input contract (inherited from BaseVisionLLM):
    image_path: str   — filesystem path; loaded and compressed before upload
    image_path: bytes — raw image bytes; compressed if larger than max_image_size

Auto-compression: if either dimension exceeds VisionLLMSettings.max_image_size
(default 2048 px), the image is shrunk proportionally using Pillow (LANCZOS)
and re-encoded as JPEG. Bytes inputs bypass the file-read step.

Required config (vision_llm section in settings.yaml):
    provider: azure
    model: gpt-4o          # or gpt-4-vision-preview
    azure_endpoint: https://<resource>.openai.azure.com/
    api_version: "2024-02-15-preview"
    deployment_name: my-gpt4o-deployment
    api_key: <key>         # or AZURE_OPENAI_API_KEY env var
"""

from __future__ import annotations

import base64
import io
import os
import re
from typing import Any, Optional

from openai import AzureOpenAI

from core.settings import VisionLLMSettings
from libs.llm.base_llm import ChatResponse
from libs.llm.base_vision_llm import BaseVisionLLM, VisionLLMError
from libs.llm.llm_factory import LLMFactory

_ENV_REF = re.compile(r"^\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)\}$")

DEFAULT_API_VERSION = "2024-02-15-preview"
_DEFAULT_MAX_IMAGE_SIZE = 2048


def _resolve(value: Optional[str], env_var: str) -> Optional[str]:
    """Resolve a literal, ${ENV_VAR} reference, or environment variable."""
    if value:
        m = _ENV_REF.match(value.strip())
        if m:
            return os.environ.get(m.group("name"))
        return value
    return os.environ.get(env_var)


class AzureVisionLLM(BaseVisionLLM):
    """Azure OpenAI vision-capable LLM (GPT-4o / GPT-4-Vision-Preview)."""

    provider_name = "azure"

    def __init__(self, vision_settings: VisionLLMSettings, client: Any = None) -> None:
        self.settings = vision_settings
        self._client = client if client is not None else self._build_client()

    def _build_client(self) -> AzureOpenAI:
        api_key = _resolve(self.settings.api_key, "AZURE_OPENAI_API_KEY")
        if not api_key:
            raise VisionLLMError(
                "[azure_vision] missing API key: set vision_llm.api_key in "
                "config/settings.yaml or the AZURE_OPENAI_API_KEY environment variable"
            )
        endpoint = _resolve(self.settings.azure_endpoint, "AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise VisionLLMError(
                "[azure_vision] missing endpoint: set vision_llm.azure_endpoint in "
                "config/settings.yaml or the AZURE_OPENAI_ENDPOINT environment variable"
            )
        api_version = self.settings.api_version or DEFAULT_API_VERSION
        return AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)

    @property
    def _deployment(self) -> str:
        return self.settings.deployment_name or self.settings.model

    # ------------------------------------------------------------------
    # Image preprocessing (override: applies real Pillow resize)
    # ------------------------------------------------------------------

    def preprocess_image(
        self,
        image_path: str | bytes,
        max_size: int = _DEFAULT_MAX_IMAGE_SIZE,
    ) -> bytes:
        """Load image from path or bytes and compress if any edge > max_size."""
        raw = self._load_raw(image_path)
        return self._compress(raw, max_size)

    def _load_raw(self, image_path: str | bytes) -> bytes:
        if isinstance(image_path, bytes):
            return image_path
        try:
            with open(image_path, "rb") as f:
                return f.read()
        except OSError as exc:
            raise VisionLLMError(
                f"[azure_vision] failed to read image file {image_path!r}: {exc}"
            ) from exc

    def _compress(self, raw: bytes, max_size: int) -> bytes:
        try:
            from PIL import Image  # lazy import — keeps module loadable without Pillow
        except ImportError as exc:
            raise VisionLLMError(
                "[azure_vision] image compression requires Pillow: "
                "pip install pillow"
            ) from exc

        try:
            img = Image.open(io.BytesIO(raw))
        except Exception as exc:
            raise VisionLLMError(
                f"[azure_vision] cannot decode image data: {exc}"
            ) from exc

        if max(img.width, img.height) <= max_size:
            return raw  # no resize needed — return original bytes as-is

        img.thumbnail((max_size, max_size), Image.LANCZOS)
        buf = io.BytesIO()
        fmt = img.format or "JPEG"
        if fmt.upper() not in ("JPEG", "PNG", "WEBP"):
            fmt = "JPEG"
        img.save(buf, format=fmt)
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def chat_with_image(
        self,
        text: str,
        image_path: str | bytes,
        trace: Any | None = None,
    ) -> ChatResponse:
        """Send text + image to Azure OpenAI and return the model reply."""
        self.validate_text(text)
        self.validate_image_path(image_path)

        max_size = self.settings.max_image_size or _DEFAULT_MAX_IMAGE_SIZE
        image_bytes = self.preprocess_image(image_path, max_size=max_size)
        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_uri = f"data:image/jpeg;base64,{b64}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ]

        try:
            response = self._client.chat.completions.create(
                model=self._deployment,
                messages=messages,
            )
        except Exception as exc:
            error_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
            suffix = f" (Azure error code: {error_code})" if error_code else ""
            raise VisionLLMError(
                f"[azure_vision] API call failed ({type(exc).__name__}): {exc}{suffix}"
            ) from exc

        return self._parse_response(response)

    def _parse_response(self, response: Any) -> ChatResponse:
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise VisionLLMError("[azure_vision] response contained no choices")
        content = getattr(getattr(choices[0], "message", None), "content", None)
        if not isinstance(content, str):
            raise VisionLLMError("[azure_vision] response message content is missing")

        usage_dict: dict = {}
        usage = getattr(response, "usage", None)
        if usage is not None:
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                val = getattr(usage, key, None)
                if val is not None:
                    usage_dict[key] = val

        model = getattr(response, "model", "") or self._deployment
        return ChatResponse(content=content, model=model, usage=usage_dict)


LLMFactory.register_vision("azure", AzureVisionLLM)
