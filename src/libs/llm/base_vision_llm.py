"""Vision LLM abstraction base class (B8).

All vision-capable LLM providers (e.g. AzureVisionLLM added in B9) implement
`BaseVisionLLM` so upper layers depend only on this contract and the provider
can be swapped via `config/settings.yaml` (vision_llm.provider).

Image input contract:
    image_path: str  — absolute or relative filesystem path to the image file
    image_path: bytes — raw image bytes (JPEG/PNG/WebP/GIF), already in memory

Subclasses may override `preprocess_image` to apply resize/format conversion
before sending to the API (the extension point described in B8 spec).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Union

from libs.llm.base_llm import ChatResponse, LLMError


class VisionLLMError(LLMError):
    """Raised when a vision LLM call fails or its input is invalid."""


class BaseVisionLLM(ABC):
    """Abstract base class for vision-capable LLM providers.

    Concrete providers are constructed by `LLMFactory.create_vision_llm` and
    receive the `VisionLLMSettings` section from the global settings as their
    first argument.
    """

    @abstractmethod
    def chat_with_image(
        self,
        text: str,
        image_path: Union[str, bytes],
        trace: Any | None = None,
    ) -> ChatResponse:
        """Send a text prompt alongside an image and return the model response.

        Args:
            text:        The text prompt (question or instruction about the image).
            image_path:  Either a filesystem path string or raw image bytes.
            trace:       Optional TraceContext (Phase F).

        Returns:
            ChatResponse with the model's reply.

        Raises:
            VisionLLMError: On API failure, invalid input, or unsupported image format.
        """

    def preprocess_image(
        self,
        image_path: Union[str, bytes],
        max_size: int = 2048,
    ) -> bytes:
        """Load and optionally resize the image to fit within `max_size` pixels.

        This is an extension point for subclasses. The default implementation
        reads the file (if a path string) or returns the bytes as-is, without
        any resize. Subclasses that need real resize logic (e.g. AzureVisionLLM
        in B9) should override this method.

        Args:
            image_path: Filesystem path string or raw image bytes.
            max_size:   Maximum edge length in pixels (no-op in base implementation).

        Returns:
            Raw image bytes ready to send to the API.
        """
        if isinstance(image_path, bytes):
            return image_path
        try:
            with open(image_path, "rb") as f:
                return f.read()
        except OSError as exc:
            raise VisionLLMError(
                f"Failed to read image file {image_path!r}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Validation helpers (shared across subclasses)
    # ------------------------------------------------------------------

    @staticmethod
    def validate_text(text: Any) -> None:
        """Raise VisionLLMError if text is not a non-empty string."""
        if not isinstance(text, str) or not text.strip():
            raise VisionLLMError(
                f"text must be a non-empty string, got {type(text).__name__}"
            )

    @staticmethod
    def validate_image_path(image_path: Any) -> None:
        """Raise VisionLLMError if image_path is not a str or bytes."""
        if not isinstance(image_path, (str, bytes)):
            raise VisionLLMError(
                f"image_path must be a str (file path) or bytes (raw image data), "
                f"got {type(image_path).__name__}"
            )
        if isinstance(image_path, str) and not image_path.strip():
            raise VisionLLMError("image_path string must not be empty")
        if isinstance(image_path, bytes) and len(image_path) == 0:
            raise VisionLLMError("image_path bytes must not be empty")
