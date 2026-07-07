"""Ollama embedding implementation (B7.4).

Talks to a locally running Ollama server via the `/api/embed` endpoint
(Ollama 0.2+), which natively supports batch input.

Base URL resolution: embedding.base_url → OLLAMA_BASE_URL env var →
http://localhost:11434.

Unlike cloud providers there is no API key; only the base URL matters.
HTTP goes through `httpx`; unit tests inject an `httpx.Client` built on
`httpx.MockTransport`, so no network is touched.
"""

from __future__ import annotations

import os
from typing import Any, List

import httpx

from core.settings import EmbeddingSettings

from libs.embedding.base_embedding import BaseEmbedding, EmbeddingError
from libs.embedding.embedding_factory import EmbeddingFactory

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT_SECONDS = 60.0


class OllamaEmbedding(BaseEmbedding):
    """Batch embedding backed by a local Ollama server (/api/embed protocol)."""

    provider_name = "ollama"
    base_url_env = "OLLAMA_BASE_URL"

    def __init__(self, embedding_settings: EmbeddingSettings, client: httpx.Client | None = None) -> None:
        self.settings = embedding_settings
        self.base_url = (
            embedding_settings.base_url or os.environ.get(self.base_url_env) or DEFAULT_BASE_URL
        ).rstrip("/")
        self._client = client if client is not None else httpx.Client(
            base_url=self.base_url, timeout=DEFAULT_TIMEOUT_SECONDS
        )

    def embed(self, texts: List[str], trace: Any | None = None) -> List[List[float]]:
        prepared = self._prepare_texts(texts)
        payload = {"model": self.settings.model, "input": prepared}
        try:
            response = self._client.post("/api/embed", json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.TimeoutException as exc:
            raise EmbeddingError(
                f"[{self.provider_name}] request to {self.base_url} timed out: {exc}"
            ) from exc
        except httpx.ConnectError as exc:
            raise EmbeddingError(
                f"[{self.provider_name}] cannot connect to Ollama at {self.base_url} "
                f"— is `ollama serve` running? ({exc})"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise EmbeddingError(
                f"[{self.provider_name}] HTTP {exc.response.status_code} from "
                f"{self.base_url}: {exc.response.text[:200]}"
            ) from exc
        except Exception as exc:
            raise EmbeddingError(
                f"[{self.provider_name}] embed call failed ({type(exc).__name__}): {exc}"
            ) from exc
        return self._to_vectors(data, expected_count=len(prepared))

    def _prepare_texts(self, texts: List[str]) -> List[str]:
        """Validate the batch and apply the configured oversize policy."""

        self.validate_texts(texts)
        max_chars = self.settings.max_input_chars
        prepared: List[str] = []
        for index, text in enumerate(texts):
            if not text.strip():
                raise EmbeddingError(
                    f"[{self.provider_name}] texts[{index}] is empty; "
                    "embeddings require non-empty text"
                )
            if max_chars is not None and len(text) > max_chars:
                if self.settings.truncate_oversize:
                    text = text[:max_chars]
                else:
                    raise EmbeddingError(
                        f"[{self.provider_name}] texts[{index}] has {len(text)} chars, "
                        f"exceeding embedding.max_input_chars={max_chars}; "
                        "set embedding.truncate_oversize: true to truncate instead"
                    )
            prepared.append(text)
        return prepared

    def _to_vectors(self, data: Any, expected_count: int) -> List[List[float]]:
        if not isinstance(data, dict):
            raise EmbeddingError(
                f"[{self.provider_name}] unexpected response shape: {type(data).__name__}"
            )
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list):
            raise EmbeddingError(
                f"[{self.provider_name}] response missing 'embeddings' field"
            )
        if len(embeddings) != expected_count:
            raise EmbeddingError(
                f"[{self.provider_name}] expected {expected_count} embeddings, "
                f"got {len(embeddings)}"
            )
        vectors: List[List[float]] = []
        for index, vector in enumerate(embeddings):
            if not isinstance(vector, list) or not vector:
                raise EmbeddingError(
                    f"[{self.provider_name}] embeddings[{index}] is not a valid vector"
                )
            vectors.append(vector)
        return vectors


EmbeddingFactory.register("ollama", OllamaEmbedding)
