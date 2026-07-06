"""OpenAI-compatible embedding implementation (B7.3).

`OpenAICompatibleEmbedding` covers OpenAI itself plus any endpoint speaking
the same embeddings protocol; Azure subclasses it in `azure_embedding.py` and
only overrides client construction and model addressing.

Real HTTP goes through the official `openai` SDK; unit tests inject a fake
client via the `client` constructor argument, so no network is touched.
"""

from __future__ import annotations

from typing import Any, List, Optional

from openai import OpenAI

from core.settings import EmbeddingSettings

from libs.embedding.base_embedding import BaseEmbedding, EmbeddingError
from libs.llm.openai_llm import resolve_secret

from libs.embedding.embedding_factory import EmbeddingFactory


class OpenAICompatibleEmbedding(BaseEmbedding):
    """Batch embedding backed by any OpenAI-compatible embeddings endpoint."""

    provider_name = "openai"
    api_key_env = "OPENAI_API_KEY"
    default_base_url: Optional[str] = None

    def __init__(self, embedding_settings: EmbeddingSettings, client: Any = None) -> None:
        self.settings = embedding_settings
        self._client = client if client is not None else self._build_client()

    def _build_client(self) -> Any:
        api_key = resolve_secret(self.settings.api_key, self.api_key_env)
        if not api_key:
            raise EmbeddingError(
                f"[{self.provider_name}] missing API key: set embedding.api_key in "
                f"config/settings.yaml or the {self.api_key_env} environment variable"
            )
        base_url = self.settings.base_url or self.default_base_url
        return OpenAI(api_key=api_key, base_url=base_url)

    @property
    def _model(self) -> str:
        return self.settings.model

    def embed(self, texts: List[str], trace: Any | None = None) -> List[List[float]]:
        prepared = self._prepare_texts(texts)
        try:
            response = self._client.embeddings.create(model=self._model, input=prepared)
        except Exception as exc:
            raise EmbeddingError(
                f"[{self.provider_name}] embedding call failed ({type(exc).__name__}): {exc}"
            ) from exc
        return self._to_vectors(response, expected_count=len(prepared))

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

    def _to_vectors(self, response: Any, expected_count: int) -> List[List[float]]:
        data = getattr(response, "data", None) or []
        if len(data) != expected_count:
            raise EmbeddingError(
                f"[{self.provider_name}] expected {expected_count} embeddings, "
                f"got {len(data)}"
            )
        # The API may return items out of order; restore input order via .index.
        ordered = sorted(data, key=lambda item: getattr(item, "index", 0))
        vectors: List[List[float]] = []
        for item in ordered:
            vector = getattr(item, "embedding", None)
            if not isinstance(vector, list) or not vector:
                raise EmbeddingError(f"[{self.provider_name}] response item has no embedding vector")
            if len(vector) != self.settings.dimensions:
                raise EmbeddingError(
                    f"[{self.provider_name}] model returned {len(vector)}-dim vectors but "
                    f"embedding.dimensions={self.settings.dimensions}; fix config/settings.yaml"
                )
            vectors.append(vector)
        return vectors


class OpenAIEmbedding(OpenAICompatibleEmbedding):
    """OpenAI native API (api.openai.com), or a custom endpoint via embedding.base_url."""


EmbeddingFactory.register("openai", OpenAIEmbedding)
