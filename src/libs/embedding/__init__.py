"""Embedding pluggable layer: importing this package registers built-in providers."""

from libs.embedding.base_embedding import BaseEmbedding, EmbeddingError
from libs.embedding.embedding_factory import EmbeddingFactory

# Importing provider modules registers them with the factory (B7.3).
from libs.embedding import azure_embedding, openai_embedding  # noqa: F401  isort: skip

__all__ = ["BaseEmbedding", "EmbeddingError", "EmbeddingFactory"]
