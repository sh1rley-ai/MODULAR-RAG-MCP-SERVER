"""VectorStore pluggable layer: importing this package registers built-in providers."""

from libs.vector_store.base_vector_store import BaseVectorStore, VectorStoreError
from libs.vector_store.vector_store_factory import VectorStoreFactory

# Importing provider modules registers them with the factory (B7.6).
from libs.vector_store import chroma_store  # noqa: F401  isort: skip

__all__ = ["BaseVectorStore", "VectorStoreError", "VectorStoreFactory"]
