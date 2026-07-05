"""VectorStore abstraction base class (contract only, no real DB yet).

All vector store backends (Chroma added in B7.6) implement `BaseVectorStore`
so upper layers depend only on this contract and the backend can be swapped
via `config/settings.yaml` (vector_store.provider).

Record contract (upsert input, aligned with the future ChunkRecord in C1):
    {"id": str, "vector": list[float], "text": str?, "metadata": dict?}

Result contract (query output):
    {"id": str, "score": float, "text": str?, "metadata": dict?}
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class VectorStoreError(RuntimeError):
    """Raised when a vector store call fails or its input is invalid."""


class BaseVectorStore(ABC):
    """Abstract base class for vector store backends.

    Concrete backends are constructed by `VectorStoreFactory` and receive the
    `VectorStoreSettings` section from the global settings as their first argument.
    """

    @abstractmethod
    def upsert(self, records: List[Dict[str, Any]], trace: Any | None = None) -> None:
        """Idempotently write a batch of records (same id overwrites).

        Each record follows the record contract documented at module level.
        `trace` is an optional TraceContext (Phase F).
        """

    @abstractmethod
    def query(
        self,
        vector: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        trace: Any | None = None,
    ) -> List[Dict[str, Any]]:
        """Return up to `top_k` results, most similar first.

        Each result follows the result contract documented at module level.
        `filters` is an optional metadata equality filter.
        """

    @staticmethod
    def validate_records(records: Any) -> None:
        """Validate the upsert batch shape; raise VectorStoreError with a readable message."""

        if not isinstance(records, list) or not records:
            raise VectorStoreError("records must be a non-empty list of dicts")
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise VectorStoreError(
                    f"records[{index}] must be a dict, got {type(record).__name__}"
                )
            record_id = record.get("id")
            if not isinstance(record_id, str) or not record_id:
                raise VectorStoreError(f"records[{index}].id must be a non-empty string")
            vector = record.get("vector")
            if not isinstance(vector, list) or not vector:
                raise VectorStoreError(
                    f"records[{index}].vector must be a non-empty list of numbers"
                )
            if not all(isinstance(value, (int, float)) for value in vector):
                raise VectorStoreError(f"records[{index}].vector must contain only numbers")
            metadata = record.get("metadata")
            if metadata is not None and not isinstance(metadata, dict):
                raise VectorStoreError(f"records[{index}].metadata must be a dict when present")

    @staticmethod
    def validate_query_args(vector: Any, top_k: Any, filters: Any = None) -> None:
        """Validate query inputs; raise VectorStoreError with a readable message."""

        if not isinstance(vector, list) or not vector:
            raise VectorStoreError("vector must be a non-empty list of numbers")
        if not all(isinstance(value, (int, float)) for value in vector):
            raise VectorStoreError("vector must contain only numbers")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise VectorStoreError("top_k must be a positive integer")
        if filters is not None and not isinstance(filters, dict):
            raise VectorStoreError(f"filters must be a dict when present, got {type(filters).__name__}")
