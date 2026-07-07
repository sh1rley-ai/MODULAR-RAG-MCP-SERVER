"""ChromaStore — default VectorStore backend (B7.6).

Wraps ChromaDB to provide idempotent upsert and vector query with optional
metadata filtering.  Accepts `VectorStoreSettings` from the global config.

persist_directory controls storage:
  - Any non-empty path → chromadb.PersistentClient (files on disk)
  - ":memory:" (or empty) → chromadb.EphemeralClient (in-process, no disk)

Record contract (from BaseVectorStore docstring):
    {"id": str, "vector": list[float], "text": str?, "metadata": dict?}

Result contract:
    {"id": str, "score": float, "text": str?, "metadata": dict?}

score = 1.0 / (1.0 + distance) so identical vectors yield 1.0 and
results are returned most-similar-first (chromadb already sorts by distance).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import chromadb

from core.settings import VectorStoreSettings
from libs.vector_store.base_vector_store import BaseVectorStore, VectorStoreError
from libs.vector_store.vector_store_factory import VectorStoreFactory

_MEMORY_SENTINEL = ":memory:"


class ChromaStore(BaseVectorStore):
    """ChromaDB-backed vector store (local persistent or in-memory)."""

    provider_name = "chroma"

    def __init__(self, vector_store_settings: VectorStoreSettings) -> None:
        self.settings = vector_store_settings
        persist_dir = (vector_store_settings.persist_directory or "").strip()
        if not persist_dir or persist_dir == _MEMORY_SENTINEL:
            self._client = chromadb.EphemeralClient()
        else:
            self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=vector_store_settings.collection_name
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upsert(self, records: List[Dict[str, Any]], trace: Any | None = None) -> None:
        self.validate_records(records)
        ids: List[str] = []
        embeddings: List[List[float]] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        has_metadata = False

        for record in records:
            ids.append(record["id"])
            embeddings.append(record["vector"])
            documents.append(record.get("text") or "")
            meta = record.get("metadata") or {}
            metadatas.append(meta)
            if meta:
                has_metadata = True

        kwargs: Dict[str, Any] = {
            "ids": ids,
            "embeddings": embeddings,
            "documents": documents,
        }
        # ChromaDB 1.5+ rejects empty metadata dicts; only pass metadatas when
        # at least one record has a non-empty dict.
        if has_metadata:
            kwargs["metadatas"] = metadatas

        try:
            self._collection.upsert(**kwargs)
        except Exception as exc:
            raise VectorStoreError(
                f"[{self.provider_name}] upsert failed: {exc}"
            ) from exc

    def query(
        self,
        vector: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        trace: Any | None = None,
    ) -> List[Dict[str, Any]]:
        self.validate_query_args(vector, top_k, filters)
        kwargs: Dict[str, Any] = {
            "query_embeddings": [vector],
            "n_results": top_k,
        }
        if filters:
            kwargs["where"] = filters

        try:
            raw = self._collection.query(**kwargs)
        except Exception as exc:
            raise VectorStoreError(
                f"[{self.provider_name}] query failed: {exc}"
            ) from exc

        results: List[Dict[str, Any]] = []
        ids = (raw.get("ids") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]
        documents = (raw.get("documents") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]

        for i, record_id in enumerate(ids):
            distance = distances[i] if i < len(distances) else 0.0
            score = 1.0 / (1.0 + distance)
            text = documents[i] if i < len(documents) else None
            meta = metadatas[i] if i < len(metadatas) else None
            results.append(
                {
                    "id": record_id,
                    "score": score,
                    "text": text or None,
                    "metadata": meta or {},
                }
            )
        return results


VectorStoreFactory.register("chroma", ChromaStore)
