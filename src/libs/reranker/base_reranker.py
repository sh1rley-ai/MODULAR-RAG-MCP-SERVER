"""Reranker abstraction base class with the None fallback implementation.

All reranker backends (CrossEncoder/LLM added in B7.7/B7.8) implement
`BaseReranker` so upper layers depend only on this contract and the backend
can be swapped via `config/settings.yaml` (rerank.provider).

Candidate contract (rerank input/output, aligned with the vector store
query result contract):
    {"id": str, "score": float?, "text": str?, "metadata": dict?}

`rerank` returns the same candidates reordered (most relevant first). It must
not drop or invent candidates — truncation to `rerank.top_k` is the Core
layer's responsibility (D6).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class RerankerError(RuntimeError):
    """Raised when a rerank call fails or its input is invalid."""


class BaseReranker(ABC):
    """Abstract base class for reranker backends.

    Concrete backends are constructed by `RerankerFactory` and receive the
    `RerankSettings` section from the global settings as their first argument
    (model / top_k live there).
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Any | None = None,
    ) -> List[Dict[str, Any]]:
        """Return the candidates reordered by relevance to `query`.

        Each candidate follows the candidate contract documented at module
        level. `trace` is an optional TraceContext (Phase F).
        """

    @staticmethod
    def validate_query(query: Any) -> None:
        """Validate the query shape; raise RerankerError with a readable message."""

        if not isinstance(query, str) or not query.strip():
            raise RerankerError(
                f"query must be a non-empty string, got {type(query).__name__}"
            )

    @staticmethod
    def validate_candidates(candidates: Any) -> None:
        """Validate the candidate batch shape; raise RerankerError with a readable message."""

        if not isinstance(candidates, list):
            raise RerankerError(
                f"candidates must be a list of dicts, got {type(candidates).__name__}"
            )
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                raise RerankerError(
                    f"candidates[{index}] must be a dict, got {type(candidate).__name__}"
                )
            candidate_id = candidate.get("id")
            if not isinstance(candidate_id, str) or not candidate_id:
                raise RerankerError(f"candidates[{index}].id must be a non-empty string")


class NoneReranker(BaseReranker):
    """Default fallback backend: keeps the original candidate order untouched.

    Used when reranking is disabled (rerank.enabled=false) or provider=none,
    and as the safe fallback target when a real backend fails (D6).
    """

    def __init__(self, rerank_settings: Any) -> None:
        self.settings = rerank_settings

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Any | None = None,
    ) -> List[Dict[str, Any]]:
        self.validate_query(query)
        self.validate_candidates(candidates)
        return list(candidates)
