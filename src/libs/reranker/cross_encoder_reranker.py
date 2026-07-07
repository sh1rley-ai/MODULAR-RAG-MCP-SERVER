"""CrossEncoderReranker — reranks candidates using a cross-encoder model (B7.8).

A cross-encoder scores each (query, candidate_text) pair jointly, producing a
float relevance score. Candidates are reordered by descending score.

Production path: lazily imports `sentence_transformers.CrossEncoder` using the
model name from `rerank_settings.model`. The import is deferred so the library
is optional — only required when this backend is actually instantiated.

Test path: an injectable `scorer` callable bypasses the real model entirely,
keeping unit tests fast and deterministic.

Scorer protocol:
    scorer(query: str, pairs: List[Tuple[str, str]]) -> List[float]

    pairs is [(query, text) for each candidate].
    Must return exactly len(pairs) floats in the same order.

On any failure (import error, scorer exception, wrong output shape) a
`RerankerError` is raised so the Core layer (D6) can catch it and fall back to
the fusion-ranked list.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from libs.reranker.base_reranker import BaseReranker, RerankerError
from libs.reranker.reranker_factory import RerankerFactory


class CrossEncoderReranker(BaseReranker):
    """Reranker backed by a cross-encoder model.

    Args:
        rerank_settings: RerankSettings from global config (``model``, ``top_k``).
        scorer: Optional injectable scorer callable. When provided it replaces the
                sentence-transformers model — intended for unit tests only.
    """

    provider_name = "cross_encoder"

    def __init__(
        self,
        rerank_settings: Any,
        scorer: Optional[Callable[[str, List[Tuple[str, str]]], List[float]]] = None,
    ) -> None:
        self.settings = rerank_settings
        self._scorer = scorer

        if scorer is None:
            # Eagerly validate that sentence_transformers is installed so callers
            # get a clear error at construction time rather than at rerank() time.
            model_name = getattr(rerank_settings, "model", None) or ""
            if not model_name or model_name.lower() == "none":
                raise RerankerError(
                    "[cross_encoder] rerank.model must be set to a valid cross-encoder "
                    "model name (e.g. 'cross-encoder/ms-marco-MiniLM-L-6-v2'). "
                    "Got: {!r}".format(model_name)
                )
            try:
                from sentence_transformers import CrossEncoder  # type: ignore

                self._model = CrossEncoder(model_name)
            except ImportError as exc:
                raise RerankerError(
                    "[cross_encoder] sentence-transformers is not installed. "
                    "Install it with: pip install sentence-transformers"
                ) from exc
            except Exception as exc:
                raise RerankerError(
                    f"[cross_encoder] failed to load model {model_name!r}: {exc}"
                ) from exc
        else:
            self._model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        trace: Any | None = None,
    ) -> List[Dict[str, Any]]:
        self.validate_query(query)
        self.validate_candidates(candidates)

        if not candidates:
            return []

        pairs: List[Tuple[str, str]] = [
            (query, (c.get("text") or "").strip()) for c in candidates
        ]

        scores = self._score(query, pairs)

        if len(scores) != len(candidates):
            raise RerankerError(
                f"[cross_encoder] scorer returned {len(scores)} scores for "
                f"{len(candidates)} candidates"
            )

        indexed = sorted(
            zip(scores, range(len(candidates))),
            key=lambda x: x[0],
            reverse=True,
        )
        return [candidates[i] for _, i in indexed]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _score(self, query: str, pairs: List[Tuple[str, str]]) -> List[float]:
        if self._scorer is not None:
            try:
                result = self._scorer(query, pairs)
            except Exception as exc:
                raise RerankerError(
                    f"[cross_encoder] scorer raised an exception: {exc}"
                ) from exc
            return [float(s) for s in result]

        # Real sentence-transformers path
        try:
            raw = self._model.predict(pairs)
        except Exception as exc:
            raise RerankerError(
                f"[cross_encoder] model.predict failed: {exc}"
            ) from exc
        return [float(s) for s in raw]


RerankerFactory.register("cross_encoder", CrossEncoderReranker)
