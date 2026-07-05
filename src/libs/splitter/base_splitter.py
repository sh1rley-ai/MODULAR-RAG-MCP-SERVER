"""Splitter abstraction base class.

All splitter strategies (Recursive/Semantic/Fixed, added in B7.5+) implement
`BaseSplitter` so upper layers depend only on this contract and the strategy
can be swapped via `config/settings.yaml` (ingestion.splitter).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List


class SplitterError(RuntimeError):
    """Raised when a split call fails or its input is invalid."""


class BaseSplitter(ABC):
    """Abstract base class for text splitting strategies.

    Concrete splitters are constructed by `SplitterFactory` and receive the
    `IngestionSettings` section from the global settings as their first
    argument (chunk_size / chunk_overlap live there).
    """

    @abstractmethod
    def split_text(self, text: str, trace: Any | None = None) -> List[str]:
        """Split a text into a list of fragments.

        Fragments preserve the original reading order. `trace` is an optional
        TraceContext (Phase F).
        """

    @staticmethod
    def validate_text(text: Any) -> None:
        """Validate the input text shape; raise SplitterError with a readable message."""

        if not isinstance(text, str):
            raise SplitterError(f"text must be a string, got {type(text).__name__}")
