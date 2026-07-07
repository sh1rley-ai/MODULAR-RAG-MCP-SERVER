"""Recursive Splitter default implementation (B7.5).

Wraps LangChain's `RecursiveCharacterTextSplitter` with Markdown-aware
separators so heading boundaries, code blocks, and paragraph breaks are
preferred over mid-sentence cuts.

Constructor receives `IngestionSettings` (chunk_size / chunk_overlap).
Unit tests inject tiny chunk sizes to keep fixture text small.
"""

from __future__ import annotations

from typing import Any, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.settings import IngestionSettings
from libs.splitter.base_splitter import BaseSplitter, SplitterError
from libs.splitter.splitter_factory import SplitterFactory

# Separator priority: Markdown headings first, then blank lines, newlines,
# spaces, and finally individual characters as last resort.
_MARKDOWN_SEPARATORS = [
    "\n# ",
    "\n## ",
    "\n### ",
    "\n#### ",
    "\n\n",
    "\n",
    " ",
    "",
]


class RecursiveSplitter(BaseSplitter):
    """Markdown-aware recursive character splitter backed by LangChain."""

    provider_name = "recursive"

    def __init__(self, ingestion_settings: IngestionSettings) -> None:
        self.settings = ingestion_settings
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=ingestion_settings.chunk_size,
            chunk_overlap=ingestion_settings.chunk_overlap,
            separators=_MARKDOWN_SEPARATORS,
            keep_separator=True,
        )

    def split_text(self, text: str, trace: Any | None = None) -> List[str]:
        self.validate_text(text)
        if not text:
            return []
        try:
            return self._splitter.split_text(text)
        except Exception as exc:
            raise SplitterError(
                f"[{self.provider_name}] split_text failed: {exc}"
            ) from exc


SplitterFactory.register("recursive", RecursiveSplitter)
