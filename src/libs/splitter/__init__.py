"""Splitter pluggable layer: importing this package registers built-in providers."""

from libs.splitter.base_splitter import BaseSplitter, SplitterError
from libs.splitter.splitter_factory import SplitterFactory

# Importing provider modules registers them with the factory (B7.5).
from libs.splitter import recursive_splitter  # noqa: F401  isort: skip

__all__ = ["BaseSplitter", "SplitterError", "SplitterFactory"]
