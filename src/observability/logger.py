"""Observability logger placeholder.

Provides a get_logger() factory backed by the stdlib logging module with a
stderr StreamHandler. Phase F (F2) will replace this with a JSON-Lines
structured logger; the signature stays the same so callers need no changes.
"""

import logging
import sys


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a named logger writing to stderr at the given level.

    Args:
        name:  Logger name, typically ``__name__`` of the calling module.
        level: Logging level string (DEBUG / INFO / WARNING / ERROR).
    """
    logger = logging.getLogger(name)

    # Only add a handler once to avoid duplicates when called multiple times.
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger
