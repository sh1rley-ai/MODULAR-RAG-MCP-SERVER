"""
Modular RAG MCP Server - Main Entry Point

This is the entry point for the MCP Server. It initializes the configuration,
sets up logging, and starts the server.
"""

import sys
from pathlib import Path

# Ensure src/ is on the path when running directly (without `pip install -e .`).
sys.path.insert(0, str(Path(__file__).parent / "src"))


def main() -> int:
    """
    Main entry point for the MCP Server.

    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    from core.settings import load_settings
    from observability.logger import get_logger

    logger = get_logger(__name__)

    try:
        settings = load_settings("config/settings.yaml")
    except (FileNotFoundError, ValueError) as exc:
        # Fail-fast: missing config or invalid required fields → non-zero exit.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    logger.info(
        "Modular RAG MCP Server starting — "
        f"LLM: {settings.llm.provider}, "
        f"Embedding: {settings.embedding.provider}, "
        f"VectorStore: {settings.vector_store.provider}"
    )
    print("MCP Server will be implemented in Phase E.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
