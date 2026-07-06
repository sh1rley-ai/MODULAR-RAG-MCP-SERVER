"""DeepSeek LLM implementation (B7.1).

DeepSeek exposes an OpenAI-compatible chat.completions API, so this provider
only customizes the provider name, API-key env var and default base URL.
"""

from __future__ import annotations

from libs.llm.llm_factory import LLMFactory
from libs.llm.openai_llm import OpenAICompatibleLLM


class DeepSeekLLM(OpenAICompatibleLLM):
    """DeepSeek cloud API (api.deepseek.com), OpenAI-compatible protocol."""

    provider_name = "deepseek"
    api_key_env = "DEEPSEEK_API_KEY"
    default_base_url = "https://api.deepseek.com/v1"


LLMFactory.register("deepseek", DeepSeekLLM)
