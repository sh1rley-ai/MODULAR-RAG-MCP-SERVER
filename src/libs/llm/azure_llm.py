"""Azure OpenAI LLM implementation (B7.1).

Azure speaks the OpenAI chat.completions protocol but requires its own client
configuration (endpoint + api-version) and addresses models by deployment
name, so only client construction and model resolution differ from the base.
"""

from __future__ import annotations

from typing import Any

from openai import AzureOpenAI

from libs.llm.base_llm import LLMError
from libs.llm.llm_factory import LLMFactory
from libs.llm.openai_llm import OpenAICompatibleLLM, resolve_secret

DEFAULT_API_VERSION = "2024-06-01"


class AzureLLM(OpenAICompatibleLLM):
    """Azure OpenAI service; model is addressed by deployment name."""

    provider_name = "azure"
    api_key_env = "AZURE_OPENAI_API_KEY"
    endpoint_env = "AZURE_OPENAI_ENDPOINT"

    def _build_client(self) -> Any:
        api_key = resolve_secret(self.settings.api_key, self.api_key_env)
        if not api_key:
            raise LLMError(
                f"[{self.provider_name}] missing API key: set llm.api_key in "
                f"config/settings.yaml or the {self.api_key_env} environment variable"
            )
        endpoint = resolve_secret(self.settings.azure_endpoint, self.endpoint_env)
        if not endpoint:
            raise LLMError(
                f"[{self.provider_name}] missing endpoint: set llm.azure_endpoint in "
                f"config/settings.yaml or the {self.endpoint_env} environment variable"
            )
        api_version = self.settings.api_version or DEFAULT_API_VERSION
        return AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)

    @property
    def _model(self) -> str:
        return self.settings.deployment_name or self.settings.model


LLMFactory.register("azure", AzureLLM)
