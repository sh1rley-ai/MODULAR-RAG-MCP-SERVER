"""Azure OpenAI embedding implementation (B7.3).

Same embeddings protocol as OpenAI, but Azure requires its own client
configuration (endpoint + api-version) and addresses models by deployment
name — so only client construction and model resolution differ from the base.
"""

from __future__ import annotations

from typing import Any

from openai import AzureOpenAI

from libs.embedding.base_embedding import EmbeddingError
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.embedding.openai_embedding import OpenAICompatibleEmbedding
from libs.llm.openai_llm import resolve_secret

DEFAULT_API_VERSION = "2024-06-01"


class AzureEmbedding(OpenAICompatibleEmbedding):
    """Azure OpenAI embeddings; model is addressed by deployment name."""

    provider_name = "azure"
    api_key_env = "AZURE_OPENAI_API_KEY"
    endpoint_env = "AZURE_OPENAI_ENDPOINT"

    def _build_client(self) -> Any:
        api_key = resolve_secret(self.settings.api_key, self.api_key_env)
        if not api_key:
            raise EmbeddingError(
                f"[{self.provider_name}] missing API key: set embedding.api_key in "
                f"config/settings.yaml or the {self.api_key_env} environment variable"
            )
        endpoint = resolve_secret(self.settings.azure_endpoint, self.endpoint_env)
        if not endpoint:
            raise EmbeddingError(
                f"[{self.provider_name}] missing endpoint: set embedding.azure_endpoint in "
                f"config/settings.yaml or the {self.endpoint_env} environment variable"
            )
        api_version = self.settings.api_version or DEFAULT_API_VERSION
        return AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)

    @property
    def _model(self) -> str:
        return self.settings.deployment_name or self.settings.model


EmbeddingFactory.register("azure", AzureEmbedding)
