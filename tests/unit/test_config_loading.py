"""A3 — Tests for config loading and validation (Settings).

All tests are pure unit tests: they write temporary YAML files and call
load_settings() directly. No network, no external services.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from core.settings import (
    Settings,
    SettingsError,
    load_settings,
    validate_settings,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Full valid YAML matching the new required schema in settings.py
FULL_VALID_YAML = textwrap.dedent("""\
    llm:
      provider: "openai"
      model: "gpt-4o-mini"
      temperature: 0.0
      max_tokens: 2048
    embedding:
      provider: "openai"
      model: "text-embedding-3-small"
      dimensions: 1536
    vector_store:
      provider: "chroma"
      persist_directory: "data/db/chroma"
      collection_name: "default"
    retrieval:
      dense_top_k: 20
      sparse_top_k: 20
      fusion_top_k: 10
      rrf_k: 60
    rerank:
      enabled: false
      provider: "none"
      model: "none"
      top_k: 5
    evaluation:
      enabled: false
      provider: "custom"
      metrics:
        - "hit_rate"
        - "mrr"
    observability:
      log_level: "INFO"
      trace_enabled: true
      trace_file: "logs/traces.jsonl"
      structured_logging: false
""")


def write_yaml(tmp_path: Path, content: str) -> Path:
    cfg = tmp_path / "settings.yaml"
    cfg.write_text(content, encoding="utf-8")
    return cfg


# ---------------------------------------------------------------------------
# TestLoadSettings
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestLoadSettings:
    """Tests for load_settings() parsing and structure."""

    def test_load_valid_config(self, tmp_path: Path) -> None:
        """load_settings returns a Settings object for valid YAML."""
        cfg = write_yaml(tmp_path, FULL_VALID_YAML)
        settings = load_settings(str(cfg))
        assert isinstance(settings, Settings)

    def test_llm_fields_parsed(self, tmp_path: Path) -> None:
        """llm section fields are read correctly from YAML."""
        cfg = write_yaml(tmp_path, FULL_VALID_YAML)
        settings = load_settings(str(cfg))
        assert settings.llm.provider == "openai"
        assert settings.llm.model == "gpt-4o-mini"
        assert settings.llm.temperature == pytest.approx(0.0)
        assert settings.llm.max_tokens == 2048

    def test_embedding_fields_parsed(self, tmp_path: Path) -> None:
        """embedding section fields are read correctly from YAML."""
        cfg = write_yaml(tmp_path, FULL_VALID_YAML)
        settings = load_settings(str(cfg))
        assert settings.embedding.provider == "openai"
        assert settings.embedding.model == "text-embedding-3-small"
        assert settings.embedding.dimensions == 1536

    def test_vector_store_fields_parsed(self, tmp_path: Path) -> None:
        """vector_store section fields are read correctly from YAML."""
        cfg = write_yaml(tmp_path, FULL_VALID_YAML)
        settings = load_settings(str(cfg))
        assert settings.vector_store.provider == "chroma"
        assert settings.vector_store.collection_name == "default"

    def test_retrieval_fields_parsed(self, tmp_path: Path) -> None:
        """retrieval section fields are read correctly, including fusion_top_k."""
        cfg = write_yaml(tmp_path, FULL_VALID_YAML)
        settings = load_settings(str(cfg))
        assert settings.retrieval.dense_top_k == 20
        assert settings.retrieval.fusion_top_k == 10
        assert settings.retrieval.rrf_k == 60

    def test_rerank_fields_parsed(self, tmp_path: Path) -> None:
        """rerank section fields are read correctly."""
        cfg = write_yaml(tmp_path, FULL_VALID_YAML)
        settings = load_settings(str(cfg))
        assert settings.rerank.enabled is False
        assert settings.rerank.provider == "none"
        assert settings.rerank.top_k == 5

    def test_evaluation_fields_parsed(self, tmp_path: Path) -> None:
        """evaluation section fields including metrics list are read correctly."""
        cfg = write_yaml(tmp_path, FULL_VALID_YAML)
        settings = load_settings(str(cfg))
        assert settings.evaluation.enabled is False
        assert settings.evaluation.provider == "custom"
        assert "hit_rate" in settings.evaluation.metrics

    def test_observability_fields_parsed(self, tmp_path: Path) -> None:
        """observability section fields are read correctly."""
        cfg = write_yaml(tmp_path, FULL_VALID_YAML)
        settings = load_settings(str(cfg))
        assert settings.observability.log_level == "INFO"
        assert settings.observability.trace_enabled is True
        assert settings.observability.trace_file == "logs/traces.jsonl"
        assert settings.observability.structured_logging is False

    def test_ingestion_section_optional(self, tmp_path: Path) -> None:
        """ingestion section is optional; Settings.ingestion is None when absent."""
        cfg = write_yaml(tmp_path, FULL_VALID_YAML)  # no ingestion section
        settings = load_settings(str(cfg))
        assert settings.ingestion is None

    def test_ingestion_section_parsed_when_present(self, tmp_path: Path) -> None:
        """ingestion section is parsed when present in YAML."""
        yaml_with_ingestion = FULL_VALID_YAML + textwrap.dedent("""\
            ingestion:
              chunk_size: 500
              chunk_overlap: 100
              splitter: "recursive"
              batch_size: 8
        """)
        cfg = write_yaml(tmp_path, yaml_with_ingestion)
        settings = load_settings(str(cfg))
        assert settings.ingestion is not None
        assert settings.ingestion.chunk_size == 500
        assert settings.ingestion.batch_size == 8

    def test_file_not_found_raises_settings_error(self, tmp_path: Path) -> None:
        """load_settings raises SettingsError (not FileNotFoundError) for missing file."""
        with pytest.raises(SettingsError, match="not found"):
            load_settings(str(tmp_path / "nonexistent.yaml"))

    def test_settings_error_is_value_error(self, tmp_path: Path) -> None:
        """SettingsError is a subclass of ValueError for broad compatibility."""
        with pytest.raises(ValueError):
            load_settings(str(tmp_path / "nonexistent.yaml"))


# ---------------------------------------------------------------------------
# TestValidateSettings
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestValidateSettings:
    """Tests for validate_settings() required-field enforcement."""

    def _make_settings_via_yaml(self, tmp_path: Path, overrides: str = "") -> Settings:
        """Helper: build a valid Settings by writing YAML and loading it."""
        cfg = write_yaml(tmp_path, FULL_VALID_YAML + overrides)
        return load_settings(str(cfg))

    def test_valid_settings_does_not_raise(self, tmp_path: Path) -> None:
        """validate_settings does not raise for a complete settings object."""
        settings = self._make_settings_via_yaml(tmp_path)
        validate_settings(settings)  # no exception

    def test_missing_required_section_raises(self, tmp_path: Path) -> None:
        """Missing required top-level section raises SettingsError."""
        yaml_no_retrieval = FULL_VALID_YAML.replace(
            "retrieval:", "# retrieval:"
        ).replace("  dense_top_k: 20\n", "").replace(
            "  sparse_top_k: 20\n", ""
        ).replace("  fusion_top_k: 10\n", "").replace("  rrf_k: 60\n", "")
        cfg = write_yaml(tmp_path, yaml_no_retrieval)
        with pytest.raises(SettingsError, match="retrieval"):
            load_settings(str(cfg))

    def test_missing_llm_provider_raises(self, tmp_path: Path) -> None:
        """Missing llm.provider field raises SettingsError naming the field path."""
        yaml_no_llm_provider = FULL_VALID_YAML.replace(
            '  provider: "openai"\n  model: "gpt-4o-mini"',
            '  model: "gpt-4o-mini"',
            1,  # only first occurrence (llm section)
        )
        cfg = write_yaml(tmp_path, yaml_no_llm_provider)
        with pytest.raises(SettingsError, match="llm.provider"):
            load_settings(str(cfg))

    def test_missing_embedding_provider_raises(self, tmp_path: Path) -> None:
        """Missing embedding.provider raises SettingsError naming the field path."""
        yaml_no_emb_provider = FULL_VALID_YAML.replace(
            'embedding:\n  provider: "openai"\n',
            'embedding:\n',
        )
        cfg = write_yaml(tmp_path, yaml_no_emb_provider)
        with pytest.raises(SettingsError, match="embedding.provider"):
            load_settings(str(cfg))

    def test_missing_embedding_dimensions_raises(self, tmp_path: Path) -> None:
        """Missing embedding.dimensions raises SettingsError (it is required)."""
        yaml_no_dims = FULL_VALID_YAML.replace("  dimensions: 1536\n", "")
        cfg = write_yaml(tmp_path, yaml_no_dims)
        with pytest.raises(SettingsError, match="embedding.dimensions"):
            load_settings(str(cfg))

    def test_error_message_names_missing_field(self, tmp_path: Path) -> None:
        """SettingsError message clearly identifies the missing field path."""
        yaml_no_dims = FULL_VALID_YAML.replace("  dimensions: 1536\n", "")
        cfg = write_yaml(tmp_path, yaml_no_dims)
        with pytest.raises(SettingsError) as exc_info:
            load_settings(str(cfg))
        assert "embedding.dimensions" in str(exc_info.value)
