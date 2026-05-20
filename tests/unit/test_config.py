"""Config model tests."""

from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from topicgpt.config import (
    AgglomerativeConfig,
    CTFIDFConfig,
    HDBSCANConfig,
    KeyBERTRepresentationConfig,
    KMeansConfig,
    LLMRepresentationConfig,
    OpenAIEmbeddingConfig,
    PCAConfig,
    SBERTEmbeddingConfig,
    Settings,
    UMAPConfig,
)

# ---------- Settings ----------


@pytest.mark.unit
def test_settings_reads_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    s = Settings()
    assert isinstance(s.openai_api_key, SecretStr)
    assert s.openai_api_key.get_secret_value() == "sk-test-123"


@pytest.mark.unit
def test_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    s = Settings()
    assert s.openai_api_key is None
    assert s.log_level == "INFO"
    assert s.max_retries == 4


@pytest.mark.unit
def test_settings_frozen() -> None:
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        s.max_retries = 99  # type: ignore[misc]


# ---------- per-stage configs ----------


@pytest.mark.unit
def test_openai_embedding_default_model() -> None:
    c = OpenAIEmbeddingConfig()
    assert c.model == "text-embedding-3-small"
    assert c.dimensions is None
    assert c.batch_size == 256


@pytest.mark.unit
def test_openai_embedding_dim_bounds() -> None:
    with pytest.raises(ValidationError):
        OpenAIEmbeddingConfig(dimensions=10)
    with pytest.raises(ValidationError):
        OpenAIEmbeddingConfig(dimensions=4096)


@pytest.mark.unit
def test_sbert_default_model() -> None:
    c = SBERTEmbeddingConfig()
    assert "bge" in c.model.lower()


@pytest.mark.unit
def test_umap_defaults() -> None:
    c = UMAPConfig()
    assert c.n_components == 5
    assert c.metric == "cosine"


@pytest.mark.unit
def test_pca_validates_components() -> None:
    with pytest.raises(ValidationError):
        PCAConfig(n_components=1)


@pytest.mark.unit
def test_hdbscan_defaults_and_bounds() -> None:
    c = HDBSCANConfig()
    assert c.min_cluster_size == 30
    with pytest.raises(ValidationError):
        HDBSCANConfig(min_cluster_size=1)


@pytest.mark.unit
def test_kmeans_n_init_auto_or_int() -> None:
    assert KMeansConfig().n_init == "auto"
    assert KMeansConfig(n_init=5).n_init == 5


@pytest.mark.unit
def test_agglomerative_ward_requires_euclidean() -> None:
    AgglomerativeConfig(linkage="ward", metric="euclidean")  # ok
    with pytest.raises(ValidationError, match="ward"):
        AgglomerativeConfig(linkage="ward", metric="cosine")


@pytest.mark.unit
def test_agglomerative_other_linkages_accept_any_metric() -> None:
    AgglomerativeConfig(linkage="complete", metric="cosine")
    AgglomerativeConfig(linkage="average", metric="manhattan")


@pytest.mark.unit
def test_ctfidf_defaults() -> None:
    c = CTFIDFConfig()
    assert c.reduce_frequent_words is True
    assert c.ngram_range == (1, 1)


@pytest.mark.unit
def test_keybert_diversity_bounds() -> None:
    KeyBERTRepresentationConfig(diversity=0.0)
    KeyBERTRepresentationConfig(diversity=1.0)
    with pytest.raises(ValidationError):
        KeyBERTRepresentationConfig(diversity=1.5)


@pytest.mark.unit
def test_llm_representation_defaults_to_gpt54mini() -> None:
    c = LLMRepresentationConfig()
    assert c.model == "gpt-5.4-mini"
    assert c.max_output_tokens == 512


@pytest.mark.unit
def test_llm_temperature_optional() -> None:
    assert LLMRepresentationConfig().temperature is None
    assert LLMRepresentationConfig(temperature=0.3).temperature == 0.3
    with pytest.raises(ValidationError):
        LLMRepresentationConfig(temperature=3.0)


@pytest.mark.unit
def test_configs_are_frozen() -> None:
    c = OpenAIEmbeddingConfig()
    with pytest.raises(ValidationError):
        c.batch_size = 1  # type: ignore[misc]


@pytest.mark.unit
def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        OpenAIEmbeddingConfig(unknown_field=42)  # type: ignore[call-arg]
