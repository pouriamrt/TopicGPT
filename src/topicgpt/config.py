"""Typed configuration for the TopicGPT pipeline.

All configs are frozen Pydantic models so a fitted ``TopicModel`` carries an
immutable record of the parameters used. Environment-bound secrets and runtime
toggles live in :class:`Settings` (loaded from ``$OPENAI_API_KEY`` and the
``TOPICGPT_`` env namespace).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------- runtime settings (env-driven) ----------


class Settings(BaseSettings):
    """Process-wide settings loaded from env vars + .env."""

    model_config = SettingsConfigDict(
        env_prefix="TOPICGPT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="OPENAI_API_KEY",
        description="OpenAI API key. Required for OpenAI-backed embedders/representers.",
    )
    cache_dir: Path = Field(
        default=Path(".topicgpt_cache"),
        description="Where to persist HTTP cache + saved models.",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    request_timeout_s: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=4, ge=0, le=10)


# ---------- per-stage configs (frozen, validated) ----------


class _FrozenModel(BaseModel):
    """Base class for all stage configs."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class OpenAIEmbeddingConfig(_FrozenModel):
    """OpenAI embedding model parameters."""

    model: str = "text-embedding-3-small"
    dimensions: int | None = Field(
        default=None,
        ge=64,
        le=3072,
        description="Truncate output to N dims. None = use model default.",
    )
    batch_size: int = Field(default=256, ge=1, le=2048)
    max_tokens_per_input: int = Field(default=8191, ge=1)


class SBERTEmbeddingConfig(_FrozenModel):
    """Sentence-Transformers embedding model parameters."""

    model: str = "BAAI/bge-large-en-v1.5"
    batch_size: int = Field(default=64, ge=1, le=1024)
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    normalize_embeddings: bool = True


class UMAPConfig(_FrozenModel):
    """UMAP dimensionality-reduction parameters."""

    n_components: int = Field(default=5, ge=2, le=512)
    n_neighbors: int = Field(default=15, ge=2)
    min_dist: float = Field(default=0.0, ge=0.0, le=1.0)
    metric: Literal["cosine", "euclidean", "manhattan", "correlation"] = "cosine"
    random_state: int | None = 42
    low_memory: bool = False


class PCAConfig(_FrozenModel):
    """PCA dimensionality-reduction parameters."""

    n_components: int = Field(default=50, ge=2)
    random_state: int | None = 42
    whiten: bool = False


class HDBSCANConfig(_FrozenModel):
    """HDBSCAN clustering parameters."""

    min_cluster_size: int = Field(default=30, ge=2)
    min_samples: int | None = Field(default=None, ge=1)
    metric: Literal["euclidean", "manhattan"] = "euclidean"
    cluster_selection_method: Literal["eom", "leaf"] = "eom"
    prediction_data: bool = True


class KMeansConfig(_FrozenModel):
    """K-Means clustering parameters."""

    n_clusters: int = Field(default=10, ge=2)
    random_state: int | None = 42
    n_init: int | Literal["auto"] = "auto"


class AgglomerativeConfig(_FrozenModel):
    """Agglomerative clustering parameters."""

    n_clusters: int = Field(default=10, ge=2)
    linkage: Literal["ward", "complete", "average", "single"] = "ward"
    metric: Literal["euclidean", "cosine", "manhattan"] = "euclidean"

    @model_validator(mode="after")
    def _ward_requires_euclidean(self) -> Self:
        if self.linkage == "ward" and self.metric != "euclidean":
            raise ValueError("linkage='ward' is only valid with metric='euclidean'")
        return self


class CTFIDFConfig(_FrozenModel):
    """Class-based TF-IDF parameters (BERTopic-style)."""

    reduce_frequent_words: bool = True
    bm25_weighting: bool = False
    min_df: int = Field(default=1, ge=1)
    ngram_range: tuple[int, int] = (1, 1)


class KeyBERTRepresentationConfig(_FrozenModel):
    """KeyBERT-style top-word representation with MMR diversity."""

    top_n_words: int = Field(default=10, ge=1, le=50)
    diversity: float = Field(default=0.3, ge=0.0, le=1.0, description="MMR lambda")
    candidate_pool: int = Field(default=30, ge=1, le=200)


class LLMRepresentationConfig(_FrozenModel):
    """OpenAI chat-based topic labelling via structured outputs."""

    model: str = "gpt-5.4-mini"
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=512, ge=64, le=8192)
    n_representative_docs: int = Field(default=4, ge=0, le=20)
    top_n_words: int = Field(default=10, ge=1, le=50)
    system_prompt: str | None = None
    corpus_instruction: str = ""


__all__ = [
    "AgglomerativeConfig",
    "CTFIDFConfig",
    "HDBSCANConfig",
    "KMeansConfig",
    "KeyBERTRepresentationConfig",
    "LLMRepresentationConfig",
    "OpenAIEmbeddingConfig",
    "PCAConfig",
    "SBERTEmbeddingConfig",
    "Settings",
    "UMAPConfig",
]
