"""TopicGPT: modular LLM-driven topic modelling.

Pipeline stages:
    documents → embed → reduce → cluster → vectorize → represent → label

The public surface grows phase by phase per PLAN.md. Phase 2 exports the
configuration models, the immutable :class:`Topic`, and the exception
hierarchy.
"""

from __future__ import annotations

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
from topicgpt.embeddings import Embedder, OpenAIEmbedder, SBERTEmbedder
from topicgpt.exceptions import (
    ClusteringError,
    ConfigurationError,
    EmbeddingError,
    LLMResponseError,
    PersistenceError,
    ReductionError,
    RepresentationError,
    TopicGPTError,
    VectorizationError,
)
from topicgpt.topic import OUTLIER_ID, Topic, make_topic

__version__ = "1.0.0a0"

__all__ = [
    "OUTLIER_ID",
    "AgglomerativeConfig",
    "CTFIDFConfig",
    "ClusteringError",
    "ConfigurationError",
    "Embedder",
    "EmbeddingError",
    "HDBSCANConfig",
    "KMeansConfig",
    "KeyBERTRepresentationConfig",
    "LLMRepresentationConfig",
    "LLMResponseError",
    "OpenAIEmbedder",
    "OpenAIEmbeddingConfig",
    "PCAConfig",
    "PersistenceError",
    "ReductionError",
    "RepresentationError",
    "SBERTEmbedder",
    "SBERTEmbeddingConfig",
    "Settings",
    "Topic",
    "TopicGPTError",
    "UMAPConfig",
    "VectorizationError",
    "__version__",
    "make_topic",
]
