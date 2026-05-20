"""TopicGPT: modular LLM-driven topic modelling.

Pipeline stages:
    documents → embed → reduce → cluster → vectorize → represent → label

The public surface grows phase by phase per PLAN.md. Phase 2 exports the
configuration models, the immutable :class:`Topic`, and the exception
hierarchy.
"""

from __future__ import annotations

from topicgpt import evaluation
from topicgpt.clustering import (
    AgglomerativeClusterer,
    Clusterer,
    HDBSCANClusterer,
    KMeansClusterer,
)
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
from topicgpt.pipeline import TopicModel
from topicgpt.reduction import DimReducer, PCAReducer, UMAPReducer
from topicgpt.representation import (
    KeyBERTRepresenter,
    LLMRepresenter,
    Representer,
    TopicCandidate,
    TopicLabel,
)
from topicgpt.topic import OUTLIER_ID, Topic, make_topic
from topicgpt.vectorization import CosineSimilarityScorer, CTFIDFVectorizer, TopicVectorizer

__version__ = "1.0.0a0"

__all__ = [
    "OUTLIER_ID",
    "AgglomerativeClusterer",
    "AgglomerativeConfig",
    "CTFIDFConfig",
    "CTFIDFVectorizer",
    "Clusterer",
    "ClusteringError",
    "ConfigurationError",
    "CosineSimilarityScorer",
    "DimReducer",
    "Embedder",
    "EmbeddingError",
    "HDBSCANClusterer",
    "HDBSCANConfig",
    "KMeansClusterer",
    "KMeansConfig",
    "KeyBERTRepresentationConfig",
    "KeyBERTRepresenter",
    "LLMRepresentationConfig",
    "LLMRepresenter",
    "LLMResponseError",
    "OpenAIEmbedder",
    "OpenAIEmbeddingConfig",
    "PCAConfig",
    "PCAReducer",
    "PersistenceError",
    "ReductionError",
    "RepresentationError",
    "Representer",
    "SBERTEmbedder",
    "SBERTEmbeddingConfig",
    "Settings",
    "Topic",
    "TopicCandidate",
    "TopicGPTError",
    "TopicLabel",
    "TopicModel",
    "TopicVectorizer",
    "UMAPConfig",
    "UMAPReducer",
    "VectorizationError",
    "__version__",
    "evaluation",
    "make_topic",
]
