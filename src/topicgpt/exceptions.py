"""Domain exceptions for TopicGPT.

Every failure mode in the pipeline raises one of these. They form a small
hierarchy so callers can catch broadly (`TopicGPTError`) or narrowly
(`EmbeddingError`).
"""

from __future__ import annotations


class TopicGPTError(Exception):
    """Base class for all TopicGPT errors."""


class ConfigurationError(TopicGPTError):
    """A config object is missing a required field or holds an invalid value."""


class EmbeddingError(TopicGPTError):
    """Raised when the embedding provider returns an error or empty payload."""


class ReductionError(TopicGPTError):
    """Raised when dimensionality reduction fails (e.g. too few samples)."""


class ClusteringError(TopicGPTError):
    """Raised when clustering fails (e.g. all points marked outlier)."""


class VectorizationError(TopicGPTError):
    """Raised when c-TF-IDF or cosine vectorization fails."""


class RepresentationError(TopicGPTError):
    """Raised when topic representation (KeyBERT / LLM) fails."""


class LLMResponseError(RepresentationError):
    """Raised when the LLM returns a malformed or unparseable response."""


class PersistenceError(TopicGPTError):
    """Raised when saving or loading a fitted model fails."""


__all__ = [
    "ClusteringError",
    "ConfigurationError",
    "EmbeddingError",
    "LLMResponseError",
    "PersistenceError",
    "ReductionError",
    "RepresentationError",
    "TopicGPTError",
    "VectorizationError",
]
