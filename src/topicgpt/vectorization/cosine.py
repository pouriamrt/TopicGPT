"""Cosine-similarity scorer for vocab vs. cluster centroids.

Given precomputed vocab embeddings (e.g. from :class:`OpenAIEmbedder`) and the
per-topic centroid embeddings, score each ``(topic, word)`` pair by cosine
similarity. Produces the same ``(T, V)`` shape as c-TF-IDF so callers can mix
the two methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from topicgpt._math import cosine_similarity
from topicgpt.exceptions import VectorizationError

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray


class CosineSimilarityScorer:
    """Cosine similarity between cluster centroids and vocab embeddings."""

    def score(
        self,
        centroids: NDArray[np.float32],
        vocab_embeddings: NDArray[np.float32],
        vocab: Sequence[str],
    ) -> tuple[NDArray[np.float32], list[str]]:
        """Return ``(scores, vocab_list)`` of shape ``(T, V)``."""
        if centroids.ndim != 2 or vocab_embeddings.ndim != 2:
            raise VectorizationError("centroids and vocab_embeddings must both be 2D")
        if centroids.shape[1] != vocab_embeddings.shape[1]:
            raise VectorizationError(
                f"Centroid dim {centroids.shape[1]} != vocab dim {vocab_embeddings.shape[1]}"
            )
        if vocab_embeddings.shape[0] != len(vocab):
            raise VectorizationError(
                f"vocab_embeddings rows {vocab_embeddings.shape[0]} != vocab len {len(vocab)}"
            )

        return cosine_similarity(centroids, vocab_embeddings), list(vocab)


__all__ = ["CosineSimilarityScorer"]
