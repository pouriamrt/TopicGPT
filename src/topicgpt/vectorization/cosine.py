"""Cosine-similarity scorer for vocab vs. cluster centroids.

Given precomputed vocab embeddings (e.g. from :class:`OpenAIEmbedder`) and the
per-topic centroid embeddings, score each ``(topic, word)`` pair by cosine
similarity. Produces the same ``(T, V)`` shape as c-TF-IDF so callers can mix
the two methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from topicgpt.exceptions import VectorizationError

if TYPE_CHECKING:
    from collections.abc import Sequence

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

        c_norm = _l2_normalize(centroids)
        v_norm = _l2_normalize(vocab_embeddings)
        scores = c_norm @ v_norm.T  # (T, V)
        return scores.astype(np.float32, copy=False), list(vocab)


def _l2_normalize(X: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return np.asarray(X / np.maximum(norms, 1e-12), dtype=np.float32)


__all__ = ["CosineSimilarityScorer"]
