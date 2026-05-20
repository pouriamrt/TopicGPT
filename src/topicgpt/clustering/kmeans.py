"""K-Means clusterer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.cluster import KMeans

from topicgpt.config import KMeansConfig
from topicgpt.exceptions import ClusteringError

if TYPE_CHECKING:
    from numpy.typing import NDArray


class KMeansClusterer:
    """Fixed-k K-Means clustering."""

    def __init__(self, config: KMeansConfig | None = None) -> None:
        self.config = config or KMeansConfig()
        self._labels: NDArray[np.int64] | None = None

    @property
    def n_clusters_(self) -> int:
        """Configured cluster count."""
        return self.config.n_clusters

    def fit_predict(self, X: NDArray[np.float32]) -> NDArray[np.int64]:
        """Cluster ``X`` into exactly ``n_clusters`` groups."""
        if X.shape[0] < self.config.n_clusters:
            raise ClusteringError(
                f"K-Means needs >= n_clusters={self.config.n_clusters} rows, got {X.shape[0]}"
            )
        model = KMeans(
            n_clusters=self.config.n_clusters,
            n_init=self.config.n_init,
            random_state=self.config.random_state,
        )
        labels = np.asarray(model.fit_predict(X), dtype=np.int64)
        self._labels = labels
        return labels


__all__ = ["KMeansClusterer"]
