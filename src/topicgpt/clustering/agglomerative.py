"""Agglomerative clusterer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from topicgpt.config import AgglomerativeConfig
from topicgpt.exceptions import ClusteringError

if TYPE_CHECKING:
    from numpy.typing import NDArray


class AgglomerativeClusterer:
    """Fixed-k Agglomerative clustering."""

    def __init__(self, config: AgglomerativeConfig | None = None) -> None:
        self.config = config or AgglomerativeConfig()
        self._labels: NDArray[np.int64] | None = None

    @property
    def n_clusters_(self) -> int:
        """Configured cluster count."""
        return self.config.n_clusters

    def fit_predict(self, X: NDArray[np.float32]) -> NDArray[np.int64]:
        """Cluster ``X`` into exactly ``n_clusters`` groups."""
        if X.shape[0] < self.config.n_clusters:
            raise ClusteringError(
                f"Agglomerative needs >= n_clusters={self.config.n_clusters} rows, "
                f"got {X.shape[0]}"
            )
        model = AgglomerativeClustering(
            n_clusters=self.config.n_clusters,
            linkage=self.config.linkage,
            metric=self.config.metric,
        )
        labels = np.asarray(model.fit_predict(X), dtype=np.int64)
        self._labels = labels
        return labels


__all__ = ["AgglomerativeClusterer"]
