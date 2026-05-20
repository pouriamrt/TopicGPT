"""HDBSCAN clusterer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from topicgpt.config import HDBSCANConfig
from topicgpt.exceptions import ClusteringError

if TYPE_CHECKING:
    from numpy.typing import NDArray


class HDBSCANClusterer:
    """Density-based clustering with HDBSCAN."""

    def __init__(self, config: HDBSCANConfig | None = None) -> None:
        self.config = config or HDBSCANConfig()
        self._labels: NDArray[np.int64] | None = None

    @property
    def n_clusters_(self) -> int:
        """Number of clusters discovered (excluding the outlier bucket)."""
        if self._labels is None:
            return 0
        uniq = np.unique(self._labels)
        return int(np.sum(uniq != -1))

    def fit_predict(self, X: NDArray[np.float32]) -> NDArray[np.int64]:
        """Cluster ``X`` and return labels with ``-1`` for outliers."""
        if X.ndim != 2 or X.shape[0] < self.config.min_cluster_size:
            raise ClusteringError(
                f"HDBSCAN needs >= min_cluster_size={self.config.min_cluster_size} rows, "
                f"got shape {X.shape}"
            )
        import hdbscan

        model = hdbscan.HDBSCAN(
            min_cluster_size=self.config.min_cluster_size,
            min_samples=self.config.min_samples,
            metric=self.config.metric,
            cluster_selection_method=self.config.cluster_selection_method,
            prediction_data=self.config.prediction_data,
        )
        labels = np.asarray(model.fit_predict(X), dtype=np.int64)
        if int(np.sum(labels != -1)) == 0:
            raise ClusteringError(
                "HDBSCAN assigned every point to the outlier bucket — try lowering "
                "min_cluster_size or using more data"
            )
        self._labels = labels
        return labels


__all__ = ["HDBSCANClusterer"]
