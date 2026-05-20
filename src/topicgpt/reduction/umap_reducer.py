"""UMAP dimensionality reducer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from topicgpt.config import UMAPConfig
from topicgpt.exceptions import ReductionError

if TYPE_CHECKING:
    from numpy.typing import NDArray


class UMAPReducer:
    """UMAP-based dimensionality reduction.

    Defaults match the topic-modelling recipe (5 dims, 15 neighbours, cosine).
    Lazy import keeps ``umap-learn`` out of the hot import path.
    """

    def __init__(self, config: UMAPConfig | None = None) -> None:
        self.config = config or UMAPConfig()
        # umap.UMAP has no upstream type stubs; treat as Any to keep call sites typed.
        self._model: Any = None

    @property
    def n_components(self) -> int:
        """Target dimensionality."""
        return self.config.n_components

    def fit_transform(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """Fit and reduce ``X``."""
        if X.ndim != 2:
            raise ReductionError(f"Expected 2D embeddings, got shape {X.shape}")
        if X.shape[0] < max(self.config.n_neighbors, self.config.n_components + 1):
            raise ReductionError(
                f"Too few rows ({X.shape[0]}) for UMAP with "
                f"n_neighbors={self.config.n_neighbors}"
            )
        import umap

        self._model = umap.UMAP(
            n_components=self.config.n_components,
            n_neighbors=self.config.n_neighbors,
            min_dist=self.config.min_dist,
            metric=self.config.metric,
            random_state=self.config.random_state,
            low_memory=self.config.low_memory,
            verbose=False,
        )
        return np.asarray(self._model.fit_transform(X), dtype=np.float32)

    def transform(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply the fitted UMAP to new points."""
        if self._model is None:
            raise ReductionError("UMAPReducer.transform called before fit_transform")
        return np.asarray(self._model.transform(X), dtype=np.float32)


__all__ = ["UMAPReducer"]
