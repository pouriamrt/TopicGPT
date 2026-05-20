"""PCA dimensionality reducer (cheap baseline)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.decomposition import PCA

from topicgpt.config import PCAConfig
from topicgpt.exceptions import ReductionError

if TYPE_CHECKING:
    from numpy.typing import NDArray


class PCAReducer:
    """sklearn PCA wrapped to the :class:`DimReducer` protocol."""

    def __init__(self, config: PCAConfig | None = None) -> None:
        self.config = config or PCAConfig()
        self._model: PCA | None = None

    @property
    def n_components(self) -> int:
        """Target dimensionality."""
        return self.config.n_components

    def fit_transform(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """Fit PCA on ``X`` and return the projected matrix."""
        if X.ndim != 2:
            raise ReductionError(f"Expected 2D embeddings, got shape {X.shape}")
        if X.shape[0] < self.config.n_components:
            raise ReductionError(
                f"PCA needs at least n_components={self.config.n_components} rows; got {X.shape[0]}"
            )
        self._model = PCA(
            n_components=self.config.n_components,
            random_state=self.config.random_state,
            whiten=self.config.whiten,
        )
        return np.asarray(self._model.fit_transform(X), dtype=np.float32)

    def transform(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """Project new points using the fitted PCA."""
        if self._model is None:
            raise ReductionError("PCAReducer.transform called before fit_transform")
        return np.asarray(self._model.transform(X), dtype=np.float32)


__all__ = ["PCAReducer"]
