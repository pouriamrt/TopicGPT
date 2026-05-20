"""Clusterer protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


@runtime_checkable
class Clusterer(Protocol):
    """Cluster vectors into integer labels.

    Labels are ``int`` arrays; ``-1`` is reserved for the outlier bucket
    (only HDBSCAN actually emits outliers).
    """

    @property
    def n_clusters_(self) -> int:
        """Number of clusters found (excluding the outlier bucket)."""
        ...

    def fit_predict(self, X: NDArray[np.float32]) -> NDArray[np.int64]:
        """Cluster ``X`` and return per-row labels."""
        ...
