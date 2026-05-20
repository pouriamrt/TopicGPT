"""DimReducer protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


@runtime_checkable
class DimReducer(Protocol):
    """Fit-and-transform dimensionality reducer."""

    @property
    def n_components(self) -> int:
        """Target dimensionality."""
        ...

    def fit_transform(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """Fit on ``X`` and return the reduced embedding."""
        ...

    def transform(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply the fitted reducer to fresh points."""
        ...


__all__ = ["DimReducer"]
