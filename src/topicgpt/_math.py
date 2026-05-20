"""Small numpy helpers shared across modules.

Centralising these kills four copies of L2-normalize and three of cosine-sim.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def l2_normalize(X: NDArray[np.float32]) -> NDArray[np.float32]:
    """Row-wise L2 normalise, safe on zero rows."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return np.asarray(X / np.maximum(norms, 1e-12), dtype=np.float32)


def cosine_similarity(
    A: NDArray[np.float32],
    B: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Row-wise cosine similarity between ``A`` (m, d) and ``B`` (n, d)."""
    return np.asarray(l2_normalize(A) @ l2_normalize(B).T, dtype=np.float32)


__all__ = ["cosine_similarity", "l2_normalize"]
