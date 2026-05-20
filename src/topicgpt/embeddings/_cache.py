"""Tiny content-addressed disk cache for embedding vectors.

One file per text, sharded into two-char subdirectories. Keyed by
``sha256(model | dimensions | text)`` so model bumps invalidate cleanly.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class EmbeddingCache:
    """Content-addressed cache for embedding vectors.

    Files are stored as raw ``float32`` ``.npy`` arrays so re-loads are zero-copy.
    """

    def __init__(self, root: Path, *, model: str, dim: int | None) -> None:
        self.root = Path(root)
        self._namespace = f"{model}::{dim if dim is not None else 'default'}"

    def _key(self, text: str) -> str:
        h = hashlib.sha256(f"{self._namespace}\0{text}".encode()).hexdigest()
        return h

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.npy"

    def get(self, text: str) -> NDArray[np.float32] | None:
        """Return cached vector or ``None`` on miss."""
        path = self._path(self._key(text))
        if not path.exists():
            return None
        try:
            arr = np.load(path)
        except (OSError, ValueError):
            return None
        return np.asarray(arr, dtype=np.float32)

    def put(self, text: str, vec: NDArray[np.float32]) -> None:
        """Persist a vector to the cache."""
        path = self._path(self._key(text))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vec.astype(np.float32, copy=False))


__all__ = ["EmbeddingCache"]
