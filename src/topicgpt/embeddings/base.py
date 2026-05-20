"""Embedder protocol.

Every embedding back-end implements :class:`Embedder`. The pipeline only ever
talks to this protocol — concrete back-ends are interchangeable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray


@runtime_checkable
class Embedder(Protocol):
    """Stateless text-to-vector encoder.

    Implementations must be deterministic given a fixed model + input, and must
    produce ``float32`` arrays of shape ``(len(texts), dim)``.
    """

    @property
    def dim(self) -> int:
        """Embedding dimensionality."""
        ...

    @property
    def model_name(self) -> str:
        """Human-readable identifier of the underlying model."""
        ...

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Encode a batch of texts."""
        ...


__all__ = ["Embedder"]
