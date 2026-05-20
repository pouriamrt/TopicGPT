"""Vectorizer protocol — class-based TF-IDF and friends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray


@runtime_checkable
class TopicVectorizer(Protocol):
    """Score every vocabulary term against every cluster."""

    def fit_transform(
        self,
        documents: Sequence[str],
        labels: NDArray[np.int64],
    ) -> tuple[NDArray[np.float32], list[str]]:
        """Return ``(scores, vocab)`` where ``scores[t, v]`` ranks term ``v`` in topic ``t``."""
        ...


__all__ = ["TopicVectorizer"]
