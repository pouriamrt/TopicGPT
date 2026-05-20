"""Immutable Topic dataclass.

A ``Topic`` is the unit of output. It is intentionally a frozen dataclass: a
fitted model produces a tuple of Topics, and downstream code (eval, viz, save)
should never mutate them. To "edit" a topic, build a new one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy as np
    from numpy.typing import NDArray


OUTLIER_ID = -1


@dataclass(frozen=True, slots=True)
class Topic:
    """Single topic discovered by the pipeline.

    Attributes:
        id: Topic id. ``-1`` is reserved for the outlier topic.
        label: Short human-readable label (LLM-generated when available).
        description: Longer prose description (LLM-generated when available).
        keywords: Top keywords ranked by the chosen representer.
        keyword_scores: Per-keyword score from the representer (same length as ``keywords``).
        size: Number of documents assigned to this topic.
        representative_docs: A few exemplar documents (by similarity to centroid).
        centroid: Mean embedding of the topic in the high-dim embedding space.
        meta: Free-form per-topic metadata (representer name → payload).
    """

    id: int
    label: str = ""
    description: str = ""
    keywords: tuple[str, ...] = ()
    keyword_scores: tuple[float, ...] = ()
    size: int = 0
    representative_docs: tuple[str, ...] = ()
    # Excluded from hash/eq: numpy arrays don't compare cleanly with ``==`` and
    # ``meta`` is a dict (mutable, unhashable). Topic equality is therefore based
    # on the structured fields above; the numerical payload rides along.
    centroid: NDArray[np.float32] | None = field(default=None, hash=False, compare=False)
    meta: Mapping[str, object] = field(default_factory=dict, hash=False, compare=False)

    def __post_init__(self) -> None:
        """Validate invariants after dataclass init."""
        if len(self.keywords) != len(self.keyword_scores):
            raise ValueError(
                f"keywords and keyword_scores must have equal length; "
                f"got {len(self.keywords)} vs {len(self.keyword_scores)}"
            )
        if self.size < 0:
            raise ValueError(f"size must be non-negative, got {self.size}")

    @property
    def is_outlier(self) -> bool:
        """True if this is the outlier (unassigned) bucket."""
        return self.id == OUTLIER_ID

    def display_name(self) -> str:
        """Short human-readable identifier for logs and plots."""
        if self.is_outlier:
            return "Outlier"
        head = self.label or ", ".join(self.keywords[:3])
        return f"Topic {self.id}: {head}" if head else f"Topic {self.id}"

    def __str__(self) -> str:
        """Return the display name."""
        return self.display_name()


def make_topic(
    topic_id: int,
    *,
    keywords: Sequence[str] = (),
    keyword_scores: Sequence[float] = (),
    label: str = "",
    description: str = "",
    size: int = 0,
    representative_docs: Sequence[str] = (),
    centroid: NDArray[np.float32] | None = None,
    meta: Mapping[str, object] | None = None,
) -> Topic:
    """Build a :class:`Topic` from sequence inputs (auto-tuples them)."""
    return Topic(
        id=topic_id,
        label=label,
        description=description,
        keywords=tuple(keywords),
        keyword_scores=tuple(keyword_scores),
        size=size,
        representative_docs=tuple(representative_docs),
        centroid=centroid,
        meta=dict(meta) if meta is not None else {},
    )


__all__ = ["OUTLIER_ID", "Topic", "make_topic"]
