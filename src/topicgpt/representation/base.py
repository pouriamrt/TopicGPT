"""Representer protocol + the candidate DTO it consumes.

A ``Representer`` takes raw cluster information and returns enriched fields
on each topic. Multiple representers can be stacked: c-TF-IDF picks the top
keywords, then the LLM representer turns those keywords into a label and
description.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from topicgpt.topic import Topic


@dataclass(frozen=True)
class TopicCandidate:
    """Inputs needed to label / describe a single topic."""

    topic_id: int
    keywords: tuple[str, ...] = ()
    keyword_scores: tuple[float, ...] = ()
    representative_docs: tuple[str, ...] = ()
    size: int = 0
    extra: dict[str, object] = field(default_factory=dict)


@runtime_checkable
class Representer(Protocol):
    """Enrich a batch of candidate topics."""

    name: str

    def represent(self, candidates: Sequence[TopicCandidate]) -> list[Topic]:
        """Return a fresh list of :class:`Topic` objects."""
        ...


__all__ = ["Representer", "TopicCandidate"]
