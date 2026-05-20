"""Topic diversity metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def proportion_unique_words(topics: Sequence[Sequence[str]], *, top_n: int = 10) -> float:
    """Fraction of unique words across the top-N words of each topic.

    1.0 means every keyword is unique across topics; 0.0 means total overlap.
    """
    if not topics:
        return 0.0
    total = 0
    seen: set[str] = set()
    for t in topics:
        for w in t[:top_n]:
            total += 1
            seen.add(w.lower())
    return len(seen) / total if total else 0.0


def inverted_rbo(topics: Sequence[Sequence[str]], *, p: float = 0.9, top_n: int = 10) -> float:
    """Inverted Rank-Biased Overlap diversity (1 - mean RBO).

    Higher is better. ``p`` controls weight on the top of the list (typical
    0.9). Returns ``1.0`` when there is a single topic.
    """
    if len(topics) < 2:
        return 1.0

    truncated = [tuple(t[:top_n]) for t in topics]
    rbos: list[float] = [
        _rbo(truncated[i], truncated[j], p=p)
        for i in range(len(truncated))
        for j in range(i + 1, len(truncated))
    ]
    return 1.0 - (sum(rbos) / len(rbos)) if rbos else 1.0


def _rbo(a: tuple[str, ...], b: tuple[str, ...], *, p: float) -> float:
    """Rank-Biased Overlap (Webber, Moffat, Zobel 2010)."""
    if not a or not b:
        return 0.0
    depth = max(len(a), len(b))
    seen_a: set[str] = set()
    seen_b: set[str] = set()
    overlap = 0.0
    total = 0.0
    for d in range(depth):
        if d < len(a):
            seen_a.add(a[d])
        if d < len(b):
            seen_b.add(b[d])
        agreement = len(seen_a & seen_b) / (d + 1)
        weight = (1 - p) * (p**d)
        overlap += weight * agreement
        total += weight
    return overlap / total if total else 0.0


__all__ = ["inverted_rbo", "proportion_unique_words"]
