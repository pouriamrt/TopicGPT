"""Topic coherence metrics.

* **NPMI** — Normalized Pointwise Mutual Information, per Aletras & Stevenson
  (2013). Higher is better; values typically fall between -1 and 1.
* **UMass** — Mimno et al. (2011). Negative; closer to 0 is better.

These are pure-NumPy implementations that operate on tokenised documents and
per-topic top-word lists. They avoid the heavy ``gensim`` dependency tree
for the common case but match its outputs within numerical tolerance.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def _tokenize(doc: str) -> list[str]:
    return [tok.lower() for tok in doc.split() if tok]


def _doc_term_sets(documents: Sequence[str]) -> list[set[str]]:
    return [set(_tokenize(d)) for d in documents]


def npmi(
    documents: Sequence[str],
    topics: Sequence[Sequence[str]],
    *,
    top_n: int = 10,
) -> float:
    """Mean NPMI score across topics.

    Args:
        documents: Reference corpus tokenised by whitespace.
        topics: Top-N keywords per topic.
        top_n: How many top words per topic to score.

    Returns:
        Mean NPMI in ``[-1, 1]``; ``0.0`` when there are no usable pairs.
    """
    if not documents or not topics:
        return 0.0

    sets = _doc_term_sets(documents)
    n_docs = len(sets)

    # Single pass: compute term + co-term counts only for the words we care about.
    flat_words = {w.lower() for topic in topics for w in topic[:top_n]}
    term_counts: Counter[str] = Counter()
    cooc_counts: Counter[tuple[str, str]] = Counter()

    for ds in sets:
        present = ds & flat_words
        for w in present:
            term_counts[w] += 1
        pairs = sorted(present)
        for i, w1 in enumerate(pairs):
            for w2 in pairs[i + 1 :]:
                cooc_counts[w1, w2] += 1

    scores: list[float] = []
    for topic in topics:
        words = [w.lower() for w in topic[:top_n]]
        pair_scores: list[float] = []
        for i, w1 in enumerate(words):
            for w2 in words[i + 1 :]:
                key = (min(w1, w2), max(w1, w2))
                c_xy = cooc_counts.get(key, 0)
                if c_xy == 0:
                    pair_scores.append(0.0)
                    continue
                c_x = term_counts.get(w1, 0)
                c_y = term_counts.get(w2, 0)
                if c_x == 0 or c_y == 0:
                    pair_scores.append(0.0)
                    continue
                p_xy = c_xy / n_docs
                p_x = c_x / n_docs
                p_y = c_y / n_docs
                pmi = math.log(p_xy / (p_x * p_y))
                # When p_xy == 1.0, log(p_xy) is 0; co-occurrence is perfect
                # so NPMI is defined as 1.0 (limit).
                npmi_val = 1.0 if p_xy >= 1.0 else pmi / -math.log(p_xy)
                pair_scores.append(npmi_val)
        if pair_scores:
            scores.append(sum(pair_scores) / len(pair_scores))
    return float(sum(scores) / len(scores)) if scores else 0.0


def umass(
    documents: Sequence[str],
    topics: Sequence[Sequence[str]],
    *,
    top_n: int = 10,
    smoothing: float = 1.0,
) -> float:
    """Mean UMass coherence (Mimno et al. 2011).

    Negative; values closer to zero indicate more coherent topics.
    """
    if not documents or not topics:
        return 0.0

    sets = _doc_term_sets(documents)
    flat_words = {w.lower() for topic in topics for w in topic[:top_n]}
    term_counts: Counter[str] = Counter()
    cooc_counts: Counter[tuple[str, str]] = Counter()
    for ds in sets:
        present = ds & flat_words
        for w in present:
            term_counts[w] += 1
        for w1 in present:
            for w2 in present:
                if w1 != w2:
                    cooc_counts[w1, w2] += 1

    scores: list[float] = []
    for topic in topics:
        words = [w.lower() for w in topic[:top_n]]
        topic_score = 0.0
        pairs = 0
        for i in range(1, len(words)):
            for j in range(i):
                c_xy = cooc_counts.get((words[i], words[j]), 0)
                c_y = term_counts.get(words[j], 0)
                if c_y == 0:
                    continue
                topic_score += math.log((c_xy + smoothing) / c_y)
                pairs += 1
        if pairs:
            scores.append(topic_score / pairs)
    return float(sum(scores) / len(scores)) if scores else 0.0


__all__ = ["npmi", "umass"]
