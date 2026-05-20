"""KeyBERT-style keyword representer with MMR diversity.

Given a precomputed ``(T, V)`` similarity / score matrix and an aligned vocab,
pick the top-N keywords per topic using Maximal Marginal Relevance. MMR
balances relevance to the topic centroid against novelty vs. already-picked
keywords, so the output reads less like a thesaurus.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from topicgpt._math import cosine_similarity
from topicgpt.config import KeyBERTRepresentationConfig
from topicgpt.exceptions import RepresentationError
from topicgpt.topic import Topic, make_topic

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from topicgpt.representation.base import TopicCandidate


class KeyBERTRepresenter:
    """Pick top-N keywords per topic with MMR diversity."""

    name = "keybert"

    def __init__(
        self,
        config: KeyBERTRepresentationConfig | None = None,
        *,
        scores: NDArray[np.float32] | None = None,
        vocab: Sequence[str] | None = None,
        word_embeddings: NDArray[np.float32] | None = None,
    ) -> None:
        """Build the representer.

        Args:
            config: Knobs (top_n, diversity, candidate pool).
            scores: ``(T, V)`` score matrix (e.g. c-TF-IDF output).
            vocab: vocabulary aligned to ``scores`` columns.
            word_embeddings: optional ``(V, D)`` embeddings used for MMR
                diversity scoring. If ``None``, falls back to score-only MMR.
        """
        self.config = config or KeyBERTRepresentationConfig()
        self._scores = scores
        self._vocab = list(vocab) if vocab is not None else None
        self._word_embeddings = word_embeddings

    def bind(
        self,
        *,
        scores: NDArray[np.float32] | None = None,
        vocab: Sequence[str] | None = None,
        word_embeddings: NDArray[np.float32] | None = None,
    ) -> None:
        """Attach precomputed scores / vocab / embeddings after construction.

        Public hook used by :class:`TopicModel` to wire the vectorizer output
        into a representer the caller constructed with no arguments.
        Only fields not already set on the instance are overwritten.
        """
        if self._scores is None and scores is not None:
            self._scores = scores
        if self._vocab is None and vocab is not None:
            self._vocab = list(vocab)
        if self._word_embeddings is None and word_embeddings is not None:
            self._word_embeddings = word_embeddings

    def represent(self, candidates: Sequence[TopicCandidate]) -> list[Topic]:
        """Return enriched topics with top keywords + scores."""
        if self._scores is None or self._vocab is None:
            raise RepresentationError(
                "KeyBERTRepresenter needs `scores` and `vocab` "
                "(pass them to __init__ or call .bind())."
            )

        out: list[Topic] = []
        for cand in candidates:
            row_idx = cand.topic_id
            if row_idx < 0 or row_idx >= self._scores.shape[0]:
                raise RepresentationError(
                    f"Candidate topic_id={row_idx} out of bounds for "
                    f"scores shape {self._scores.shape}"
                )
            kw_idx, kw_scores = self._pick(self._scores[row_idx])
            words = tuple(self._vocab[i] for i in kw_idx)
            out.append(
                make_topic(
                    cand.topic_id,
                    keywords=words,
                    keyword_scores=tuple(float(s) for s in kw_scores),
                    representative_docs=cand.representative_docs,
                    size=cand.size,
                )
            )
        return out

    def _pick(self, scores: NDArray[np.float32]) -> tuple[list[int], list[float]]:
        pool_size = min(self.config.candidate_pool, scores.shape[0])
        pool = np.argpartition(scores, -pool_size)[-pool_size:]
        pool = pool[np.argsort(scores[pool])[::-1]]

        if self._word_embeddings is None or self.config.diversity == 0.0:
            picks_arr = pool[: self.config.top_n_words]
            return [int(i) for i in picks_arr], [float(scores[i]) for i in picks_arr]

        sim = cosine_similarity(self._word_embeddings[pool], self._word_embeddings[pool])
        selected_local: list[int] = []
        remaining = list(range(len(pool)))
        lam = 1.0 - self.config.diversity
        while len(selected_local) < self.config.top_n_words and remaining:
            best_local = remaining[0]
            best_value = -np.inf
            for cand in remaining:
                relevance = scores[pool[cand]]
                redundancy = float(np.max(sim[cand, selected_local])) if selected_local else 0.0
                value = lam * relevance - (1 - lam) * redundancy
                if value > best_value:
                    best_value = value
                    best_local = cand
            selected_local.append(best_local)
            remaining.remove(best_local)

        picks = [int(pool[i]) for i in selected_local]
        return picks, [float(scores[i]) for i in picks]


__all__ = ["KeyBERTRepresenter"]
