"""Class-based TF-IDF (c-TF-IDF), BERTopic-style.

Each cluster's documents are concatenated into a single "class document";
TF-IDF is then computed over the class-corpus. This favours words that are
*specific* to a topic over words frequent across the whole corpus.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from topicgpt.config import CTFIDFConfig
from topicgpt.exceptions import VectorizationError
from topicgpt.topic import OUTLIER_ID

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


class CTFIDFVectorizer:
    """Compute (topic x vocab) class-based TF-IDF scores."""

    def __init__(self, config: CTFIDFConfig | None = None) -> None:
        self.config = config or CTFIDFConfig()

    def fit_transform(
        self,
        documents: Sequence[str],
        labels: NDArray[np.int64],
    ) -> tuple[NDArray[np.float32], list[str]]:
        """Return ``(scores, vocab)``.

        ``scores`` has shape ``(n_topics, V)``; topics are ordered by ascending
        label id (the outlier bucket -1, if present, is excluded).
        """
        if len(documents) != labels.shape[0]:
            raise VectorizationError(
                f"documents/labels length mismatch: {len(documents)} vs {labels.shape[0]}"
            )

        topic_ids = sorted(int(t) for t in np.unique(labels) if int(t) != OUTLIER_ID)
        if not topic_ids:
            raise VectorizationError("No non-outlier topics found")

        # Single pass over the corpus instead of T full scans.
        buckets: dict[int, list[str]] = defaultdict(list)
        for d, t in zip(documents, labels, strict=True):
            tid = int(t)
            if tid != OUTLIER_ID:
                buckets[tid].append(d)
        class_docs = [" ".join(buckets[tid]) for tid in topic_ids]

        # 2. count terms in class corpus
        cv = CountVectorizer(
            ngram_range=self.config.ngram_range,
            min_df=self.config.min_df,
            lowercase=True,
        )
        try:
            counts = cv.fit_transform(class_docs).toarray().astype(np.float64)
        except ValueError as e:
            raise VectorizationError(f"CountVectorizer failed: {e}") from e
        vocab: list[str] = list(cv.get_feature_names_out())

        tf = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
        avg = float(counts.sum() / counts.shape[0]) if counts.shape[0] else 1.0
        word_freq = counts.sum(axis=0)  # term freq across all classes
        if self.config.bm25_weighting:
            # BERTopic's BM25-style stabilisation
            idf = np.log((1.0 + avg) / (1.0 + np.maximum(word_freq, 1.0))) + 1.0
        else:
            idf = np.log(1.0 + avg / np.maximum(word_freq, 1.0))

        scores = tf * idf  # broadcast (T, V) * (V,) → (T, V)

        if self.config.reduce_frequent_words:
            # down-weight terms appearing in many classes
            class_doc_freq = (counts > 0).sum(axis=0)
            penalty = 1.0 / np.maximum(class_doc_freq, 1.0)
            scores = scores * penalty

        return scores.astype(np.float32, copy=False), vocab


__all__ = ["CTFIDFVectorizer"]
