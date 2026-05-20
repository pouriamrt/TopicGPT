"""TopicModel — orchestrates the full pipeline.

Composes an :class:`Embedder`, :class:`DimReducer`, :class:`Clusterer`,
:class:`TopicVectorizer`, and a list of :class:`Representer` instances.
Output is a tuple of immutable :class:`Topic` objects.

Persistence uses ``joblib`` for sklearn-style estimators and ``json``
for the lightweight config payload.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import cdist

from topicgpt.clustering import Clusterer, HDBSCANClusterer
from topicgpt.embeddings import Embedder, OpenAIEmbedder
from topicgpt.exceptions import PersistenceError, TopicGPTError
from topicgpt.reduction import DimReducer, UMAPReducer
from topicgpt.representation import KeyBERTRepresenter, Representer, TopicCandidate
from topicgpt.topic import OUTLIER_ID, Topic, make_topic
from topicgpt.vectorization import CTFIDFVectorizer, TopicVectorizer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

_LOG = logging.getLogger(__name__)


@dataclass(slots=True)
class _FitState:
    """State recorded after ``fit`` so ``transform``/``find_topics`` work."""

    embeddings: NDArray[np.float32]
    reduced: NDArray[np.float32]
    labels: NDArray[np.int64]
    centroids: NDArray[np.float32]  # in reduced space, ordered by topic id
    topic_ids: list[int]  # excludes outliers
    vocab: list[str] = field(default_factory=list)
    ctfidf_scores: NDArray[np.float32] | None = None


class TopicModel:
    """The pluggable BERTopic-style topic model."""

    def __init__(
        self,
        *,
        embedder: Embedder | None = None,
        reducer: DimReducer | None = None,
        clusterer: Clusterer | None = None,
        vectorizer: TopicVectorizer | None = None,
        representers: Sequence[Representer] | None = None,
        n_representative_docs: int = 4,
    ) -> None:
        self.embedder = embedder or OpenAIEmbedder()
        self.reducer = reducer or UMAPReducer()
        self.clusterer = clusterer or HDBSCANClusterer()
        self.vectorizer = vectorizer or CTFIDFVectorizer()
        self.representers: list[Representer] = list(representers or [])
        self.n_representative_docs = n_representative_docs
        self._state: _FitState | None = None
        self._topics: tuple[Topic, ...] = ()
        self._documents: list[str] = []

    # ------------------------------------------------------------------
    # Public API

    @property
    def topics_(self) -> tuple[Topic, ...]:
        """Topics produced by the last ``fit`` call."""
        self._require_fitted()
        return self._topics

    def fit(self, documents: Sequence[str]) -> TopicModel:
        """Run the full pipeline on ``documents``."""
        if not documents:
            raise TopicGPTError("Cannot fit on an empty corpus")

        docs = list(documents)
        self._documents = docs

        _LOG.info("Embedding %d documents...", len(docs))
        embeddings = self.embedder.embed(docs)

        _LOG.info("Reducing to %dd...", self.reducer.n_components)
        reduced = self.reducer.fit_transform(embeddings)

        _LOG.info("Clustering...")
        labels = self.clusterer.fit_predict(reduced)

        topic_ids = sorted(int(t) for t in np.unique(labels) if int(t) != OUTLIER_ID)
        centroids = _centroids_by_label(reduced, labels, topic_ids)

        _LOG.info("Vectorizing topics...")
        scores, vocab = self.vectorizer.fit_transform(docs, labels)
        # Inject scores+vocab into KeyBERT-style representers that need them.
        for r in self.representers:
            self._inject_vocab(r, scores, vocab)

        self._state = _FitState(
            embeddings=embeddings,
            reduced=reduced,
            labels=labels,
            centroids=centroids,
            topic_ids=topic_ids,
            vocab=vocab,
            ctfidf_scores=scores,
        )

        candidates = self._build_candidates(docs, labels, topic_ids, scores, vocab)
        topics = self._apply_representers(candidates)
        self._topics = tuple(topics)
        return self

    def fit_transform(self, documents: Sequence[str]) -> NDArray[np.int64]:
        """Fit on ``documents`` and return per-document topic labels."""
        self.fit(documents)
        assert self._state is not None
        return self._state.labels

    def transform(self, documents: Sequence[str]) -> NDArray[np.int64]:
        """Assign new documents to the closest existing topic."""
        self._require_fitted()
        assert self._state is not None
        emb = self.embedder.embed(list(documents))
        reduced = self.reducer.transform(emb)
        # Nearest centroid in reduced space.
        if self._state.centroids.shape[0] == 0:
            raise TopicGPTError("Model has no non-outlier topics to assign to.")
        dists = cdist(reduced, self._state.centroids)
        nearest = np.argmin(dists, axis=1)
        return np.array(
            [self._state.topic_ids[int(i)] for i in nearest],
            dtype=np.int64,
        )

    def get_topic_info(self) -> pd.DataFrame:
        """Return a DataFrame summary of all topics."""
        self._require_fitted()
        rows = [
            {
                "topic_id": t.id,
                "label": t.label or t.display_name(),
                "size": t.size,
                "keywords": ", ".join(t.keywords[:6]),
            }
            for t in self._topics
        ]
        return pd.DataFrame(rows)

    def get_topic(self, topic_id: int) -> Topic:
        """Return a single :class:`Topic`."""
        self._require_fitted()
        for t in self._topics:
            if t.id == topic_id:
                return t
        raise KeyError(f"No topic with id={topic_id}")

    def find_topics(self, query: str, top_k: int = 5) -> list[tuple[Topic, float]]:
        """Return the ``top_k`` topics most similar to ``query``."""
        self._require_fitted()
        assert self._state is not None
        q_emb = self.embedder.embed([query])  # (1, D)
        q_reduced = self.reducer.transform(q_emb)
        if self._state.centroids.shape[0] == 0:
            return []
        sims = _cosine(q_reduced, self._state.centroids)[0]  # (T,)
        order = np.argsort(sims)[::-1][:top_k]
        return [(self.get_topic(self._state.topic_ids[int(i)]), float(sims[i])) for i in order]

    def hierarchical_topics(self) -> pd.DataFrame:
        """Hierarchical clustering of topics via Ward linkage on c-TF-IDF rows."""
        self._require_fitted()
        assert self._state is not None
        if self._state.ctfidf_scores is None:
            raise TopicGPTError("No c-TF-IDF scores available for hierarchy")
        if self._state.ctfidf_scores.shape[0] < 2:
            return pd.DataFrame(columns=["parent_id", "left", "right", "distance"])
        Z = linkage(self._state.ctfidf_scores, method="ward")
        n = self._state.ctfidf_scores.shape[0]
        rows = []
        for i, (left, right, dist, _) in enumerate(Z):
            rows.append(
                {
                    "parent_id": n + i,
                    "left": int(left),
                    "right": int(right),
                    "distance": float(dist),
                }
            )
        return pd.DataFrame(rows)

    def save(self, path: str | Path) -> None:
        """Persist the fitted model to ``path`` (creates the directory)."""
        self._require_fitted()
        assert self._state is not None
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        try:
            joblib.dump(self._state, path / "state.joblib")
            joblib.dump(self.reducer, path / "reducer.joblib")
            joblib.dump(self.clusterer, path / "clusterer.joblib")
            joblib.dump(self._topics, path / "topics.joblib")
            (path / "documents.json").write_text(
                json.dumps(self._documents), encoding="utf-8"
            )
            (path / "meta.json").write_text(
                json.dumps({"embedder_model": self.embedder.model_name}),
                encoding="utf-8",
            )
        except OSError as e:
            raise PersistenceError(f"Failed to save model: {e}") from e

    @classmethod
    def load(cls, path: str | Path, embedder: Embedder) -> TopicModel:
        """Load a fitted model. Caller must supply the embedder back-end."""
        path = Path(path)
        try:
            state: _FitState = joblib.load(path / "state.joblib")
            reducer = joblib.load(path / "reducer.joblib")
            clusterer = joblib.load(path / "clusterer.joblib")
            topics: tuple[Topic, ...] = joblib.load(path / "topics.joblib")
            documents = json.loads((path / "documents.json").read_text(encoding="utf-8"))
        except (OSError, EOFError) as e:
            raise PersistenceError(f"Failed to load model from {path}: {e}") from e
        model = cls(embedder=embedder, reducer=reducer, clusterer=clusterer)
        model._state = state
        model._topics = topics
        model._documents = documents
        return model

    # ------------------------------------------------------------------
    # Internals

    def _require_fitted(self) -> None:
        if self._state is None:
            raise TopicGPTError("Model is not fitted yet. Call .fit(documents) first.")

    def _build_candidates(
        self,
        documents: Sequence[str],
        labels: NDArray[np.int64],
        topic_ids: list[int],
        scores: NDArray[np.float32],
        vocab: list[str],
    ) -> list[TopicCandidate]:
        out: list[TopicCandidate] = []
        for row, tid in enumerate(topic_ids):
            mask = labels == tid
            size = int(mask.sum())
            # Top keywords by raw score (representers may refine).
            top_idx = np.argsort(scores[row])[::-1][:10]
            kws = tuple(vocab[i] for i in top_idx)
            kw_scores = tuple(float(scores[row, i]) for i in top_idx)
            rep_docs = self._pick_representative_docs(documents, mask)
            out.append(
                TopicCandidate(
                    topic_id=tid,
                    keywords=kws,
                    keyword_scores=kw_scores,
                    representative_docs=rep_docs,
                    size=size,
                )
            )
        return out

    def _pick_representative_docs(
        self,
        documents: Sequence[str],
        mask: NDArray[np.bool_],
    ) -> tuple[str, ...]:
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return ()
        # Pick docs nearest to the local centroid in reduced space.
        assert self._state is not None
        local = self._state.reduced[idx]
        centroid = local.mean(axis=0, keepdims=True)
        d = cdist(local, centroid).flatten()
        order = np.argsort(d)[: self.n_representative_docs]
        return tuple(documents[int(idx[i])] for i in order)

    def _apply_representers(self, candidates: list[TopicCandidate]) -> list[Topic]:
        if not self.representers:
            return [
                make_topic(
                    c.topic_id,
                    keywords=c.keywords,
                    keyword_scores=c.keyword_scores,
                    representative_docs=c.representative_docs,
                    size=c.size,
                )
                for c in candidates
            ]
        current = candidates
        topics: list[Topic] = []
        for r in self.representers:
            topics = r.represent(current)
            # Feed enriched output into the next representer.
            current = [
                TopicCandidate(
                    topic_id=t.id,
                    keywords=t.keywords,
                    keyword_scores=t.keyword_scores,
                    representative_docs=t.representative_docs,
                    size=t.size,
                )
                for t in topics
            ]
        return topics

    @staticmethod
    def _inject_vocab(
        representer: Representer,
        scores: NDArray[np.float32],
        vocab: list[str],
    ) -> None:
        """Attach vocab+scores to KeyBERT-style representers that need them."""
        if isinstance(representer, KeyBERTRepresenter):
            # Allow callers to use the simpler `KeyBERTRepresenter()` ctor
            # and have the pipeline supply scores/vocab automatically.
            if representer._scores is None:
                representer._scores = scores
            if representer._vocab is None:
                representer._vocab = vocab
        # Other representers ignore this hook.
        _ = scores, vocab


# ----------------------------------------------------------------------
# Helpers


def _centroids_by_label(
    X: NDArray[np.float32],
    labels: NDArray[np.int64],
    topic_ids: list[int],
) -> NDArray[np.float32]:
    """Return one centroid per id in ``topic_ids`` (outliers excluded)."""
    if not topic_ids:
        return np.zeros((0, X.shape[1]), dtype=np.float32)
    centroids = np.vstack([X[labels == tid].mean(axis=0) for tid in topic_ids])
    return np.asarray(centroids, dtype=np.float32)


def _cosine(A: NDArray[np.float32], B: NDArray[np.float32]) -> NDArray[np.float32]:
    """Row-wise cosine similarity between matrices A (m,d) and B (n,d)."""
    a_norm = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    b_norm = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    return np.asarray(a_norm @ b_norm.T, dtype=np.float32)


__all__ = ["TopicModel"]


# Silence "unused noqa" if mypy strips comments
_ = Any
