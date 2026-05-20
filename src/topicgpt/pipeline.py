"""TopicModel — orchestrates the full pipeline.

Composes an :class:`Embedder`, :class:`DimReducer`, :class:`Clusterer`,
:class:`TopicVectorizer`, and a list of :class:`Representer` instances.
Output is a tuple of immutable :class:`Topic` objects.

Persistence uses ``joblib`` for sklearn-style estimators and ``json`` for the
lightweight metadata payload.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import cdist

from topicgpt._math import cosine_similarity
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
    """State recorded after ``fit`` so ``transform``/``find_topics`` work.

    We intentionally drop the full ``embeddings`` and ``reduced`` matrices
    after fit — they're only needed during candidate building. ``centroids``
    in reduced space and ``ctfidf_scores`` (T x V) are kept so transform and
    visualisation paths work without re-running the embedder.
    """

    labels: NDArray[np.int64]
    centroids: NDArray[np.float32]
    topic_ids: list[int]
    vocab: list[str] = field(default_factory=list)
    ctfidf_scores: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.float32)
    )


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

    # ------------------------------------------------------------------
    # Public API

    @property
    def topics_(self) -> tuple[Topic, ...]:
        """Topics produced by the last ``fit`` call."""
        self._get_state()
        return self._topics

    @property
    def state_(self) -> _FitState:
        """Read-only handle on the internal fit state (for viz / introspection)."""
        return self._get_state()

    def fit(self, documents: Sequence[str]) -> TopicModel:
        """Run the full pipeline on ``documents``."""
        if not documents:
            raise TopicGPTError("Cannot fit on an empty corpus")

        docs = list(documents)
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
        for r in self.representers:
            if isinstance(r, KeyBERTRepresenter):
                r.bind(scores=scores, vocab=vocab)

        candidates = self._build_candidates(docs, labels, topic_ids, scores, vocab, reduced)
        self._topics = tuple(self._apply_representers(candidates))
        # Persist only the slim view-state, not the full embedding matrices.
        self._state = _FitState(
            labels=labels,
            centroids=centroids,
            topic_ids=topic_ids,
            vocab=vocab,
            ctfidf_scores=scores,
        )
        return self

    def fit_transform(self, documents: Sequence[str]) -> NDArray[np.int64]:
        """Fit on ``documents`` and return per-document topic labels."""
        self.fit(documents)
        return self._get_state().labels

    def transform(self, documents: Sequence[str]) -> NDArray[np.int64]:
        """Assign new documents to the closest existing topic."""
        state = self._get_state()
        if state.centroids.shape[0] == 0:
            raise TopicGPTError("Model has no non-outlier topics to assign to.")
        emb = self.embedder.embed(list(documents))
        reduced = self.reducer.transform(emb)
        dists = cdist(reduced, state.centroids)
        nearest = np.argmin(dists, axis=1)
        return np.array([state.topic_ids[int(i)] for i in nearest], dtype=np.int64)

    def get_topic_info(self) -> pd.DataFrame:
        """Return a DataFrame summary of all topics."""
        self._get_state()
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
        self._get_state()
        for t in self._topics:
            if t.id == topic_id:
                return t
        raise KeyError(f"No topic with id={topic_id}")

    def find_topics(self, query: str, top_k: int = 5) -> list[tuple[Topic, float]]:
        """Return the ``top_k`` topics most similar to ``query``."""
        state = self._get_state()
        if state.centroids.shape[0] == 0:
            return []
        q_emb = self.embedder.embed([query])
        q_reduced = self.reducer.transform(q_emb)
        sims = cosine_similarity(q_reduced, state.centroids)[0]
        order = np.argsort(sims)[::-1][:top_k]
        return [(self.get_topic(state.topic_ids[int(i)]), float(sims[i])) for i in order]

    def hierarchical_topics(self) -> pd.DataFrame:
        """Hierarchical clustering of topics via Ward linkage on c-TF-IDF rows."""
        state = self._get_state()
        if state.ctfidf_scores.shape[0] < 2:
            return pd.DataFrame(columns=["parent_id", "left", "right", "distance"])
        linkage_matrix = linkage(state.ctfidf_scores, method="ward")
        n = state.ctfidf_scores.shape[0]
        df = pd.DataFrame(linkage_matrix[:, :3], columns=["left", "right", "distance"])
        df = df.astype({"left": int, "right": int, "distance": float})
        df.insert(0, "parent_id", np.arange(n, n + len(linkage_matrix)))
        return df

    def save(self, path: str | Path) -> None:
        """Persist the fitted model to ``path`` (creates the directory)."""
        state = self._get_state()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        try:
            joblib.dump(state, path / "state.joblib")
            joblib.dump(self.reducer, path / "reducer.joblib")
            joblib.dump(self.clusterer, path / "clusterer.joblib")
            joblib.dump(self._topics, path / "topics.joblib")
            (path / "meta.json").write_text(
                json.dumps({"embedder_model": self.embedder.model_name}),
                encoding="utf-8",
            )
        except OSError as e:
            raise PersistenceError(f"Failed to save model: {e}") from e

    @classmethod
    def load(cls, path: str | Path, embedder: Embedder | None = None) -> TopicModel:
        """Load a fitted model.

        ``embedder`` is only required for ``transform`` / ``find_topics``. Read-only
        ops like ``get_topic_info`` work without one.
        """
        path = Path(path)
        try:
            state: _FitState = joblib.load(path / "state.joblib")
            reducer = joblib.load(path / "reducer.joblib")
            clusterer = joblib.load(path / "clusterer.joblib")
            topics: tuple[Topic, ...] = joblib.load(path / "topics.joblib")
        except (OSError, EOFError) as e:
            raise PersistenceError(f"Failed to load model from {path}: {e}") from e
        model = cls(
            embedder=embedder if embedder is not None else _NoEmbedder(),
            reducer=reducer,
            clusterer=clusterer,
        )
        model._state = state
        model._topics = topics
        return model

    # ------------------------------------------------------------------
    # Internals

    def _get_state(self) -> _FitState:
        """Return ``self._state`` or raise. Replaces hand-rolled asserts."""
        if self._state is None:
            raise TopicGPTError("Model is not fitted yet. Call .fit(documents) first.")
        return self._state

    def _build_candidates(
        self,
        documents: Sequence[str],
        labels: NDArray[np.int64],
        topic_ids: list[int],
        scores: NDArray[np.float32],
        vocab: list[str],
        reduced: NDArray[np.float32],
    ) -> list[TopicCandidate]:
        # Single pass: collect per-topic doc indices once instead of T full scans.
        idx_by_topic: dict[int, list[int]] = {tid: [] for tid in topic_ids}
        for i, tid in enumerate(labels):
            tid_int = int(tid)
            if tid_int in idx_by_topic:
                idx_by_topic[tid_int].append(i)

        out: list[TopicCandidate] = []
        for row, tid in enumerate(topic_ids):
            doc_idx = idx_by_topic[tid]
            top_idx = _top_k_indices(scores[row], k=10)
            kws = tuple(vocab[i] for i in top_idx)
            kw_scores = tuple(float(scores[row, i]) for i in top_idx)
            rep_docs = _pick_representative(documents, reduced, doc_idx, self.n_representative_docs)
            out.append(
                TopicCandidate(
                    topic_id=tid,
                    keywords=kws,
                    keyword_scores=kw_scores,
                    representative_docs=rep_docs,
                    size=len(doc_idx),
                )
            )
        return out

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

        topics: list[Topic] = []
        current = candidates
        for r in self.representers:
            topics = r.represent(current)
            # Forward enriched fields (label/description/meta) to the next stage
            # via TopicCandidate.extra so chained representers don't lose them.
            current = [_topic_to_candidate(t) for t in topics]
        return topics


# ----------------------------------------------------------------------
# Module helpers


class _NoEmbedder:
    """Placeholder used when a model is loaded purely for read-only inspection."""

    model_name = "unset"
    dim = 0

    def embed(self, _texts: Sequence[str]) -> NDArray[np.float32]:
        """Raise — this sentinel has no model behind it."""
        raise TopicGPTError(
            "This model was loaded without an embedder. Pass embedder=... to load() "
            "before calling transform()/find_topics()."
        )


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


def _top_k_indices(scores: NDArray[np.float32], *, k: int) -> NDArray[np.int64]:
    """Return the indices of the top ``k`` scores, sorted descending."""
    k = min(k, scores.shape[0])
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    part = np.argpartition(scores, -k)[-k:]
    return part[np.argsort(scores[part])[::-1]]


def _pick_representative(
    documents: Sequence[str],
    reduced: NDArray[np.float32],
    doc_idx: list[int],
    n: int,
) -> tuple[str, ...]:
    if not doc_idx:
        return ()
    local = reduced[doc_idx]
    centroid = local.mean(axis=0, keepdims=True)
    distances = cdist(local, centroid).flatten()
    order = np.argsort(distances)[:n]
    return tuple(documents[doc_idx[int(i)]] for i in order)


def _topic_to_candidate(t: Topic) -> TopicCandidate:
    return TopicCandidate(
        topic_id=t.id,
        keywords=t.keywords,
        keyword_scores=t.keyword_scores,
        representative_docs=t.representative_docs,
        size=t.size,
        extra={"label": t.label, "description": t.description, "meta": dict(t.meta)},
    )


__all__ = ["TopicModel"]
