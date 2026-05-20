"""End-to-end pipeline test with mocked back-ends.

We swap every external dependency (OpenAI, UMAP) with deterministic fakes so
the test exercises wiring + control flow without paying for slow back-ends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from topicgpt import (
    CTFIDFVectorizer,
    KeyBERTRepresenter,
    KMeansClusterer,
    KMeansConfig,
    PCAConfig,
    PCAReducer,
    Topic,
    TopicModel,
)
from topicgpt.embeddings import Embedder
from topicgpt.exceptions import TopicGPTError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from numpy.typing import NDArray

    from topicgpt.representation import TopicCandidate


# ---------- fixtures ----------


class _StubEmbedder:
    """Deterministic embedder: maps each unique token to a fixed unit vector.

    Documents whose tokens overlap with 'cat'/'tiger' land near one cluster,
    those with 'ocean'/'wave' near another.
    """

    model_name = "stub-embedder"

    def __init__(self) -> None:
        self._dim = 4
        # token → direction
        self._dirs = {
            "cat": np.array([1, 0, 0, 0], dtype=np.float32),
            "tiger": np.array([1, 0.1, 0, 0], dtype=np.float32),
            "lion": np.array([0.9, 0.2, 0, 0], dtype=np.float32),
            "ocean": np.array([0, 0, 1, 0], dtype=np.float32),
            "wave": np.array([0, 0, 0.9, 0.1], dtype=np.float32),
            "beach": np.array([0, 0, 0.8, 0.2], dtype=np.float32),
        }

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            v = np.zeros(self._dim, dtype=np.float32)
            for tok in t.lower().split():
                v += self._dirs.get(tok, np.zeros(self._dim, dtype=np.float32))
            n = np.linalg.norm(v)
            v = v / n if n > 0 else np.array([0, 1, 0, 0], dtype=np.float32)
            out[i] = v
        return out


CATS = ["cat tiger lion", "lion tiger", "cat lion", "tiger lion cat"]
OCEAN = ["ocean wave beach", "wave ocean", "beach ocean", "wave beach ocean"]


@pytest.fixture
def documents() -> list[str]:
    return CATS + OCEAN


@pytest.fixture
def model() -> TopicModel:
    return TopicModel(
        embedder=_StubEmbedder(),
        reducer=PCAReducer(PCAConfig(n_components=2, random_state=0)),
        clusterer=KMeansClusterer(KMeansConfig(n_clusters=2, random_state=0)),
        vectorizer=CTFIDFVectorizer(),
        representers=[KeyBERTRepresenter()],
    )


# ---------- happy path ----------


@pytest.mark.integration
def test_fit_recovers_two_topics(model: TopicModel, documents: list[str]) -> None:
    model.fit(documents)
    topics = model.topics_
    assert len(topics) == 2
    # Topic IDs are 0 and 1
    assert {t.id for t in topics} == {0, 1}
    # Each topic has keywords + a size summing to total docs
    assert sum(t.size for t in topics) == len(documents)
    for t in topics:
        assert len(t.keywords) > 0


@pytest.mark.integration
def test_fit_transform_returns_labels(model: TopicModel, documents: list[str]) -> None:
    labels = model.fit_transform(documents)
    assert labels.shape == (len(documents),)
    # Two contiguous blocks should get the same label (one each).
    assert len(set(labels[:4])) == 1
    assert len(set(labels[4:])) == 1


@pytest.mark.integration
def test_get_topic_info_returns_dataframe(model: TopicModel, documents: list[str]) -> None:
    model.fit(documents)
    df = model.get_topic_info()
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"topic_id", "label", "size", "keywords"}
    assert len(df) == 2


@pytest.mark.integration
def test_get_topic_by_id(model: TopicModel, documents: list[str]) -> None:
    model.fit(documents)
    t0 = model.get_topic(0)
    assert isinstance(t0, Topic)
    assert t0.id == 0


@pytest.mark.integration
def test_get_topic_missing_raises(model: TopicModel, documents: list[str]) -> None:
    model.fit(documents)
    with pytest.raises(KeyError):
        model.get_topic(999)


@pytest.mark.integration
def test_find_topics_ranks_by_query(model: TopicModel, documents: list[str]) -> None:
    model.fit(documents)
    matches = model.find_topics("cat tiger lion", top_k=2)
    assert len(matches) == 2
    top_topic, top_score = matches[0]
    _, second_score = matches[1]
    # Top hit should win by score AND surface a cats keyword in its top words.
    assert top_score >= second_score
    cat_words = {"cat", "tiger", "lion"}
    assert any(k in cat_words for k in top_topic.keywords[:6])


@pytest.mark.integration
def test_transform_new_documents(model: TopicModel, documents: list[str]) -> None:
    model.fit(documents)
    labels = model.transform(["beach ocean", "cat tiger"])
    assert labels.shape == (2,)


@pytest.mark.integration
def test_hierarchical_topics_shape(model: TopicModel, documents: list[str]) -> None:
    model.fit(documents)
    h = model.hierarchical_topics()
    assert set(h.columns) == {"parent_id", "left", "right", "distance"}
    # n=2 topics → 1 merge.
    assert len(h) == 1


@pytest.mark.integration
def test_save_load_round_trip(model: TopicModel, documents: list[str], tmp_path: Path) -> None:
    model.fit(documents)
    out = tmp_path / "model"
    model.save(out)
    restored = TopicModel.load(out, embedder=_StubEmbedder())
    assert {t.id for t in restored.topics_} == {t.id for t in model.topics_}
    # transform still works after reload
    labels = restored.transform(["cat", "ocean"])
    assert labels.shape == (2,)


# ---------- error paths ----------


@pytest.mark.unit
def test_empty_corpus_raises() -> None:
    model = TopicModel(embedder=_StubEmbedder())
    with pytest.raises(TopicGPTError, match="empty corpus"):
        model.fit([])


@pytest.mark.unit
def test_get_topic_info_before_fit_raises() -> None:
    model = TopicModel(embedder=_StubEmbedder())
    with pytest.raises(TopicGPTError, match="not fitted"):
        model.get_topic_info()


@pytest.mark.unit
def test_transform_before_fit_raises() -> None:
    model = TopicModel(embedder=_StubEmbedder())
    with pytest.raises(TopicGPTError, match="not fitted"):
        model.transform(["x"])


@pytest.mark.unit
def test_topic_model_implements_embedder_protocol_check() -> None:
    """Smoke-test: our stub embedder satisfies the Embedder protocol."""
    assert isinstance(_StubEmbedder(), Embedder)


@pytest.mark.integration
def test_llm_representer_is_run_when_supplied(model: TopicModel, documents: list[str]) -> None:
    """A non-KeyBERT representer should also be invoked, producing labels."""
    llm = MagicMock()
    llm.name = "llm"

    def label_each(cands: list[TopicCandidate]) -> list[Topic]:
        from topicgpt.topic import make_topic

        return [
            make_topic(
                c.topic_id,
                label=f"Topic-{c.topic_id}",
                description="auto",
                keywords=c.keywords,
                keyword_scores=c.keyword_scores,
                representative_docs=c.representative_docs,
                size=c.size,
            )
            for c in cands
        ]

    llm.represent.side_effect = label_each
    model.representers.append(llm)
    model.fit(documents)
    assert all(t.label.startswith("Topic-") for t in model.topics_)
