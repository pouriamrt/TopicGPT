"""Visualization smoke tests.

We don't validate pixel output — we just check that each function returns a
plotly Figure with sensible structure (right number of traces, correct title)
on a fitted model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
import pytest

from topicgpt import (
    CTFIDFVectorizer,
    KeyBERTRepresenter,
    KMeansClusterer,
    KMeansConfig,
    PCAConfig,
    PCAReducer,
    TopicModel,
)
from topicgpt.exceptions import TopicGPTError
from topicgpt.visualize import (
    visualize_barchart,
    visualize_heatmap,
    visualize_hierarchy,
    visualize_topics,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


class _StubEmbedder:
    model_name = "stub"

    def __init__(self) -> None:
        self._dim = 4

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            if "cat" in t or "tiger" in t or "lion" in t:
                out[i] = [1.0, 0.0, 0.0, 0.0]
            else:
                out[i] = [0.0, 0.0, 1.0, 0.0]
        return out


DOCS = (
    "cat tiger lion",
    "lion tiger",
    "cat lion",
    "tiger lion cat",
    "ocean wave beach",
    "wave ocean",
    "beach ocean",
    "wave beach ocean",
)


@pytest.fixture
def fitted_model() -> TopicModel:
    m = TopicModel(
        embedder=_StubEmbedder(),
        reducer=PCAReducer(PCAConfig(n_components=2, random_state=0)),
        clusterer=KMeansClusterer(KMeansConfig(n_clusters=2, random_state=0)),
        vectorizer=CTFIDFVectorizer(),
        representers=[KeyBERTRepresenter()],
    )
    m.fit(list(DOCS))
    return m


@pytest.mark.unit
def test_visualize_topics_returns_figure(fitted_model: TopicModel) -> None:
    fig = visualize_topics(fitted_model)
    assert isinstance(fig, go.Figure)
    assert "centroids" in (fig.layout.title.text or "")


@pytest.mark.unit
def test_visualize_barchart_has_one_trace_per_topic(fitted_model: TopicModel) -> None:
    fig = visualize_barchart(fitted_model)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == len(fitted_model.topics_)


@pytest.mark.unit
def test_visualize_hierarchy_returns_figure(fitted_model: TopicModel) -> None:
    fig = visualize_hierarchy(fitted_model)
    assert isinstance(fig, go.Figure)
    # 2 topics -> 1 U-link from scipy.dendrogram -> 1 Scatter trace.
    assert len(fig.data) >= 1


@pytest.mark.unit
def test_visualize_heatmap_square_shape(fitted_model: TopicModel) -> None:
    fig = visualize_heatmap(fitted_model)
    z = np.asarray(fig.data[0].z)
    n = len(fitted_model.topics_)
    assert z.shape == (n, n)
    # diagonal is 1
    np.testing.assert_allclose(np.diag(z), np.ones(n), atol=1e-5)


@pytest.mark.unit
def test_visualize_requires_fit() -> None:
    unfit = TopicModel(
        embedder=_StubEmbedder(),
        reducer=PCAReducer(PCAConfig(n_components=2)),
        clusterer=KMeansClusterer(KMeansConfig(n_clusters=2)),
    )
    with pytest.raises(TopicGPTError, match="not fitted"):
        visualize_topics(unfit)
    with pytest.raises(TopicGPTError):
        visualize_hierarchy(unfit)
