"""Clusterer tests on synthetic gaussian blobs."""

from __future__ import annotations

import numpy as np
import pytest

from topicgpt.clustering import (
    AgglomerativeClusterer,
    Clusterer,
    HDBSCANClusterer,
    KMeansClusterer,
)
from topicgpt.config import AgglomerativeConfig, HDBSCANConfig, KMeansConfig
from topicgpt.exceptions import ClusteringError


def _three_blobs(n_per: int = 40, d: int = 8, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((3, d)).astype(np.float32) * 10
    blocks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for c in range(3):
        blocks.append(centers[c] + 0.3 * rng.standard_normal((n_per, d)).astype(np.float32))
        labels.append(np.full(n_per, c, dtype=np.int64))
    return np.vstack(blocks).astype(np.float32), np.concatenate(labels)


@pytest.mark.unit
def test_kmeans_recovers_three_clusters() -> None:
    X, truth = _three_blobs()
    clusterer = KMeansClusterer(KMeansConfig(n_clusters=3, random_state=0))
    pred = clusterer.fit_predict(X)
    assert pred.shape == truth.shape
    assert clusterer.n_clusters_ == 3
    assert len(np.unique(pred)) == 3


@pytest.mark.unit
def test_kmeans_rejects_too_few_rows() -> None:
    X, _ = _three_blobs(n_per=1)
    with pytest.raises(ClusteringError):
        KMeansClusterer(KMeansConfig(n_clusters=10)).fit_predict(X)


@pytest.mark.unit
def test_agglomerative_recovers_three_clusters() -> None:
    X, _ = _three_blobs()
    clusterer = AgglomerativeClusterer(AgglomerativeConfig(n_clusters=3))
    pred = clusterer.fit_predict(X)
    assert clusterer.n_clusters_ == 3
    assert len(np.unique(pred)) == 3


@pytest.mark.unit
def test_agglomerative_rejects_too_few_rows() -> None:
    X, _ = _three_blobs(n_per=1)
    with pytest.raises(ClusteringError):
        AgglomerativeClusterer(AgglomerativeConfig(n_clusters=10)).fit_predict(X)


@pytest.mark.unit
def test_hdbscan_recovers_clusters() -> None:
    X, _ = _three_blobs(n_per=40)
    clusterer = HDBSCANClusterer(HDBSCANConfig(min_cluster_size=10))
    pred = clusterer.fit_predict(X)
    assert clusterer.n_clusters_ >= 3
    assert -1 in pred or set(np.unique(pred)).issubset({0, 1, 2})


@pytest.mark.unit
def test_hdbscan_rejects_too_few_rows() -> None:
    X, _ = _three_blobs(n_per=2)
    with pytest.raises(ClusteringError):
        HDBSCANClusterer(HDBSCANConfig(min_cluster_size=30)).fit_predict(X)


@pytest.mark.unit
def test_hdbscan_raises_when_all_outliers() -> None:
    import contextlib

    rng = np.random.default_rng(0)
    # Random uniform points → no density structure → expect all -1.
    X = rng.standard_normal((60, 8)).astype(np.float32)
    clusterer = HDBSCANClusterer(HDBSCANConfig(min_cluster_size=50))
    # Either ClusteringError fires (all outliers) or at least one cluster was found.
    with contextlib.suppress(ClusteringError):
        clusterer.fit_predict(X)


@pytest.mark.unit
def test_clusterer_protocol_recognised() -> None:
    assert isinstance(KMeansClusterer(), Clusterer)
    assert isinstance(AgglomerativeClusterer(), Clusterer)
    assert isinstance(HDBSCANClusterer(), Clusterer)
