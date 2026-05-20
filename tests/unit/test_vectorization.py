"""Vectorization tests."""

from __future__ import annotations

import numpy as np
import pytest

from topicgpt.config import CTFIDFConfig
from topicgpt.exceptions import VectorizationError
from topicgpt.vectorization import (
    CosineSimilarityScorer,
    CTFIDFVectorizer,
    TopicVectorizer,
)

# ---------- c-TF-IDF ----------


@pytest.mark.unit
def test_ctfidf_shape_and_vocab() -> None:
    docs = [
        "cat dog mouse",
        "cat lion tiger",
        "ocean wave shore",
        "sea ocean tide",
    ]
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    scores, vocab = CTFIDFVectorizer().fit_transform(docs, labels)
    assert scores.shape[0] == 2  # two topics
    assert scores.shape[1] == len(vocab)
    assert scores.dtype == np.float32


@pytest.mark.unit
def test_ctfidf_excludes_outlier_bucket() -> None:
    docs = ["alpha bravo", "charlie delta", "echo foxtrot", "golf hotel"]
    labels = np.array([0, 1, -1, -1], dtype=np.int64)
    scores, _ = CTFIDFVectorizer().fit_transform(docs, labels)
    assert scores.shape[0] == 2  # outlier topic dropped


@pytest.mark.unit
def test_ctfidf_picks_topic_specific_words() -> None:
    docs = [
        "lion tiger zebra africa wild safari savanna",
        "lion tiger leopard cheetah hunt prey wild",
        "ocean wave shore sand beach tide blue",
        "ocean tide beach sand surf wave foam",
    ]
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    vec = CTFIDFVectorizer()
    scores, vocab = vec.fit_transform(docs, labels)
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    # "lion" should rank higher in topic 0 than topic 1
    assert scores[0, word_to_idx["lion"]] > scores[1, word_to_idx["lion"]]
    assert scores[1, word_to_idx["ocean"]] > scores[0, word_to_idx["ocean"]]


@pytest.mark.unit
def test_ctfidf_rejects_length_mismatch() -> None:
    with pytest.raises(VectorizationError, match="length mismatch"):
        CTFIDFVectorizer().fit_transform(["a"], np.array([0, 0], dtype=np.int64))


@pytest.mark.unit
def test_ctfidf_rejects_all_outliers() -> None:
    with pytest.raises(VectorizationError, match="non-outlier"):
        CTFIDFVectorizer().fit_transform(["a", "b"], np.array([-1, -1], dtype=np.int64))


@pytest.mark.unit
def test_ctfidf_bm25_variant_works() -> None:
    docs = ["alpha bravo", "charlie delta"]
    labels = np.array([0, 1], dtype=np.int64)
    scores, _ = CTFIDFVectorizer(CTFIDFConfig(bm25_weighting=True)).fit_transform(docs, labels)
    assert scores.shape[0] == 2


@pytest.mark.unit
def test_ctfidf_protocol_check() -> None:
    assert isinstance(CTFIDFVectorizer(), TopicVectorizer)


# ---------- cosine scorer ----------


@pytest.mark.unit
def test_cosine_scores_dot_product() -> None:
    centroids = np.array([[1.0, 0.0]], dtype=np.float32)
    vocab_emb = np.array(
        [
            [1.0, 0.0],  # same direction
            [0.0, 1.0],  # orthogonal
            [-1.0, 0.0],  # opposite
        ],
        dtype=np.float32,
    )
    scores, vocab = CosineSimilarityScorer().score(centroids, vocab_emb, ["a", "b", "c"])
    assert scores.shape == (1, 3)
    assert vocab == ["a", "b", "c"]
    np.testing.assert_allclose(scores[0], [1.0, 0.0, -1.0], atol=1e-6)


@pytest.mark.unit
def test_cosine_rejects_dim_mismatch() -> None:
    with pytest.raises(VectorizationError, match="dim"):
        CosineSimilarityScorer().score(
            np.zeros((1, 4), dtype=np.float32),
            np.zeros((2, 8), dtype=np.float32),
            ["a", "b"],
        )


@pytest.mark.unit
def test_cosine_rejects_vocab_size_mismatch() -> None:
    with pytest.raises(VectorizationError, match="vocab"):
        CosineSimilarityScorer().score(
            np.zeros((1, 4), dtype=np.float32),
            np.zeros((3, 4), dtype=np.float32),
            ["a", "b"],
        )


@pytest.mark.unit
def test_cosine_rejects_1d() -> None:
    with pytest.raises(VectorizationError, match="2D"):
        CosineSimilarityScorer().score(
            np.zeros(4, dtype=np.float32),
            np.zeros((2, 4), dtype=np.float32),
            ["a", "b"],
        )
