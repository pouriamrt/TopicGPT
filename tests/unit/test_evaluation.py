"""Evaluation metric tests."""

from __future__ import annotations

import math

import pytest

from topicgpt.evaluation import (
    inverted_rbo,
    npmi,
    proportion_unique_words,
    umass,
)

# ---------- NPMI ----------


@pytest.mark.unit
def test_npmi_perfect_cooccurrence() -> None:
    docs = ["lion tiger", "lion tiger", "lion tiger", "lion tiger"]
    topics = [["lion", "tiger"]]
    score = npmi(docs, topics)
    # lion and tiger always co-occur → NPMI -> 1
    assert score == pytest.approx(1.0, abs=1e-6)


@pytest.mark.unit
def test_npmi_independent_words() -> None:
    docs = ["lion ocean", "tiger wave", "lion wave", "tiger ocean"]
    topics = [["lion", "ocean"]]
    score = npmi(docs, topics)
    # joint = 1/4, marginals = 1/2 → PMI = log(1) = 0 → NPMI = 0
    assert score == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
def test_npmi_zero_cooccurrence_returns_zero() -> None:
    docs = ["lion", "tiger", "lion", "tiger"]
    topics = [["lion", "tiger"]]
    score = npmi(docs, topics)
    assert score == 0.0


@pytest.mark.unit
def test_npmi_empty_inputs() -> None:
    assert npmi([], [["a", "b"]]) == 0.0
    assert npmi(["doc"], []) == 0.0


@pytest.mark.unit
def test_npmi_multiple_topics_averages() -> None:
    docs = ["lion tiger", "lion tiger", "ocean wave", "ocean wave"]
    topics = [["lion", "tiger"], ["ocean", "wave"]]
    score = npmi(docs, topics)
    # both topics have perfect co-occurrence → mean = 1
    assert score == pytest.approx(1.0, abs=1e-6)


# ---------- UMass ----------


@pytest.mark.unit
def test_umass_returns_negative_for_independent() -> None:
    # alpha and bravo appear separately, but never together.
    docs = ["alpha gamma", "bravo gamma", "alpha gamma", "bravo gamma"]
    topics = [["alpha", "bravo"]]
    score = umass(docs, topics)
    # No co-occurrence ⇒ c_xy = 0 ⇒ log((0+1)/c_y) is negative.
    assert score < 0


@pytest.mark.unit
def test_umass_handles_empty_inputs() -> None:
    assert umass([], [["x"]]) == 0.0
    assert umass(["d"], []) == 0.0


# ---------- diversity ----------


@pytest.mark.unit
def test_proportion_unique_words_all_unique() -> None:
    topics = [["a", "b"], ["c", "d"]]
    assert proportion_unique_words(topics) == 1.0


@pytest.mark.unit
def test_proportion_unique_words_total_overlap() -> None:
    topics = [["a", "b"], ["a", "b"]]
    assert proportion_unique_words(topics) == pytest.approx(0.5)


@pytest.mark.unit
def test_proportion_unique_words_empty() -> None:
    assert proportion_unique_words([]) == 0.0


@pytest.mark.unit
def test_inverted_rbo_identical_topics() -> None:
    topics = [["a", "b", "c"], ["a", "b", "c"]]
    # Identical → RBO = 1 → inverted = 0
    assert inverted_rbo(topics) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
def test_inverted_rbo_disjoint_topics() -> None:
    topics = [["a", "b", "c"], ["x", "y", "z"]]
    # disjoint → RBO = 0 → inverted = 1
    assert inverted_rbo(topics) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.unit
def test_inverted_rbo_single_topic_is_one() -> None:
    assert inverted_rbo([["a", "b"]]) == 1.0


# ---------- bench harness ----------


@pytest.mark.unit
def test_run_bench_aggregates_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    from topicgpt.evaluation import bench

    # Fake model whose topics depend on seed so we get non-zero stdev.
    class _Topic:
        def __init__(self, words: list[str]) -> None:
            self.keywords = tuple(words)

    class _Model:
        def __init__(self, seed: int) -> None:
            self.seed = seed

        def fit(self, _docs: list[str]) -> None:
            return None

        @property
        def topics_(self) -> list[_Topic]:
            if self.seed == 0:
                return [_Topic(["lion", "tiger"]), _Topic(["ocean", "wave"])]
            return [_Topic(["lion", "tiger"]), _Topic(["sand", "beach"])]

    result = bench.run_bench(
        _Model,
        ["lion tiger", "ocean wave"],
        seeds=(0, 1),
        metrics=("npmi", "diversity", "rbo"),
    )
    assert {"npmi", "diversity", "rbo"} <= result.metrics.keys()
    # std should be defined for each metric
    for k in result.metrics:
        assert k in result.stds
    assert len(result.per_run) == 2
    # __str__ should be one-line summary
    assert "npmi=" in str(result)
    assert "±" in str(result)
    assert not math.isnan(result.metrics["diversity"])


@pytest.mark.unit
def test_run_bench_rejects_empty_seeds() -> None:
    from topicgpt.evaluation import bench

    with pytest.raises(ValueError, match="seed"):
        bench.run_bench(lambda _s: object(), ["x"], seeds=())  # type: ignore[arg-type]
