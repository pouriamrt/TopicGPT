"""Tests for the frozen Topic dataclass."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from topicgpt.topic import OUTLIER_ID, Topic, make_topic


@pytest.mark.unit
def test_topic_is_frozen() -> None:
    t = make_topic(1, keywords=("a",), keyword_scores=(0.5,))
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.id = 2  # type: ignore[misc]


@pytest.mark.unit
def test_make_topic_defaults() -> None:
    t = make_topic(0)
    assert t.id == 0
    assert t.keywords == ()
    assert t.keyword_scores == ()
    assert t.size == 0
    assert t.label == ""
    assert t.meta == {}
    assert not t.is_outlier


@pytest.mark.unit
def test_outlier_property() -> None:
    t = make_topic(OUTLIER_ID)
    assert t.is_outlier
    assert t.display_name() == "Outlier"


@pytest.mark.unit
def test_display_name_with_label() -> None:
    t = make_topic(3, label="Climate change")
    assert t.display_name() == "Topic 3: Climate change"


@pytest.mark.unit
def test_display_name_falls_back_to_keywords() -> None:
    t = make_topic(
        2,
        keywords=("ocean", "warming", "polar", "sea-level"),
        keyword_scores=(0.9, 0.8, 0.7, 0.6),
    )
    assert "ocean, warming, polar" in t.display_name()


@pytest.mark.unit
def test_display_name_bare() -> None:
    t = make_topic(7)
    assert t.display_name() == "Topic 7"


@pytest.mark.unit
def test_keyword_score_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="equal length"):
        Topic(id=0, keywords=("a", "b"), keyword_scores=(0.5,))


@pytest.mark.unit
def test_negative_size_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        Topic(id=0, size=-1)


@pytest.mark.unit
def test_topic_hashable_when_centroid_none() -> None:
    t1 = make_topic(1, keywords=("a",), keyword_scores=(0.5,))
    t2 = make_topic(1, keywords=("a",), keyword_scores=(0.5,))
    assert t1 == t2
    assert hash(t1) == hash(t2)


@pytest.mark.unit
def test_centroid_carried_through() -> None:
    c = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    t = make_topic(1, centroid=c)
    assert t.centroid is not None
    np.testing.assert_array_equal(t.centroid, c)


@pytest.mark.unit
@given(
    topic_id=st.integers(min_value=-1, max_value=10_000),
    label=st.text(max_size=50),
    size=st.integers(min_value=0, max_value=10_000),
)
def test_str_roundtrip_property(topic_id: int, label: str, size: int) -> None:
    t = make_topic(topic_id, label=label, size=size)
    assert isinstance(str(t), str)
    assert isinstance(repr(t), str)
