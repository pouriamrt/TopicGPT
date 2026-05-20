"""Exception hierarchy tests."""

from __future__ import annotations

import pytest

from topicgpt import exceptions as exc


@pytest.mark.unit
def test_all_inherit_from_base() -> None:
    for cls in (
        exc.ConfigurationError,
        exc.EmbeddingError,
        exc.ReductionError,
        exc.ClusteringError,
        exc.VectorizationError,
        exc.RepresentationError,
        exc.PersistenceError,
    ):
        assert issubclass(cls, exc.TopicGPTError)


@pytest.mark.unit
def test_llm_response_error_is_representation_error() -> None:
    assert issubclass(exc.LLMResponseError, exc.RepresentationError)
    assert issubclass(exc.LLMResponseError, exc.TopicGPTError)


@pytest.mark.unit
def test_raising_and_catching() -> None:
    with pytest.raises(exc.TopicGPTError, match="boom"):
        raise exc.EmbeddingError("boom")
