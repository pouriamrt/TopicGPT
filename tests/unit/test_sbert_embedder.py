"""SBERTEmbedder tests with a stub SentenceTransformer."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from topicgpt.config import SBERTEmbeddingConfig
from topicgpt.embeddings.sbert import SBERTEmbedder
from topicgpt.exceptions import EmbeddingError


def _stub_model(dim: int = 8) -> MagicMock:
    m = MagicMock()
    m.get_sentence_embedding_dimension.return_value = dim
    m.encode.side_effect = lambda texts, **_: np.stack(
        [np.full(dim, fill_value=float(i), dtype=np.float32) for i, _ in enumerate(texts)]
    )
    return m


@pytest.mark.unit
def test_embeds_with_injected_model() -> None:
    model = _stub_model(dim=8)
    emb = SBERTEmbedder(SBERTEmbeddingConfig(model="x"), model=model)
    out = emb.embed(["a", "b", "c"])
    assert out.shape == (3, 8)
    assert out.dtype == np.float32
    assert emb.dim == 8


@pytest.mark.unit
def test_model_name_passthrough() -> None:
    emb = SBERTEmbedder(SBERTEmbeddingConfig(model="custom/x"), model=_stub_model())
    assert emb.model_name == "custom/x"


@pytest.mark.unit
def test_empty_raises() -> None:
    emb = SBERTEmbedder(model=_stub_model())
    with pytest.raises(EmbeddingError):
        emb.embed([])


@pytest.mark.unit
def test_encode_kwargs_propagate() -> None:
    model = _stub_model(dim=4)
    emb = SBERTEmbedder(
        SBERTEmbeddingConfig(model="x", batch_size=32, device="cpu", normalize_embeddings=False),
        model=model,
    )
    emb.embed(["a"])
    kwargs = model.encode.call_args.kwargs
    assert kwargs["batch_size"] == 32
    assert kwargs["device"] == "cpu"
    assert kwargs["normalize_embeddings"] is False
    assert kwargs["convert_to_numpy"] is True


@pytest.mark.unit
def test_auto_device_passes_none() -> None:
    model = _stub_model()
    emb = SBERTEmbedder(SBERTEmbeddingConfig(model="x", device="auto"), model=model)
    emb.embed(["a"])
    assert model.encode.call_args.kwargs["device"] is None
