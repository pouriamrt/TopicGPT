"""OpenAIEmbedder tests with a mocked SDK client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from topicgpt.config import OpenAIEmbeddingConfig, Settings
from topicgpt.embeddings.openai import OpenAIEmbedder
from topicgpt.exceptions import ConfigurationError, EmbeddingError

if TYPE_CHECKING:
    from pathlib import Path

# ---------- fake OpenAI response shape ----------


@dataclass
class _FakeDatum:
    index: int
    embedding: list[float]


@dataclass
class _FakeResponse:
    data: list[_FakeDatum]


def _make_response(batch: list[str], dim: int = 4) -> _FakeResponse:
    data = [
        _FakeDatum(index=i, embedding=[float(i + j) for j in range(dim)])
        for i, _ in enumerate(batch)
    ]
    return _FakeResponse(data=data)


def _build_embedder(
    *,
    batch_size: int = 256,
    dimensions: int | None = None,
    cache_dir: Path | None = None,
) -> tuple[OpenAIEmbedder, MagicMock]:
    client = MagicMock()
    client.embeddings.create.side_effect = lambda **kw: _make_response(
        list(kw["input"]),
        dim=dimensions or 4,
    )
    cfg = OpenAIEmbeddingConfig(
        model="text-embedding-3-large",
        batch_size=batch_size,
        dimensions=dimensions,
    )
    emb = OpenAIEmbedder(cfg, settings=Settings(), client=client, cache_dir=cache_dir)
    return emb, client


# ---------- core behavior ----------


@pytest.mark.unit
def test_default_model_is_text_embedding_3_large() -> None:
    emb, _ = _build_embedder(dimensions=64)
    assert emb.model_name == "text-embedding-3-large"


@pytest.mark.unit
def test_dim_unknown_before_first_call() -> None:
    emb, _ = _build_embedder()
    with pytest.raises(EmbeddingError):
        _ = emb.dim


@pytest.mark.unit
def test_dim_set_after_call() -> None:
    emb, _ = _build_embedder(dimensions=64)
    out = emb.embed(["one", "two"])
    assert out.shape == (2, 64)
    assert out.dtype == np.float32
    assert emb.dim == 64


@pytest.mark.unit
def test_empty_input_raises() -> None:
    emb, _ = _build_embedder()
    with pytest.raises(EmbeddingError):
        emb.embed([])


@pytest.mark.unit
def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ConfigurationError):
        OpenAIEmbedder()


# ---------- batching ----------


@pytest.mark.unit
def test_batches_respect_batch_size() -> None:
    emb, client = _build_embedder(batch_size=2, dimensions=64)
    emb.embed(["a", "b", "c", "d", "e"])
    # 5 inputs / batch_size 2 => 3 calls (2, 2, 1)
    assert client.embeddings.create.call_count == 3
    batch_sizes = [len(call.kwargs["input"]) for call in client.embeddings.create.call_args_list]
    assert batch_sizes == [2, 2, 1]


@pytest.mark.unit
def test_batches_capped_at_openai_request_limit() -> None:
    emb, client = _build_embedder(batch_size=2048, dimensions=64)
    n = 2049
    emb.embed([f"text-{i}" for i in range(n)])
    sizes = [len(call.kwargs["input"]) for call in client.embeddings.create.call_args_list]
    assert all(s <= 2048 for s in sizes)
    assert sum(sizes) == n


@pytest.mark.unit
def test_dimensions_kwarg_forwarded_when_set() -> None:
    emb, client = _build_embedder(batch_size=10, dimensions=64)
    emb.embed(["x"])
    sent = client.embeddings.create.call_args.kwargs
    assert sent["dimensions"] == 64


@pytest.mark.unit
def test_dimensions_kwarg_omitted_when_none() -> None:
    emb, client = _build_embedder(batch_size=10, dimensions=None)
    emb.embed(["x"])
    sent = client.embeddings.create.call_args.kwargs
    assert "dimensions" not in sent


@pytest.mark.unit
def test_results_are_reordered_by_index() -> None:
    """OpenAI guarantees index field reflects input order even if shuffled in response."""
    client = MagicMock()

    def shuffled(**kw: Any) -> _FakeResponse:
        rsp = _make_response(list(kw["input"]), dim=64)
        rsp.data = list(reversed(rsp.data))
        return rsp

    client.embeddings.create.side_effect = shuffled
    emb = OpenAIEmbedder(
        OpenAIEmbeddingConfig(batch_size=3, dimensions=64),
        settings=Settings(),
        client=client,
    )
    out = emb.embed(["a", "b", "c"])
    # _make_response builds row i as [i, i+1, ..., i+dim-1]; sort by .index restores order
    assert out.shape == (3, 64)
    np.testing.assert_array_equal(out[0, :4], np.array([0, 1, 2, 3], dtype=np.float32))
    np.testing.assert_array_equal(out[2, :4], np.array([2, 3, 4, 5], dtype=np.float32))


# ---------- caching ----------


@pytest.mark.unit
def test_cache_hit_skips_api(tmp_path: Path) -> None:
    emb, client = _build_embedder(batch_size=10, dimensions=64, cache_dir=tmp_path)
    emb.embed(["alpha", "beta"])
    first_calls = client.embeddings.create.call_count
    # Second call: same inputs should hit cache only.
    emb.embed(["alpha", "beta"])
    assert client.embeddings.create.call_count == first_calls


@pytest.mark.unit
def test_cache_partial_hit_sends_only_misses(tmp_path: Path) -> None:
    emb, client = _build_embedder(batch_size=10, dimensions=64, cache_dir=tmp_path)
    emb.embed(["alpha"])
    client.embeddings.create.reset_mock()
    emb.embed(["alpha", "beta"])  # alpha cached, beta is a miss
    sent = client.embeddings.create.call_args.kwargs["input"]
    assert sent == ["beta"]


# ---------- truncation ----------


@pytest.mark.unit
def test_long_input_truncated_to_max_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    emb, client = _build_embedder(batch_size=10, dimensions=64)
    # Replace tokenizer with a deterministic stub
    long = "word " * 20_000
    emb.embed([long])
    sent = client.embeddings.create.call_args.kwargs["input"][0]
    # tokeniser-based truncation: ensure the payload is shorter than the source.
    assert len(sent) < len(long)


@pytest.mark.unit
def test_blank_input_replaced_with_space() -> None:
    emb, client = _build_embedder(batch_size=10, dimensions=64)
    emb.embed([""])
    sent = client.embeddings.create.call_args.kwargs["input"][0]
    assert sent == " "


# ---------- retry ----------


@pytest.mark.unit
def test_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    from openai import APIConnectionError

    # tenacity sleeps between attempts — patch to make tests fast.
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda _: None)

    client = MagicMock()
    call_count = {"n": 0}

    def flaky(**kw: Any) -> _FakeResponse:
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise APIConnectionError(request=MagicMock())
        return _make_response(list(kw["input"]), dim=64)

    client.embeddings.create.side_effect = flaky
    emb = OpenAIEmbedder(
        OpenAIEmbeddingConfig(batch_size=10, dimensions=64),
        settings=Settings(max_retries=4),
        client=client,
    )
    out = emb.embed(["hi"])
    assert out.shape == (1, 64)
    assert call_count["n"] == 3


@pytest.mark.unit
def test_retries_exhausted_raises_embedding_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from openai import APIConnectionError

    monkeypatch.setattr("tenacity.nap.time.sleep", lambda _: None)
    client = MagicMock()
    client.embeddings.create.side_effect = APIConnectionError(request=MagicMock())
    emb = OpenAIEmbedder(
        OpenAIEmbeddingConfig(batch_size=10, dimensions=64),
        settings=Settings(max_retries=1),
        client=client,
    )
    with pytest.raises(EmbeddingError):
        emb.embed(["hi"])
