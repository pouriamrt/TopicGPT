"""Disk-cache tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from topicgpt.embeddings._cache import EmbeddingCache

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.unit
def test_round_trip(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path, model="m", dim=8)
    vec = np.arange(8, dtype=np.float32)
    cache.put("hello", vec)
    out = cache.get("hello")
    assert out is not None
    np.testing.assert_array_equal(out, vec)


@pytest.mark.unit
def test_miss_returns_none(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path, model="m", dim=8)
    assert cache.get("absent") is None


@pytest.mark.unit
def test_namespace_isolates_models(tmp_path: Path) -> None:
    c1 = EmbeddingCache(tmp_path, model="m1", dim=8)
    c2 = EmbeddingCache(tmp_path, model="m2", dim=8)
    c1.put("hi", np.arange(8, dtype=np.float32))
    assert c2.get("hi") is None


@pytest.mark.unit
def test_namespace_isolates_dimensions(tmp_path: Path) -> None:
    c1 = EmbeddingCache(tmp_path, model="m", dim=8)
    c2 = EmbeddingCache(tmp_path, model="m", dim=16)
    c1.put("hi", np.arange(8, dtype=np.float32))
    assert c2.get("hi") is None


@pytest.mark.unit
def test_corrupted_file_returns_none(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path, model="m", dim=8)
    cache.put("hi", np.arange(8, dtype=np.float32))
    # corrupt the on-disk file
    sole = next(tmp_path.rglob("*.npy"))
    sole.write_bytes(b"not a numpy file")
    assert cache.get("hi") is None
