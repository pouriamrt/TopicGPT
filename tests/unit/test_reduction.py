"""DimReducer tests."""

from __future__ import annotations

import numpy as np
import pytest

from topicgpt.config import PCAConfig, UMAPConfig
from topicgpt.exceptions import ReductionError
from topicgpt.reduction import DimReducer, PCAReducer, UMAPReducer


def _blobs(n: int = 60, d: int = 16, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((3, d)).astype(np.float32) * 5
    blocks = [
        centers[c] + 0.3 * rng.standard_normal((n // 3, d)).astype(np.float32) for c in range(3)
    ]
    return np.vstack(blocks).astype(np.float32)


@pytest.mark.unit
def test_pca_reduces_to_n_components() -> None:
    X = _blobs()
    red = PCAReducer(PCAConfig(n_components=3))
    Y = red.fit_transform(X)
    assert Y.shape == (X.shape[0], 3)
    assert Y.dtype == np.float32


@pytest.mark.unit
def test_pca_transform_after_fit() -> None:
    red = PCAReducer(PCAConfig(n_components=3))
    red.fit_transform(_blobs())
    Y = red.transform(_blobs(n=12))
    assert Y.shape == (12, 3)


@pytest.mark.unit
def test_pca_transform_before_fit_raises() -> None:
    with pytest.raises(ReductionError, match="before fit"):
        PCAReducer(PCAConfig(n_components=3)).transform(_blobs())


@pytest.mark.unit
def test_pca_rejects_too_few_rows() -> None:
    red = PCAReducer(PCAConfig(n_components=10))
    with pytest.raises(ReductionError):
        red.fit_transform(_blobs(n=5))


@pytest.mark.unit
def test_pca_rejects_1d_input() -> None:
    with pytest.raises(ReductionError):
        PCAReducer().fit_transform(np.zeros(10, dtype=np.float32))


@pytest.mark.unit
def test_dim_reducer_protocol_recognised() -> None:
    assert isinstance(PCAReducer(), DimReducer)
    assert isinstance(UMAPReducer(UMAPConfig(n_neighbors=2, n_components=2)), DimReducer)


@pytest.mark.unit
def test_umap_round_trip_with_stubbed_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify wiring without paying UMAP's startup cost."""
    import sys
    from unittest.mock import MagicMock

    stub = MagicMock()
    fake = MagicMock()
    fake.fit_transform.return_value = np.zeros((90, 3), dtype=np.float32)
    fake.transform.return_value = np.zeros((5, 3), dtype=np.float32)
    stub.UMAP.return_value = fake
    monkeypatch.setitem(sys.modules, "umap", stub)

    red = UMAPReducer(UMAPConfig(n_components=3, n_neighbors=10, random_state=0))
    Y = red.fit_transform(_blobs(n=90, d=16))
    assert Y.shape == (90, 3)
    Y2 = red.transform(_blobs(n=5, d=16))
    assert Y2.shape == (5, 3)
    # UMAP() was constructed with our config knobs.
    kwargs = stub.UMAP.call_args.kwargs
    assert kwargs["n_components"] == 3
    assert kwargs["n_neighbors"] == 10
    assert kwargs["random_state"] == 0


@pytest.mark.unit
def test_umap_rejects_small_input() -> None:
    red = UMAPReducer(UMAPConfig(n_components=3, n_neighbors=10))
    with pytest.raises(ReductionError):
        red.fit_transform(_blobs(n=5))


@pytest.mark.unit
def test_umap_transform_before_fit_raises() -> None:
    red = UMAPReducer(UMAPConfig(n_components=3, n_neighbors=10))
    with pytest.raises(ReductionError):
        red.transform(_blobs())
