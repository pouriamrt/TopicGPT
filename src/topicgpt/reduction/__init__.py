"""Dimensionality reduction back-ends."""

from __future__ import annotations

from topicgpt.reduction.base import DimReducer
from topicgpt.reduction.pca import PCAReducer
from topicgpt.reduction.umap_reducer import UMAPReducer

__all__ = ["DimReducer", "PCAReducer", "UMAPReducer"]
