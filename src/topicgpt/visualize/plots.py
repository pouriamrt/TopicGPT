"""Plotly figures for inspecting fitted models.

Each function returns a ``plotly.graph_objects.Figure`` so callers can pipe
the result into ``fig.show()``, ``fig.write_html(...)``, or a notebook.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

from topicgpt.exceptions import TopicGPTError

if TYPE_CHECKING:
    from topicgpt.pipeline import TopicModel, _FitState


def visualize_topics(model: TopicModel, *, dim: int = 2) -> go.Figure:
    """2D scatter of cluster centroids using a fresh PCA."""
    state = _state(model)
    centroids = state.centroids
    if centroids.shape[0] == 0:
        raise TopicGPTError("Model has no topics to visualize.")
    n_comp = min(dim, max(1, centroids.shape[0] - 1), centroids.shape[1])
    if centroids.shape[1] > n_comp:
        proj = PCA(n_components=n_comp).fit_transform(centroids)
    else:
        proj = centroids
    x = proj[:, 0]
    y = proj[:, 1] if proj.shape[1] > 1 else np.zeros(proj.shape[0])
    sizes = [max(10, t.size) for t in model.topics_]
    labels = [t.display_name() for t in model.topics_]
    fig = go.Figure(
        data=[
            go.Scatter(
                x=x,
                y=y,
                mode="markers+text",
                text=labels,
                marker={"size": sizes, "sizemode": "area", "sizeref": 1.0},
                hovertext=labels,
            )
        ]
    )
    fig.update_layout(title="Topic centroids (PCA projection)", showlegend=False)
    return fig


def visualize_barchart(model: TopicModel, *, top_n: int = 8) -> go.Figure:
    """Per-topic barchart of the top-N keyword scores."""
    topics = model.topics_
    if not topics:
        raise TopicGPTError("Model has no topics to visualize.")
    fig = go.Figure()
    for t in topics:
        words = list(t.keywords[:top_n])
        scores = list(t.keyword_scores[:top_n])
        # plotly draws top-of-list at the bottom; reverse so the largest score
        # sits at the top of the per-topic block.
        fig.add_trace(
            go.Bar(
                x=list(reversed(scores)),
                y=list(reversed(words)),
                name=t.display_name(),
                orientation="h",
            )
        )
    fig.update_layout(
        title="Top words per topic",
        barmode="group",
        height=max(300, 80 * len(topics)),
    )
    return fig


def visualize_hierarchy(model: TopicModel) -> go.Figure:
    """Dendrogram of topics via Ward linkage on c-TF-IDF rows."""
    state = _state(model)
    if state.ctfidf_scores is None or state.ctfidf_scores.shape[0] < 2:
        raise TopicGPTError("Need at least two topics for a hierarchy.")
    Z = linkage(state.ctfidf_scores, method="ward")
    fig = _dendrogram_figure(Z, labels=[t.display_name() for t in model.topics_])
    fig.update_layout(title="Topic hierarchy")
    return fig


def visualize_heatmap(model: TopicModel) -> go.Figure:
    """Topic-topic cosine similarity heatmap."""
    state = _state(model)
    if state.centroids.shape[0] == 0:
        raise TopicGPTError("Model has no topics to visualize.")
    c = state.centroids
    norms = np.linalg.norm(c, axis=1, keepdims=True)
    cn = c / np.maximum(norms, 1e-12)
    sim = cn @ cn.T
    labels = [t.display_name() for t in model.topics_]
    fig = go.Figure(
        data=go.Heatmap(
            z=sim,
            x=labels,
            y=labels,
            colorscale="Viridis",
            zmin=-1,
            zmax=1,
        )
    )
    fig.update_layout(title="Topic similarity (cosine)")
    return fig


# ----------------------------------------------------------------------
# Helpers


def _state(model: TopicModel) -> _FitState:
    """Pull the protected ``_FitState`` off ``model`` after fit."""
    state: _FitState | None = getattr(model, "_state", None)
    if state is None:
        raise TopicGPTError("Model is not fitted yet; visualisation needs a fit first.")
    return state


def _dendrogram_figure(linkage_matrix: np.ndarray, *, labels: list[str]) -> go.Figure:
    """Minimal dendrogram renderer that avoids the scipy/plotly factory.

    We compute the leaf positions and intermediate node coordinates manually
    so we don't pull in ``plotly.figure_factory`` (which is being deprecated).
    """
    n_leaves = len(labels)
    # Leaf x positions in the order they were given to linkage.
    leaf_x: dict[int, float] = {i: float(i) for i in range(n_leaves)}
    leaf_height: dict[int, float] = dict.fromkeys(range(n_leaves), 0.0)

    fig = go.Figure()
    for i, row in enumerate(linkage_matrix):
        left, right, dist, _ = row
        left_i = int(left)
        right_i = int(right)
        x1 = leaf_x[left_i]
        x2 = leaf_x[right_i]
        y1 = leaf_height[left_i]
        y2 = leaf_height[right_i]
        new_id = n_leaves + i
        leaf_x[new_id] = (x1 + x2) / 2.0
        leaf_height[new_id] = float(dist)

        # Vertical lines from each child up to the merge level.
        fig.add_trace(go.Scatter(x=[x1, x1], y=[y1, dist], mode="lines", showlegend=False))
        fig.add_trace(go.Scatter(x=[x2, x2], y=[y2, dist], mode="lines", showlegend=False))
        # Horizontal connector.
        fig.add_trace(go.Scatter(x=[x1, x2], y=[dist, dist], mode="lines", showlegend=False))

    fig.update_layout(
        xaxis={"tickmode": "array", "tickvals": list(range(n_leaves)), "ticktext": labels},
        yaxis_title="Distance",
        showlegend=False,
    )
    return fig


# silence unused-import on scipy.spatial.distance — exposed for downstream users
_ = pdist
_ = squareform


__all__ = [
    "visualize_barchart",
    "visualize_heatmap",
    "visualize_hierarchy",
    "visualize_topics",
]
