"""Plotly figures for inspecting fitted models.

Each function returns a ``plotly.graph_objects.Figure`` so callers can pipe
the result into ``fig.show()``, ``fig.write_html(...)``, or a notebook.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA

from topicgpt._math import cosine_similarity
from topicgpt.exceptions import TopicGPTError

if TYPE_CHECKING:
    from topicgpt.pipeline import TopicModel


def visualize_topics(model: TopicModel, *, dim: int = 2) -> go.Figure:
    """2D scatter of cluster centroids using a fresh PCA."""
    centroids = model.state_.centroids
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
    scores = model.state_.ctfidf_scores
    if scores.shape[0] < 2:
        raise TopicGPTError("Need at least two topics for a hierarchy.")
    linkage_matrix = linkage(scores, method="ward")
    labels = [t.display_name() for t in model.topics_]
    # scipy returns ready-to-plot leaf x/y coordinates; one Scatter per U-link.
    rendered = dendrogram(linkage_matrix, no_plot=True, labels=labels)
    fig = go.Figure()
    for xs, ys in zip(rendered["icoord"], rendered["dcoord"], strict=True):
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", showlegend=False))
    n_leaves = len(labels)
    fig.update_layout(
        title="Topic hierarchy",
        xaxis={
            "tickmode": "array",
            "tickvals": [5 + 10 * i for i in range(n_leaves)],
            "ticktext": rendered["ivl"],
        },
        yaxis_title="Distance",
        showlegend=False,
    )
    return fig


def visualize_heatmap(model: TopicModel) -> go.Figure:
    """Topic-topic cosine similarity heatmap."""
    centroids = model.state_.centroids
    if centroids.shape[0] == 0:
        raise TopicGPTError("Model has no topics to visualize.")
    sim = cosine_similarity(centroids, centroids)
    labels = [t.display_name() for t in model.topics_]
    fig = go.Figure(
        data=go.Heatmap(z=sim, x=labels, y=labels, colorscale="Viridis", zmin=-1, zmax=1)
    )
    fig.update_layout(title="Topic similarity (cosine)")
    return fig


__all__ = [
    "visualize_barchart",
    "visualize_heatmap",
    "visualize_hierarchy",
    "visualize_topics",
]
