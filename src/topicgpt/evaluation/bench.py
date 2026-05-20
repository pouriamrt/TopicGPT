"""OCTIS-style benchmark harness.

Runs a model factory across multiple seeds, collects metric values, and
returns mean / std per metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean, stdev
from typing import TYPE_CHECKING

from topicgpt.evaluation.coherence import npmi, umass
from topicgpt.evaluation.diversity import inverted_rbo, proportion_unique_words

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from topicgpt.pipeline import TopicModel


_DEFAULT_METRICS = ("npmi", "diversity", "rbo", "umass")


@dataclass(frozen=True)
class BenchResult:
    """Aggregated benchmark output."""

    metrics: dict[str, float]
    stds: dict[str, float]
    per_run: list[dict[str, float]]

    def __str__(self) -> str:
        """Pretty single-line summary."""
        return ", ".join(f"{k}={self.metrics[k]:+.4f}±{self.stds[k]:.4f}" for k in self.metrics)


def run_bench(
    model_factory: Callable[[int], TopicModel],
    documents: Sequence[str],
    *,
    seeds: Sequence[int] = (0, 1, 2),
    metrics: Sequence[str] = _DEFAULT_METRICS,
    top_n: int = 10,
) -> BenchResult:
    """Run ``model_factory(seed).fit(documents)`` for each seed and aggregate."""
    if not seeds:
        raise ValueError("Need at least one seed.")

    per_run: list[dict[str, float]] = []
    for s in seeds:
        model = model_factory(s)
        model.fit(documents)
        topic_words = [list(t.keywords[:top_n]) for t in model.topics_]
        scores: dict[str, float] = {}
        if "npmi" in metrics:
            scores["npmi"] = npmi(documents, topic_words, top_n=top_n)
        if "umass" in metrics:
            scores["umass"] = umass(documents, topic_words, top_n=top_n)
        if "diversity" in metrics:
            scores["diversity"] = proportion_unique_words(topic_words, top_n=top_n)
        if "rbo" in metrics:
            scores["rbo"] = inverted_rbo(topic_words, top_n=top_n)
        per_run.append(scores)

    keys = sorted({k for r in per_run for k in r})
    means = {k: float(fmean(r[k] for r in per_run)) for k in keys}
    stds = {k: float(stdev(r[k] for r in per_run)) if len(per_run) > 1 else 0.0 for k in keys}
    return BenchResult(metrics=means, stds=stds, per_run=per_run)


__all__ = ["BenchResult", "run_bench"]
