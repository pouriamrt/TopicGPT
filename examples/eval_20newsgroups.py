"""Benchmark on a small slice of 20newsgroups."""

from __future__ import annotations

from topicgpt import (
    CTFIDFVectorizer,
    HDBSCANClusterer,
    HDBSCANConfig,
    KeyBERTRepresenter,
    OpenAIEmbedder,
    TopicModel,
    UMAPReducer,
)
from topicgpt.evaluation import run_bench


def _load_docs(limit: int = 400) -> list[str]:
    from sklearn.datasets import fetch_20newsgroups

    raw = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
    return [d.strip() for d in raw.data[:limit] if d.strip()]


def _make_model(seed: int) -> TopicModel:
    return TopicModel(
        embedder=OpenAIEmbedder(),
        reducer=UMAPReducer(),
        clusterer=HDBSCANClusterer(HDBSCANConfig(min_cluster_size=10)),
        vectorizer=CTFIDFVectorizer(),
        representers=[KeyBERTRepresenter()],
    )


def main() -> None:
    docs = _load_docs()
    result = run_bench(_make_model, docs, seeds=(0, 1, 2))
    print(result)


if __name__ == "__main__":
    main()
