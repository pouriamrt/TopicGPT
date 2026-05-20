"""Custom-pipeline example: swap defaults, run eval, save the model."""

from __future__ import annotations

from pathlib import Path

from topicgpt import (
    CTFIDFConfig,
    CTFIDFVectorizer,
    KeyBERTRepresentationConfig,
    KeyBERTRepresenter,
    KMeansClusterer,
    KMeansConfig,
    OpenAIEmbedder,
    OpenAIEmbeddingConfig,
    PCAConfig,
    PCAReducer,
    TopicModel,
)
from topicgpt.evaluation import inverted_rbo, npmi, proportion_unique_words

DOCS = [
    "lions hunt zebras",
    "tigers stalk deer",
    "cheetahs are fast",
    "ocean tides shape beaches",
    "waves crash on sand",
    "sea surf in winter",
]


def main() -> None:
    model = TopicModel(
        embedder=OpenAIEmbedder(OpenAIEmbeddingConfig(dimensions=512)),
        reducer=PCAReducer(PCAConfig(n_components=4)),
        clusterer=KMeansClusterer(KMeansConfig(n_clusters=2, random_state=0)),
        vectorizer=CTFIDFVectorizer(CTFIDFConfig(bm25_weighting=True)),
        representers=[KeyBERTRepresenter(KeyBERTRepresentationConfig(diversity=0.5))],
    )
    model.fit(DOCS)
    print(model.get_topic_info().to_string(index=False))

    topic_words = [list(t.keywords) for t in model.topics_]
    print(f"NPMI:      {npmi(DOCS, topic_words):+.4f}")
    print(f"Diversity: {proportion_unique_words(topic_words):.4f}")
    print(f"RBO:       {inverted_rbo(topic_words):.4f}")

    out = Path("./out_model")
    model.save(out)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
