"""TopicGPT v1.0 quickstart.

Requires ``OPENAI_API_KEY`` in the environment.
Run via: ``uv run python examples/quickstart.py``
"""

from __future__ import annotations

from topicgpt import (
    CTFIDFVectorizer,
    HDBSCANClusterer,
    HDBSCANConfig,
    KeyBERTRepresenter,
    LLMRepresenter,
    OpenAIEmbedder,
    TopicModel,
    UMAPReducer,
)

DOCUMENTS = [
    "Lions hunt in prides on the savanna.",
    "Tigers are solitary apex predators in Asian forests.",
    "Cheetahs are the fastest land animals on Earth.",
    "Climate change is melting Arctic sea ice at record pace.",
    "Carbon emissions and global warming threaten coral reefs.",
    "Renewable energy adoption is accelerating worldwide.",
    "Quantum computers exploit superposition for certain speedups.",
    "Photonic processors promise faster, cooler computation.",
    "Neural networks have transformed machine learning.",
]


def main() -> None:
    model = TopicModel(
        embedder=OpenAIEmbedder(),  # defaults to text-embedding-3-small
        reducer=UMAPReducer(),
        clusterer=HDBSCANClusterer(HDBSCANConfig(min_cluster_size=2)),
        vectorizer=CTFIDFVectorizer(),
        representers=[KeyBERTRepresenter(), LLMRepresenter()],
    )
    model.fit(DOCUMENTS)
    print(model.get_topic_info().to_string(index=False))

    for topic, score in model.find_topics("global warming", top_k=2):
        print(f"  {score:+.3f}  {topic.label or topic.display_name()}")


if __name__ == "__main__":
    main()
