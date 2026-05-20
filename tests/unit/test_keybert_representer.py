"""KeyBERTRepresenter tests."""

from __future__ import annotations

import numpy as np
import pytest

from topicgpt.config import KeyBERTRepresentationConfig
from topicgpt.exceptions import RepresentationError
from topicgpt.representation import KeyBERTRepresenter, Representer, TopicCandidate


def _scores() -> tuple[np.ndarray, list[str]]:
    vocab = ["lion", "tiger", "leopard", "ocean", "wave", "shore", "beach"]
    scores = np.array(
        [
            [0.9, 0.8, 0.7, 0.1, 0.05, 0.05, 0.05],  # topic 0: cats
            [0.05, 0.05, 0.05, 0.9, 0.8, 0.7, 0.6],  # topic 1: ocean
        ],
        dtype=np.float32,
    )
    return scores, vocab


@pytest.mark.unit
def test_picks_top_n_by_score() -> None:
    scores, vocab = _scores()
    rep = KeyBERTRepresenter(
        KeyBERTRepresentationConfig(top_n_words=3, diversity=0.0),
        scores=scores,
        vocab=vocab,
    )
    topics = rep.represent([TopicCandidate(topic_id=0), TopicCandidate(topic_id=1)])
    assert topics[0].keywords[:3] == ("lion", "tiger", "leopard")
    assert topics[1].keywords[:3] == ("ocean", "wave", "shore")


@pytest.mark.unit
def test_mmr_uses_word_embeddings() -> None:
    scores, vocab = _scores()
    # Make lion/tiger near-duplicates; leopard far from both → MMR should
    # prefer leopard over tiger when diversity is high.
    word_emb = np.array(
        [
            [1.0, 0.0],  # lion
            [0.99, 0.01],  # tiger ~= lion
            [0.0, 1.0],  # leopard (orthogonal)
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    rep = KeyBERTRepresenter(
        KeyBERTRepresentationConfig(top_n_words=2, diversity=0.99, candidate_pool=5),
        scores=scores,
        vocab=vocab,
        word_embeddings=word_emb,
    )
    [t0] = rep.represent([TopicCandidate(topic_id=0)])
    assert "leopard" in t0.keywords


@pytest.mark.unit
def test_unconfigured_scores_raises() -> None:
    with pytest.raises(RepresentationError, match="needs"):
        KeyBERTRepresenter().represent([TopicCandidate(topic_id=0)])


@pytest.mark.unit
def test_out_of_bounds_topic_id_raises() -> None:
    scores, vocab = _scores()
    rep = KeyBERTRepresenter(scores=scores, vocab=vocab)
    with pytest.raises(RepresentationError, match="out of bounds"):
        rep.represent([TopicCandidate(topic_id=99)])


@pytest.mark.unit
def test_representer_protocol_check() -> None:
    scores, vocab = _scores()
    rep = KeyBERTRepresenter(scores=scores, vocab=vocab)
    assert isinstance(rep, Representer)
