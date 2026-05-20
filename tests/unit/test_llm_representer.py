"""LLMRepresenter tests with a mocked OpenAI Responses API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from topicgpt.config import LLMRepresentationConfig, Settings
from topicgpt.exceptions import ConfigurationError, LLMResponseError
from topicgpt.representation import LLMRepresenter, TopicCandidate, TopicLabel


@dataclass
class _FakeResp:
    output_parsed: object


def _client_returning(label: TopicLabel) -> MagicMock:
    client = MagicMock()
    client.responses.parse.return_value = _FakeResp(output_parsed=label.model_dump())
    return client


@pytest.fixture
def fake_label() -> TopicLabel:
    return TopicLabel(
        label="Big cats",
        description="Documents about lions, tigers, and other large felines.",
        keywords=["lion", "tiger", "leopard", "cheetah"],
    )


@pytest.mark.unit
def test_default_model_is_gpt54mini() -> None:
    rep = LLMRepresenter(client=MagicMock())
    assert rep.model_name == "gpt-5.4-mini"


@pytest.mark.unit
def test_represent_returns_enriched_topics(fake_label: TopicLabel) -> None:
    rep = LLMRepresenter(client=_client_returning(fake_label))
    [topic] = rep.represent(
        [
            TopicCandidate(
                topic_id=3,
                keywords=("lion", "tiger"),
                keyword_scores=(0.9, 0.8),
                representative_docs=("Lions hunt in prides.",),
                size=20,
            )
        ]
    )
    assert topic.id == 3
    assert topic.label == fake_label.label
    assert topic.description == fake_label.description
    assert topic.keywords == tuple(fake_label.keywords)
    assert topic.size == 20
    assert topic.meta["llm_model"] == "gpt-5.4-mini"


@pytest.mark.unit
def test_response_parse_called_with_text_format(fake_label: TopicLabel) -> None:
    client = _client_returning(fake_label)
    LLMRepresenter(client=client).represent([TopicCandidate(topic_id=0, keywords=("a",))])
    sent = client.responses.parse.call_args.kwargs
    assert sent["model"] == "gpt-5.4-mini"
    assert sent["text_format"] is TopicLabel


@pytest.mark.unit
def test_temperature_forwarded_when_set(fake_label: TopicLabel) -> None:
    client = _client_returning(fake_label)
    rep = LLMRepresenter(LLMRepresentationConfig(temperature=0.2), client=client)
    rep.represent([TopicCandidate(topic_id=0)])
    sent = client.responses.parse.call_args.kwargs
    assert sent["temperature"] == 0.2


@pytest.mark.unit
def test_missing_output_parsed_raises() -> None:
    client = MagicMock()
    client.responses.parse.return_value = _FakeResp(output_parsed=None)
    with pytest.raises(LLMResponseError, match="missing"):
        LLMRepresenter(client=client).represent([TopicCandidate(topic_id=0)])


@pytest.mark.unit
def test_invalid_schema_raises() -> None:
    client = MagicMock()
    client.responses.parse.return_value = _FakeResp(output_parsed={"label": "x"})  # missing keys
    with pytest.raises(LLMResponseError):
        LLMRepresenter(client=client).represent([TopicCandidate(topic_id=0)])


@pytest.mark.unit
def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ConfigurationError):
        LLMRepresenter()


@pytest.mark.unit
def test_prompt_contains_keywords_and_docs(fake_label: TopicLabel) -> None:
    client = _client_returning(fake_label)
    LLMRepresenter(client=client).represent(
        [
            TopicCandidate(
                topic_id=1,
                keywords=("lion", "tiger"),
                representative_docs=("Lions roar.", "Tigers stalk."),
                size=5,
            )
        ]
    )
    messages = client.responses.parse.call_args.kwargs["input"]
    user_msg = next(m for m in messages if m["role"] == "user")
    assert "lion" in user_msg["content"]
    assert "Lions roar" in user_msg["content"]


@pytest.mark.unit
def test_corpus_instruction_appended_to_system_prompt(fake_label: TopicLabel) -> None:
    client = _client_returning(fake_label)
    rep = LLMRepresenter(
        LLMRepresentationConfig(corpus_instruction="wildlife magazine articles"),
        client=client,
    )
    rep.represent([TopicCandidate(topic_id=0)])
    messages = client.responses.parse.call_args.kwargs["input"]
    sys_msg = next(m for m in messages if m["role"] == "system")
    assert "wildlife magazine" in sys_msg["content"]


@pytest.mark.unit
def test_retries_then_succeeds(
    fake_label: TopicLabel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openai import APIConnectionError

    monkeypatch.setattr("tenacity.nap.time.sleep", lambda _: None)

    client = MagicMock()
    counter = {"n": 0}

    def flaky(**_kw: Any) -> _FakeResp:
        counter["n"] += 1
        if counter["n"] < 3:
            raise APIConnectionError(request=MagicMock())
        return _FakeResp(output_parsed=fake_label.model_dump())

    client.responses.parse.side_effect = flaky
    rep = LLMRepresenter(client=client, settings=Settings(max_retries=4))
    [t] = rep.represent([TopicCandidate(topic_id=0)])
    assert t.label == fake_label.label
    assert counter["n"] == 3
