"""LLM representer — labels and describes topics via OpenAI structured outputs.

Default model is ``gpt-5.4-mini``. Calls ``client.responses.parse`` with a
Pydantic ``TopicLabel`` schema so the LLM is forced to return well-typed JSON.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from pydantic import ValidationError

from topicgpt._openai_common import RETRYABLE_ERRORS, build_openai_client, with_openai_retry
from topicgpt.config import LLMRepresentationConfig, Settings
from topicgpt.exceptions import LLMResponseError, RepresentationError
from topicgpt.representation.schemas import TopicLabel
from topicgpt.topic import Topic, make_topic

if TYPE_CHECKING:
    from collections.abc import Sequence

    from openai import OpenAI

    from topicgpt.representation.base import TopicCandidate


_DEFAULT_SYSTEM_PROMPT = (
    "You are a topic-modelling assistant. Given representative keywords and "
    "example documents from one topic, produce a short label, a 1-2 sentence "
    "description, and a refined keyword list. Stay strictly grounded in the "
    "provided text - do not invent facts."
)


class LLMRepresenter:
    """Use an OpenAI chat model to label + describe topics."""

    name = "llm"

    def __init__(
        self,
        config: LLMRepresentationConfig | None = None,
        *,
        settings: Settings | None = None,
        client: OpenAI | None = None,
    ) -> None:
        self.config = config or LLMRepresentationConfig()
        self.settings = settings or Settings()
        self._client = client or build_openai_client(self.settings)

    @property
    def model_name(self) -> str:
        """Underlying OpenAI chat model id."""
        return self.config.model

    def represent(self, candidates: Sequence[TopicCandidate]) -> list[Topic]:
        """Label each candidate via the LLM and return enriched topics.

        ``keyword_scores`` is intentionally empty on the returned topics: the
        LLM rewrites the keyword list, so prior representers' scores no longer
        align. Pair with KeyBERT/c-TF-IDF if you need ranked scores.
        """
        out: list[Topic] = []
        for cand in candidates:
            label = self._label_one(cand)
            out.append(
                make_topic(
                    cand.topic_id,
                    label=label.label,
                    description=label.description,
                    keywords=tuple(label.keywords),
                    keyword_scores=(),
                    representative_docs=cand.representative_docs,
                    size=cand.size,
                    meta={"llm_model": self.config.model},
                )
            )
        return out

    def _label_one(self, cand: TopicCandidate) -> TopicLabel:
        prompt = self._build_prompt(cand)
        kwargs: dict[str, object] = {
            "model": self.config.model,
            "input": cast("Any", prompt),
            "text_format": TopicLabel,
            "max_output_tokens": self.config.max_output_tokens,
        }
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature

        def _call() -> TopicLabel:
            try:
                rsp = self._client.responses.parse(**kwargs)  # type: ignore[arg-type]
            except RETRYABLE_ERRORS:
                raise
            except Exception as e:
                raise LLMResponseError(f"LLM call failed: {e}") from e

            parsed: object = rsp.output_parsed
            if parsed is None:
                raise LLMResponseError("LLM response missing parsed payload")
            try:
                return TopicLabel.model_validate(parsed)
            except ValidationError as e:
                raise LLMResponseError(f"LLM returned invalid schema: {e}") from e

        try:
            return with_openai_retry(self.settings, _call)
        except RETRYABLE_ERRORS as e:
            raise RepresentationError(f"LLM call exhausted retries: {e}") from e

    def _build_prompt(self, cand: TopicCandidate) -> list[dict[str, str]]:
        sys = self.config.system_prompt or _DEFAULT_SYSTEM_PROMPT
        if self.config.corpus_instruction:
            sys = f"{sys}\n\nCorpus context: {self.config.corpus_instruction}"

        kws = ", ".join(cand.keywords[: self.config.top_n_words]) or "(none)"
        docs = "\n\n---\n".join(cand.representative_docs[: self.config.n_representative_docs])
        if not docs:
            docs = "(no representative documents)"

        user = (
            f"Topic id: {cand.topic_id} (size={cand.size})\n"
            f"Top keywords: {kws}\n\n"
            f"Representative documents:\n{docs}"
        )
        return [
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ]


__all__ = ["LLMRepresenter"]
