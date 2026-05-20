"""LLM representer — labels and describes topics via OpenAI structured outputs.

Default model is ``gpt-5.4-mini``. Calls ``client.responses.parse`` with a
Pydantic ``TopicLabel`` schema so the LLM is forced to return well-typed JSON.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from openai import APIConnectionError, APIError, OpenAI, RateLimitError
from pydantic import ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from topicgpt.config import LLMRepresentationConfig, Settings
from topicgpt.exceptions import ConfigurationError, LLMResponseError, RepresentationError
from topicgpt.representation.schemas import TopicLabel
from topicgpt.topic import Topic, make_topic

if TYPE_CHECKING:
    from collections.abc import Sequence

    from topicgpt.representation.base import TopicCandidate


_RETRYABLE_ERRORS = (APIConnectionError, RateLimitError, APIError)

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
        self._client = client or self._build_client()

    @property
    def model_name(self) -> str:
        """Underlying OpenAI chat model id."""
        return self.config.model

    def represent(self, candidates: Sequence[TopicCandidate]) -> list[Topic]:
        """Label each candidate via the LLM and return enriched topics."""
        out: list[Topic] = []
        for cand in candidates:
            label = self._label_one(cand)
            kw_scores = _pad_or_truncate(cand.keyword_scores, len(label.keywords))
            out.append(
                make_topic(
                    cand.topic_id,
                    label=label.label,
                    description=label.description,
                    keywords=tuple(label.keywords),
                    keyword_scores=kw_scores,
                    representative_docs=cand.representative_docs,
                    size=cand.size,
                    meta={"llm_model": self.config.model},
                )
            )
        return out

    # ------------------------------------------------------------------
    # Internals

    def _build_client(self) -> OpenAI:
        if self.settings.openai_api_key is None:
            raise ConfigurationError("OPENAI_API_KEY is required for LLMRepresenter.")
        return OpenAI(
            api_key=self.settings.openai_api_key.get_secret_value(),
            timeout=self.settings.request_timeout_s,
            max_retries=0,
        )

    def _label_one(self, cand: TopicCandidate) -> TopicLabel:
        prompt = self._build_prompt(cand)

        @retry(
            retry=retry_if_exception_type(_RETRYABLE_ERRORS),
            wait=wait_exponential(multiplier=1, min=1, max=20),
            stop=stop_after_attempt(self.settings.max_retries + 1),
            reraise=True,
        )
        def _call() -> TopicLabel:
            try:
                # The OpenAI SDK uses tightly-typed TypedDicts for `input`;
                # our generic role/content dicts conform structurally so we
                # cast to keep mypy quiet without losing runtime validation.
                api_input = cast("Any", prompt)
                if self.config.temperature is None:
                    rsp = self._client.responses.parse(
                        model=self.config.model,
                        input=api_input,
                        text_format=TopicLabel,
                        max_output_tokens=self.config.max_output_tokens,
                    )
                else:
                    rsp = self._client.responses.parse(
                        model=self.config.model,
                        input=api_input,
                        text_format=TopicLabel,
                        max_output_tokens=self.config.max_output_tokens,
                        temperature=self.config.temperature,
                    )
            except _RETRYABLE_ERRORS:
                raise
            except Exception as e:
                raise LLMResponseError(f"LLM call failed: {e}") from e

            parsed = getattr(rsp, "output_parsed", None)
            if parsed is None:
                raise LLMResponseError("LLM response missing parsed payload")
            try:
                return TopicLabel.model_validate(parsed)
            except ValidationError as e:
                raise LLMResponseError(f"LLM returned invalid schema: {e}") from e

        try:
            return _call()
        except _RETRYABLE_ERRORS as e:
            raise RepresentationError(f"LLM call exhausted retries: {e}") from e

    def _build_prompt(self, cand: TopicCandidate) -> list[dict[str, str]]:
        sys = self.config.system_prompt or _DEFAULT_SYSTEM_PROMPT
        if self.config.corpus_instruction:
            sys = f"{sys}\n\nCorpus context: {self.config.corpus_instruction}"

        kws = ", ".join(cand.keywords[: self.config.n_top_words]) or "(none)"
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


def _pad_or_truncate(scores: tuple[float, ...], n: int) -> tuple[float, ...]:
    """Right-pad with 0.0 or truncate so the result has exactly ``n`` items."""
    if len(scores) >= n:
        return tuple(scores[:n])
    return scores + (0.0,) * (n - len(scores))


__all__ = ["LLMRepresenter"]
