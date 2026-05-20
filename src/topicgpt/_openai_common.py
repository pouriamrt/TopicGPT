"""Shared OpenAI client + retry helpers used by every OpenAI-backed module."""

from __future__ import annotations

from typing import TYPE_CHECKING

from openai import APIConnectionError, APIError, OpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from topicgpt.exceptions import ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Callable

    from topicgpt.config import Settings


# Transient OpenAI errors worth retrying.
RETRYABLE_ERRORS: tuple[type[Exception], ...] = (
    APIConnectionError,
    RateLimitError,
    APIError,
)


def build_openai_client(settings: Settings) -> OpenAI:
    """Construct an :class:`openai.OpenAI` client from runtime ``Settings``.

    We disable the SDK's own retries (``max_retries=0``) because we wrap calls
    with our own tenacity loop so the wait policy and retry count come from
    ``Settings``.
    """
    if settings.openai_api_key is None:
        raise ConfigurationError(
            "OPENAI_API_KEY is required. Set it in the environment or inject a client."
        )
    return OpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
        timeout=settings.request_timeout_s,
        max_retries=0,
    )


def with_openai_retry[T](settings: Settings, fn: Callable[[], T]) -> T:
    """Run ``fn`` under the shared exponential-backoff retry policy."""

    @retry(
        retry=retry_if_exception_type(RETRYABLE_ERRORS),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(settings.max_retries + 1),
        reraise=True,
    )
    def _wrapped() -> T:
        return fn()

    return _wrapped()


__all__ = ["RETRYABLE_ERRORS", "build_openai_client", "with_openai_retry"]
