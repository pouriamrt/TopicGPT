"""OpenAI embedding back-end.

Uses ``text-embedding-3-small`` by default. Batches requests, retries
transient failures, and optionally caches vectors on disk so re-runs over the
same corpus are free.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
import tiktoken

from topicgpt._openai_common import RETRYABLE_ERRORS, build_openai_client, with_openai_retry
from topicgpt.config import OpenAIEmbeddingConfig, Settings
from topicgpt.embeddings._cache import EmbeddingCache
from topicgpt.exceptions import EmbeddingError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from numpy.typing import NDArray
    from openai import OpenAI

_LOG = logging.getLogger(__name__)

# OpenAI hard limit per /v1/embeddings request.
_MAX_BATCH_PER_REQUEST = 2048


class _EmbeddingDatum(Protocol):
    """Structural shape of one row in an OpenAI embeddings response."""

    index: int
    embedding: list[float]


class _EmbeddingResponse(Protocol):
    """Structural shape of an OpenAI embeddings response payload."""

    data: list[_EmbeddingDatum]


class OpenAIEmbedder:
    """Encode texts via the OpenAI embeddings API."""

    def __init__(
        self,
        config: OpenAIEmbeddingConfig | None = None,
        *,
        settings: Settings | None = None,
        client: OpenAI | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        """Build an embedder.

        Args:
            config: Embedding-model parameters. Defaults to :class:`OpenAIEmbeddingConfig`().
            settings: Runtime settings (API key, retries). Inferred from env if omitted.
            client: Pre-built :class:`openai.OpenAI` client (handy for tests / proxies).
            cache_dir: Where to persist vectors. ``None`` disables caching.
        """
        self.config = config or OpenAIEmbeddingConfig()
        self.settings = settings or Settings()
        self._client = client or build_openai_client(self.settings)
        self._cache = (
            EmbeddingCache(cache_dir, model=self.config.model, dim=self.config.dimensions)
            if cache_dir is not None
            else None
        )
        self._dim: int | None = self.config.dimensions
        try:
            self._tokenizer = tiktoken.encoding_for_model(self.config.model)
        except KeyError:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")

    @property
    def dim(self) -> int:
        """Embedding dimensionality. Lazily resolved on first embed call."""
        if self._dim is None:
            raise EmbeddingError(
                "Embedding dimensionality is unknown until the first call to .embed()."
            )
        return self._dim

    @property
    def model_name(self) -> str:
        """Underlying OpenAI model id."""
        return self.config.model

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Encode ``texts`` and return a ``(len(texts), dim)`` float32 array."""
        if not texts:
            raise EmbeddingError("Cannot embed an empty sequence.")

        prepared = [self._prepare(t) for t in texts]
        cached, miss_idx, miss_texts = self._lookup_cache(prepared)

        if miss_texts:
            new_vecs = self._embed_batched(miss_texts)
            for idx, vec, text in zip(miss_idx, new_vecs, miss_texts, strict=True):
                cached[idx] = vec
                if self._cache is not None:
                    self._cache.put(text, vec)

        stacked = np.vstack(cast("list[NDArray[np.float32]]", cached))
        self._dim = stacked.shape[1]
        return stacked.astype(np.float32, copy=False)

    def _prepare(self, text: str) -> str:
        """Truncate ``text`` to ``max_tokens_per_input`` tokens."""
        if not text:
            return " "
        tokens = self._tokenizer.encode(text)
        if len(tokens) <= self.config.max_tokens_per_input:
            return text
        truncated = self._tokenizer.decode(tokens[: self.config.max_tokens_per_input])
        _LOG.debug(
            "Truncated input from %d to %d tokens",
            len(tokens),
            self.config.max_tokens_per_input,
        )
        return truncated

    def _lookup_cache(
        self,
        texts: Sequence[str],
    ) -> tuple[list[NDArray[np.float32] | None], list[int], list[str]]:
        slots: list[NDArray[np.float32] | None] = [None] * len(texts)
        if self._cache is None:
            return slots, list(range(len(texts))), list(texts)

        miss_idx: list[int] = []
        miss_texts: list[str] = []
        for i, text in enumerate(texts):
            hit = self._cache.get(text)
            if hit is not None:
                slots[i] = hit
            else:
                miss_idx.append(i)
                miss_texts.append(text)
        return slots, miss_idx, miss_texts

    def _embed_batched(self, texts: Sequence[str]) -> list[NDArray[np.float32]]:
        out: list[NDArray[np.float32]] = []
        batch_size = min(self.config.batch_size, _MAX_BATCH_PER_REQUEST)
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            out.extend(self._embed_one_batch(list(chunk)))
        return out

    def _embed_one_batch(self, batch: list[str]) -> list[NDArray[np.float32]]:
        try:
            response = self._call_api(batch)
        except RETRYABLE_ERRORS as e:
            raise EmbeddingError(f"OpenAI embeddings call failed: {e}") from e

        # OpenAI returns items in input order regardless of payload size.
        data = sorted(response.data, key=lambda d: d.index)
        return [np.asarray(d.embedding, dtype=np.float32) for d in data]

    def _call_api(self, batch: list[str]) -> _EmbeddingResponse:
        kwargs: dict[str, object] = {"model": self.config.model, "input": batch}
        if self.config.dimensions is not None:
            kwargs["dimensions"] = self.config.dimensions

        def _do() -> _EmbeddingResponse:
            return cast(
                "_EmbeddingResponse",
                self._client.embeddings.create(**kwargs),  # type: ignore[arg-type]
            )

        return with_openai_retry(self.settings, _do)


__all__ = ["OpenAIEmbedder"]
