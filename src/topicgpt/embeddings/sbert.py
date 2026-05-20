"""Sentence-Transformers embedding back-end.

Offline alternative to :class:`OpenAIEmbedder`. Default model is
``BAAI/bge-large-en-v1.5``; the underlying ``SentenceTransformer`` is lazy-loaded
on the first call to keep import-time cheap.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from topicgpt.config import SBERTEmbeddingConfig
from topicgpt.exceptions import EmbeddingError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray
    from sentence_transformers import SentenceTransformer

_LOG = logging.getLogger(__name__)


class SBERTEmbedder:
    """Encode texts with a local Sentence-Transformers model."""

    def __init__(
        self,
        config: SBERTEmbeddingConfig | None = None,
        *,
        model: SentenceTransformer | None = None,
    ) -> None:
        """Build an embedder. Pass ``model=`` to inject a pre-built ST instance (handy in tests)."""
        self.config = config or SBERTEmbeddingConfig()
        self._model: SentenceTransformer | None = model
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        """Embedding dimensionality."""
        if self._dim is None:
            self._dim = self._get_model().get_sentence_embedding_dimension()
            if self._dim is None:  # pragma: no cover - defensive
                raise EmbeddingError("Sentence-Transformers reported a null dimension.")
        return self._dim

    @property
    def model_name(self) -> str:
        """Model id passed to Sentence-Transformers."""
        return self.config.model

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Encode ``texts`` and return a ``(len(texts), dim)`` float32 array."""
        if not texts:
            raise EmbeddingError("Cannot embed an empty sequence.")
        device = None if self.config.device == "auto" else self.config.device
        arr = self._get_model().encode(
            list(texts),
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
            device=device,
        )
        return np.asarray(arr, dtype=np.float32)

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:  # pragma: no cover - guarded by dependency
                raise EmbeddingError("sentence-transformers is required for SBERTEmbedder.") from e
            device = None if self.config.device == "auto" else self.config.device
            _LOG.info("Loading SentenceTransformer model: %s", self.config.model)
            self._model = SentenceTransformer(self.config.model, device=device)
        return self._model


__all__ = ["SBERTEmbedder"]
