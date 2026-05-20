"""Embedding back-ends.

Public surface:
    - :class:`Embedder` protocol
    - :class:`OpenAIEmbedder` (default; ``text-embedding-3-large``)
    - :class:`SBERTEmbedder` (offline alternative)
"""

from __future__ import annotations

from topicgpt.embeddings.base import Embedder
from topicgpt.embeddings.openai import OpenAIEmbedder
from topicgpt.embeddings.sbert import SBERTEmbedder

__all__ = ["Embedder", "OpenAIEmbedder", "SBERTEmbedder"]
