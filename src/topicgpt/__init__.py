"""TopicGPT: modular LLM-driven topic modelling.

Pipeline stages:
    documents → embed → reduce → cluster → vectorize → represent → label

Public API will expand as later phases land. Phase 1 only exports the version.
"""

from __future__ import annotations

__version__ = "1.0.0a0"

__all__ = ["__version__"]
