"""Pydantic schemas for LLM structured outputs."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TopicLabel(BaseModel):
    """Structured topic description returned by the LLM."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, max_length=80, description="Short topic label.")
    description: str = Field(
        min_length=1,
        max_length=400,
        description="One- or two-sentence prose description.",
    )
    keywords: list[str] = Field(
        min_length=3,
        max_length=12,
        description="Most representative keywords for the topic.",
    )


__all__ = ["TopicLabel"]
