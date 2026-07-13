"""Audience segment data model."""

from typing import Literal

from pydantic import BaseModel, Field

from .article import Article


class AudienceSegment(BaseModel):
    """A commercially useful audience profile supported by trend data."""

    id: str
    name: str
    description: str
    topic_cluster_ids: list[str]
    size_index: float = Field(ge=0, le=100)
    buying_power: Literal["high", "medium", "low"]
    buying_power_reason: str
    brand_categories: list[str]
    supporting_articles: list[Article]
    commercial_confidence: float = Field(ge=0, le=1)
    commercial_confidence_reason: str
