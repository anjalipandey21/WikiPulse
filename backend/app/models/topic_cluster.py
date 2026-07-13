"""Topic cluster data model."""

from pydantic import BaseModel, Field

from .article import Article


class TopicCluster(BaseModel):
    """A group of related Wikipedia articles."""

    id: str
    name: str
    description: str | None = None
    articles: list[Article]
    keywords: list[str]
    total_views: int = Field(ge=0)
    article_count: int = Field(ge=0)
    confidence_score: float | None = Field(default=None, ge=0, le=1)
