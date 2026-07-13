"""Article data model."""

from datetime import date
from typing import Self

from pydantic import BaseModel, Field, NonNegativeInt, model_validator


class Article(BaseModel):
    """A Wikipedia article and its views during an analysis period."""

    title: str
    normalized_title: str
    url: str
    weekly_views: int = Field(ge=0)
    daily_views: dict[date, NonNegativeInt]
    summary: str | None = None
    analysis_start_date: date
    analysis_end_date: date

    @model_validator(mode="after")
    def validate_analysis_date_order(self) -> Self:
        """Ensure the analysis period does not end before it starts."""
        if self.analysis_start_date > self.analysis_end_date:
            raise ValueError(
                "analysis_start_date must be on or before analysis_end_date"
            )
        return self
