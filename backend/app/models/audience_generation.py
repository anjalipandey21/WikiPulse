"""Strict internal contracts for cluster-level audience generation."""

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
)


ContractIdentifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    ),
]
TopicName = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=120),
]
AudienceName = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=3, max_length=80),
]
DescriptionText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=20, max_length=500),
]
ReasonText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=20, max_length=300),
]
ArticleTitle = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=250),
]
ArticleSummary = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=1_000),
]
KeywordText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=80),
]
BrandCategory = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=2, max_length=60),
]


class StrictAudienceContract(BaseModel):
    """Base configuration shared by internal structured-output contracts."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
        frozen=True,
    )


class CompactArticleContext(StrictAudienceContract):
    """Bounded article evidence supplied to an audience-generation provider."""

    reference_id: ContractIdentifier
    title: ArticleTitle
    weekly_views: int = Field(ge=0)
    summary: ArticleSummary | None = None


class CompactClusterContext(StrictAudienceContract):
    """Compact, traceable context for one commercially eligible topic."""

    cluster_id: ContractIdentifier
    name: TopicName
    keywords: list[KeywordText] = Field(default_factory=list, max_length=5)
    total_views: int = Field(ge=0)
    article_count: int = Field(ge=2)
    topic_confidence: float = Field(ge=0, le=1)
    articles: list[CompactArticleContext] = Field(min_length=2, max_length=5)

    @field_validator("articles")
    @classmethod
    def reject_duplicate_article_references(
        cls,
        articles: list[CompactArticleContext],
    ) -> list[CompactArticleContext]:
        reference_ids = [article.reference_id for article in articles]
        if len(reference_ids) != len(set(reference_ids)):
            raise ValueError("article context reference IDs must be unique")
        return articles


class CreateAudienceDecision(StrictAudienceContract):
    """Structured decision to create one audience from one source cluster."""

    decision: Literal["create_audience"]
    cluster_id: ContractIdentifier
    name: AudienceName
    description: DescriptionText
    supporting_article_reference_ids: list[ContractIdentifier] = Field(
        min_length=2,
        max_length=5,
    )
    buying_power: Literal["high", "medium", "low"]
    buying_power_reason: ReasonText
    brand_categories: list[BrandCategory] = Field(min_length=1, max_length=5)
    commercial_confidence: float = Field(ge=0, le=1)
    commercial_confidence_reason: ReasonText

    @field_validator("supporting_article_reference_ids")
    @classmethod
    def reject_duplicate_supporting_references(
        cls,
        reference_ids: list[str],
    ) -> list[str]:
        if len(reference_ids) != len(set(reference_ids)):
            raise ValueError("supporting article reference IDs must be unique")
        return reference_ids

    @field_validator("brand_categories")
    @classmethod
    def reject_duplicate_brand_categories(
        cls,
        categories: list[str],
    ) -> list[str]:
        normalized_categories = [category.casefold() for category in categories]
        if len(normalized_categories) != len(set(normalized_categories)):
            raise ValueError("brand categories must be unique")
        return categories


class SkipClusterDecision(StrictAudienceContract):
    """Structured decision not to create an audience for a source cluster."""

    decision: Literal["skip_cluster"]
    cluster_id: ContractIdentifier
    reason: ReasonText


AudienceDecision = Annotated[
    CreateAudienceDecision | SkipClusterDecision,
    Field(discriminator="decision"),
]


class AudienceGenerationResponse(StrictAudienceContract):
    """Typed provider response containing one or more cluster decisions."""

    decisions: list[AudienceDecision] = Field(min_length=1, max_length=6)
