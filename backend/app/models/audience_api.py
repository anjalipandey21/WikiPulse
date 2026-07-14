"""Public Pydantic contracts for the audience-analysis API."""

from datetime import date
from typing import Literal

from pydantic import BaseModel


class ArticleResponse(BaseModel):
    """Dashboard-safe Wikipedia article data."""

    title: str
    normalized_title: str
    url: str
    weekly_views: int
    daily_views: dict[date, int]
    summary: str | None
    analysis_start_date: date
    analysis_end_date: date


class TopicClusterResponse(BaseModel):
    """Public topic cluster with traceable articles."""

    id: str
    name: str
    description: str | None
    articles: list[ArticleResponse]
    keywords: list[str]
    total_views: int
    article_count: int
    confidence_score: float | None


class AudienceSegmentResponse(BaseModel):
    """Public commercial audience segment."""

    id: str
    name: str
    description: str
    topic_cluster_ids: list[str]
    size_index: float
    buying_power: Literal["high", "medium", "low"]
    buying_power_reason: str
    brand_categories: list[str]
    supporting_articles: list[ArticleResponse]
    commercial_confidence: float
    commercial_confidence_reason: str


class RejectedArticleResponse(BaseModel):
    """Noise-filtered article and deterministic reason."""

    article: ArticleResponse
    reason: str


class CommercialSkippedClusterResponse(BaseModel):
    """Topic excluded before provider generation."""

    cluster_id: str
    cluster_name: str
    reason: str


class ProviderSkippedClusterResponse(BaseModel):
    """Valid provider decision not to create an audience."""

    cluster_id: str
    cluster_name: str
    reason: str


class AudienceDecisionIssueResponse(BaseModel):
    """Stable validation issue safe for dashboard diagnostics."""

    code: str
    reference_id: str | None


class DroppedAudienceDecisionResponse(BaseModel):
    """Unresolved or unmatched decision excluded from publication."""

    cluster_id: str
    source_known: bool
    phase: Literal["initial", "revision"]
    drop_code: str
    issues: list[AudienceDecisionIssueResponse]


class TopicAnalysisMetricsResponse(BaseModel):
    """Deterministic topic-analysis metrics."""

    fetched_article_count: int
    rejected_article_count: int
    eligible_article_count: int
    top_n_omitted_article_count: int
    selected_article_count: int
    summary_available_article_count: int
    summary_missing_article_count: int
    topic_cluster_count: int
    clustered_article_count: int
    unclustered_article_count: int
    selected_pageviews: int


class AudienceFunnelMetricsResponse(BaseModel):
    """Deterministic cross-stage audience funnel metrics."""

    topic_cluster_count: int
    commercial_eligible_cluster_count: int
    commercial_skipped_cluster_count: int
    prepared_cluster_count: int
    final_segment_count: int
    provider_skipped_cluster_count: int
    validation_dropped_source_cluster_count: int
    unmatched_provider_output_count: int
    commercial_eligible_pageviews: int
    represented_audience_pageviews: int
    commercial_skip_counts_by_reason: dict[str, int]


class AudienceWorkflowMetricsResponse(BaseModel):
    """Safe validation, revision, and aggregate provider-usage metrics."""

    initial_decision_count: int
    initial_valid_decision_count: int
    initial_invalid_report_count: int
    revision_count: int
    revision_requested_cluster_count: int
    revision_decision_count: int
    revision_valid_decision_count: int
    final_valid_decision_count: int
    final_segment_count: int
    final_provider_skip_count: int
    dropped_source_cluster_count: int
    dropped_unmatched_decision_count: int
    provider_call_count: int
    provider_input_tokens: int
    provider_output_tokens: int
    provider_total_tokens: int
    provider_elapsed_seconds: float
    validation_issue_count: int
    validation_issue_counts_by_code: dict[str, int]
    drop_counts_by_code: dict[str, int]


class AudienceAnalysisMetricsResponse(BaseModel):
    """Public metrics grouped by pipeline stage."""

    topic_analysis: TopicAnalysisMetricsResponse
    audience_funnel: AudienceFunnelMetricsResponse
    workflow: AudienceWorkflowMetricsResponse


class AudienceAnalysisResponse(BaseModel):
    """Dashboard-ready topic landscape and audience portfolio."""

    topics: list[TopicClusterResponse]
    audience_segments: list[AudienceSegmentResponse]
    unclustered_articles: list[ArticleResponse]
    rejected_articles: list[RejectedArticleResponse]
    commercial_skips: list[CommercialSkippedClusterResponse]
    provider_skips: list[ProviderSkippedClusterResponse]
    validation_drops: list[DroppedAudienceDecisionResponse]
    is_publishable: bool
    metrics: AudienceAnalysisMetricsResponse


class ApiErrorDetailResponse(BaseModel):
    """Stable public error code and safe message."""

    code: str
    message: str


class ApiErrorResponse(BaseModel):
    """Consistent error envelope for API failures."""

    error: ApiErrorDetailResponse
