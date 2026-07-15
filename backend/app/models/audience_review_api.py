"""Public, privacy-safe contracts for the analyst-review API."""

from datetime import date, datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .audience_review import (
    AnalystEditableField,
    EditDropCode,
    RejectReasonCode,
    ReviewFailureCode,
    ReviewTraceEventCode,
    ReviewTraceOutcome,
    ReviewTracePhase,
)


class StrictReviewApiModel(BaseModel):
    """Strict immutable public boundary with no implicit extra data."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        hide_input_in_errors=True,
    )


class AudienceReviewStartRequest(StrictReviewApiModel):
    run_id: str
    ttl_seconds: int | None = Field(default=None, gt=0)

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        parsed = UUID(value)
        if parsed.version != 4 or str(parsed) != value:
            raise ValueError("run_id must be a canonical UUIDv4")
        return value


class ApproveReviewApiCommand(StrictReviewApiModel):
    type: Literal["approve"]
    command_id: str
    review_id: str
    cluster_id: str
    expected_version: int = Field(ge=1)


class RejectReviewApiCommand(ApproveReviewApiCommand):
    type: Literal["reject"]
    reason_code: RejectReasonCode
    private_note: str | None = Field(
        default=None,
        json_schema_extra={"writeOnly": True},
    )


class EditRecommendationReviewApiCommand(ApproveReviewApiCommand):
    type: Literal["edit_recommendation"]
    feedback: str = Field(json_schema_extra={"writeOnly": True})
    fields_to_change: tuple[AnalystEditableField, ...]


AudienceReviewApiCommand = Annotated[
    ApproveReviewApiCommand
    | RejectReviewApiCommand
    | EditRecommendationReviewApiCommand,
    Field(discriminator="type"),
]


class ReviewDailyViewResponse(StrictReviewApiModel):
    day: date
    pageviews: int = Field(ge=0)


class ReviewArticleResponse(StrictReviewApiModel):
    title: str
    normalized_title: str
    url: str
    weekly_views: int = Field(ge=0)
    daily_views: tuple[ReviewDailyViewResponse, ...]
    summary: str | None
    analysis_start_date: date
    analysis_end_date: date


class ReviewEvidenceResponse(StrictReviewApiModel):
    reference_id: str
    article: ReviewArticleResponse


class ReviewRecommendationResponse(StrictReviewApiModel):
    audience_id: str
    name: str
    description: str
    topic_cluster_ids: tuple[str, ...]
    size_index: float = Field(ge=0, le=100)
    buying_power: Literal["high", "medium", "low"]
    buying_power_reason: str
    brand_categories: tuple[str, ...]
    supporting_article_reference_ids: tuple[str, ...]
    supporting_articles: tuple[ReviewArticleResponse, ...]
    commercial_confidence: float = Field(ge=0, le=1)
    commercial_confidence_reason: str


class PendingReviewResponse(StrictReviewApiModel):
    status: Literal["pending_review"] = "pending_review"
    review_id: str
    cluster_id: str
    expected_version: int = Field(ge=1)
    position: int = Field(ge=1)
    total_reviews: int = Field(ge=1)
    cluster_name: str
    cluster_pageviews: int = Field(ge=0)
    article_count: int = Field(ge=0)
    size_index: float = Field(ge=0, le=100)
    topic_confidence: float = Field(ge=0, le=1)
    original_recommendation: ReviewRecommendationResponse
    evidence: tuple[ReviewEvidenceResponse, ...]
    edit_available: bool


class EditingReviewResponse(StrictReviewApiModel):
    status: Literal["editing"] = "editing"
    review_id: str
    cluster_id: str
    position: int = Field(ge=1)
    total_reviews: int = Field(ge=1)
    cluster_name: str
    cluster_pageviews: int = Field(ge=0)
    article_count: int = Field(ge=0)
    size_index: float = Field(ge=0, le=100)
    topic_confidence: float = Field(ge=0, le=1)
    original_recommendation: ReviewRecommendationResponse
    evidence: tuple[ReviewEvidenceResponse, ...]
    edit_available: Literal[False] = False


CurrentReviewResponse = Annotated[
    PendingReviewResponse | EditingReviewResponse,
    Field(discriminator="status"),
]


class AudienceReviewProgressResponse(StrictReviewApiModel):
    total_reviews: int = Field(ge=0)
    completed_reviews: int = Field(ge=0)
    queued_reviews: int = Field(ge=0)
    current_position: int | None = Field(default=None, ge=1)


class PublishedAudienceResponse(StrictReviewApiModel):
    review_id: str
    cluster_id: str
    trace_id: str
    publication_source: Literal["original", "analyst_edit"]
    audience: ReviewRecommendationResponse


class RejectedReviewResponse(StrictReviewApiModel):
    review_id: str
    cluster_id: str
    cluster_name: str
    reason_code: RejectReasonCode


class EditValidationDropResponse(StrictReviewApiModel):
    review_id: str
    cluster_id: str
    cluster_name: str
    drop_code: EditDropCode


class ExpiredReviewResponse(StrictReviewApiModel):
    review_id: str
    cluster_id: str
    cluster_name: str


class ProviderSkipReviewResponse(StrictReviewApiModel):
    trace_id: str
    cluster_id: str
    cluster_name: str
    reason: str


class ValidationDropReviewResponse(StrictReviewApiModel):
    trace_id: str
    cluster_id: str
    cluster_name: str | None
    source_known: bool
    phase: Literal["initial", "revision"]
    drop_code: str
    issue_codes: tuple[str, ...]


class ReviewJourneyIssueResponse(StrictReviewApiModel):
    code: str
    reference_id: str | None


class ReviewJourneyEventResponse(StrictReviewApiModel):
    sequence: int = Field(ge=1)
    phase: ReviewTracePhase
    code: ReviewTraceEventCode
    outcome_code: str | None
    issues: tuple[ReviewJourneyIssueResponse, ...]


class ReviewJourneyResponse(StrictReviewApiModel):
    trace_id: str
    cluster_id: str
    cluster_name: str | None
    source_known: bool
    final_outcome: ReviewTraceOutcome
    review_id: str | None
    events: tuple[ReviewJourneyEventResponse, ...]


class ReviewMetricCodeCountResponse(StrictReviewApiModel):
    code: str
    count: int = Field(ge=0)


class ReviewWorkflowMetricsResponse(StrictReviewApiModel):
    initial_decision_count: int = Field(ge=0)
    initial_valid_decision_count: int = Field(ge=0)
    initial_invalid_report_count: int = Field(ge=0)
    revision_count: int = Field(ge=0, le=1)
    revision_requested_cluster_count: int = Field(ge=0)
    revision_decision_count: int = Field(ge=0)
    revision_valid_decision_count: int = Field(ge=0)
    final_valid_decision_count: int = Field(ge=0)
    final_segment_count: int = Field(ge=0)
    final_provider_skip_count: int = Field(ge=0)
    dropped_source_cluster_count: int = Field(ge=0)
    dropped_unmatched_decision_count: int = Field(ge=0)
    provider_call_count: int = Field(ge=0)
    provider_input_tokens: int = Field(ge=0)
    provider_output_tokens: int = Field(ge=0)
    provider_total_tokens: int = Field(ge=0)
    provider_elapsed_seconds: float = Field(ge=0)
    validation_issue_count: int = Field(ge=0)
    validation_issue_counts_by_code: tuple[ReviewMetricCodeCountResponse, ...]
    drop_counts_by_code: tuple[ReviewMetricCodeCountResponse, ...]


class AudienceReviewRunResponse(StrictReviewApiModel):
    run_id: str
    status: Literal[
        "running",
        "pending_review",
        "editing",
        "completed",
        "expired",
        "failed",
    ]
    is_complete: bool
    created_at: datetime
    expires_at: datetime
    progress: AudienceReviewProgressResponse
    current_review: CurrentReviewResponse | None
    published_audiences: tuple[PublishedAudienceResponse, ...]
    rejected_reviews: tuple[RejectedReviewResponse, ...]
    edit_validation_drops: tuple[EditValidationDropResponse, ...]
    expired_reviews: tuple[ExpiredReviewResponse, ...]
    provider_skips: tuple[ProviderSkipReviewResponse, ...]
    validation_drops: tuple[ValidationDropReviewResponse, ...]
    journey: tuple[ReviewJourneyResponse, ...]
    automatic_workflow_metrics: ReviewWorkflowMetricsResponse
    failure_code: ReviewFailureCode | None


class PublicReviewCommandReceipt(StrictReviewApiModel):
    command_id: str
    type: Literal["approve", "reject", "edit_recommendation"]
    review_id: str
    cluster_id: str
    accepted: Literal[True]
    idempotent_replay: bool
    resulting_status: Literal[
        "published",
        "rejected",
        "edit_validation_dropped",
    ]
    run_status: Literal[
        "running",
        "pending_review",
        "editing",
        "completed",
        "expired",
        "failed",
    ]


class AudienceReviewCommandResponse(StrictReviewApiModel):
    receipt: PublicReviewCommandReceipt
    run: AudienceReviewRunResponse
