"""Strict contracts for the process-local analyst-review foundation."""

from collections.abc import Mapping
from datetime import date
from enum import StrEnum
from json import dumps
from typing import Annotated, Literal
from unicodedata import category, normalize
from uuid import UUID, uuid4, uuid5

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)


# Fixed for the lifetime of the wikipulse-review-v1 identifier scheme.
AUDIENCE_REVIEW_NAMESPACE = UUID("76d7067d-f0cb-4c7c-b9c6-c14c4691fdd4")
REVIEW_THREAD_PREFIX = "wikipulse-review-v1:"
REVIEW_VERSION = 1
MAX_PRIVATE_REJECT_NOTE_LENGTH = 240
MIN_ANALYST_EDIT_FEEDBACK_LENGTH = 10
MAX_ANALYST_EDIT_FEEDBACK_LENGTH = 600


CanonicalIdentifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    ),
]
CommandDigest = Annotated[
    str,
    StringConstraints(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    ),
]


class ReviewConflictCode(StrEnum):
    """Stable reasons a review command can be refused."""

    RUN_NOT_FOUND = "run_not_found"
    WRONG_THREAD = "wrong_thread"
    COMMAND_ID_REUSED = "command_id_reused"
    RUN_ID_MISMATCH = "run_id_mismatch"
    REVIEW_ID_MISMATCH = "review_id_mismatch"
    CLUSTER_ID_MISMATCH = "cluster_id_mismatch"
    STALE_VERSION = "stale_version"
    REVIEW_NOT_PENDING = "review_not_pending"
    REVIEW_CURRENTLY_EDITING = "review_currently_editing"
    EDIT_ALREADY_ATTEMPTED = "edit_already_attempted"
    RUN_EXPIRED = "run_expired"
    RUN_TERMINAL = "run_terminal"
    CHECKPOINT_NOT_FOUND = "checkpoint_not_found"


class RejectReasonCode(StrEnum):
    """Allowlisted analyst reasons safe for public outcome codes."""

    SAFETY_CONCERN = "safety_concern"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    MISLEADING_RECOMMENDATION = "misleading_recommendation"
    NOT_COMMERCIALLY_USEFUL = "not_commercially_useful"
    DUPLICATE_AUDIENCE = "duplicate_audience"
    OTHER = "other"


class AnalystEditableField(StrEnum):
    """Allowlisted recommendation groups an analyst may ask to regenerate."""

    AUDIENCE_POSITIONING = "audience_positioning"
    SUPPORTING_EVIDENCE = "supporting_evidence"
    BUYING_POWER = "buying_power"
    BRAND_CATEGORIES = "brand_categories"
    COMMERCIAL_CONFIDENCE = "commercial_confidence"


ANALYST_EDITABLE_FIELD_ORDER = tuple(AnalystEditableField)


EditDropCode = Literal[
    "edit_provider_failed",
    "edit_provider_refused",
    "edit_provider_missing_output",
    "edit_zero_decisions",
    "edit_multiple_decisions",
    "edit_wrong_cluster",
    "edit_provider_skip_not_allowed",
    "edit_unsupported_references",
    "edit_validation_failed",
    "edit_intent_conformance_failed",
    "edit_internal_failure",
]


ReviewRecordStatus = Literal[
    "queued",
    "pending_review",
    "editing",
    "published",
    "rejected",
    "edit_validation_dropped",
    "expired",
]
ReviewRunStatus = Literal[
    "running",
    "pending_review",
    "editing",
    "completed",
    "expired",
    "failed",
]
ReviewFailureCode = Literal[
    "automatic_workflow_failed",
    "review_projection_failed",
]
ReviewTracePhase = Literal["initial", "revision", "review", "edit", "final"]
ReviewTraceEventCode = Literal[
    "generation_requested",
    "decision_received",
    "validation_passed",
    "validation_failed",
    "revision_requested",
    "revision_failed",
    "provider_skipped",
    "decision_dropped",
    "review_requested",
    "analyst_approved",
    "analyst_rejected",
    "analyst_edit_requested",
    "edited_decision_received",
    "edited_decision_validated",
    "edited_audience_published",
    "analyst_edit_dropped",
    "review_expired",
    "audience_published",
]
ReviewTraceOutcome = Literal[
    "queued",
    "pending_review",
    "published",
    "analyst_rejected",
    "editing",
    "edit_validation_dropped",
    "expired",
    "provider_skipped",
    "validation_dropped",
]


class StrictReviewContract(BaseModel):
    """Frozen, strict, JSON-projectable review boundary."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        hide_input_in_errors=True,
    )


class ReviewArticleSnapshot(StrictReviewContract):
    """Immutable Wikipedia evidence safe to checkpoint."""

    title: str
    normalized_title: str
    url: str
    weekly_views: int = Field(ge=0)
    daily_views: tuple["ReviewDailyViewSnapshot", ...]
    summary: str | None = None
    analysis_start_date: date
    analysis_end_date: date


class ReviewDailyViewSnapshot(StrictReviewContract):
    """One deterministically ordered daily pageview entry."""

    day: date
    pageviews: int = Field(ge=0)


class ReviewCompactArticleSnapshot(StrictReviewContract):
    """Immutable copy of one already-validated provider context article."""

    reference_id: CanonicalIdentifier
    title: str
    weekly_views: int = Field(ge=0)
    summary: str | None = None


class ReviewCompactClusterSnapshot(StrictReviewContract):
    """Immutable copy of the authoritative compact provider context."""

    cluster_id: CanonicalIdentifier
    name: str
    keywords: tuple[str, ...]
    total_views: int = Field(ge=0)
    article_count: int = Field(ge=2)
    topic_confidence: float = Field(ge=0, le=1)
    articles: tuple[ReviewCompactArticleSnapshot, ...]


class ReviewResolutionSnapshot(StrictReviewContract):
    """One resolved evidence reference copied without object identity."""

    reference_id: str
    owning_cluster_id: str
    article: ReviewArticleSnapshot


class ReviewReferenceOwnershipSnapshot(StrictReviewContract):
    """One entry from the authoritative global reference-owner mapping."""

    reference_id: str
    cluster_id: str


class ReviewSourceClusterSnapshot(StrictReviewContract):
    """Complete immutable copy of the authoritative source TopicCluster."""

    id: str
    name: str
    description: str | None = None
    articles: tuple[ReviewArticleSnapshot, ...]
    keywords: tuple[str, ...]
    total_views: int = Field(ge=0)
    article_count: int = Field(ge=0)
    confidence_score: float | None = Field(default=None, ge=0, le=1)


class ReviewClusterSnapshot(StrictReviewContract):
    """One prepared cluster with its source and context layers separated."""

    cluster_id: str
    cluster_pageviews: int
    source: ReviewSourceClusterSnapshot
    context: ReviewCompactClusterSnapshot
    evidence_reference_ids: tuple[str, ...]
    resolution: tuple[ReviewResolutionSnapshot, ...]


class ReviewPreparationSnapshot(StrictReviewContract):
    """JSON-safe input from which the automatic workflow is reconstructed."""

    clusters: tuple[ReviewClusterSnapshot, ...]
    total_analyzed_views: int = Field(ge=0)
    reference_owners: tuple[ReviewReferenceOwnershipSnapshot, ...]


class ReviewEvidenceSnapshot(StrictReviewContract):
    """Stable reference ownership for one review candidate."""

    reference_id: CanonicalIdentifier
    article: ReviewArticleSnapshot


class ReviewRecommendationSnapshot(StrictReviewContract):
    """Original deterministically validated recommendation."""

    audience_id: str
    name: str
    description: str
    topic_cluster_ids: tuple[str, ...]
    size_index: float = Field(ge=0, le=100)
    buying_power: Literal["high", "medium", "low"]
    buying_power_reason: str
    brand_categories: tuple[str, ...]
    supporting_article_reference_ids: tuple[CanonicalIdentifier, ...] = Field(
        min_length=2,
        max_length=5,
    )
    supporting_articles: tuple[ReviewArticleSnapshot, ...] = Field(
        min_length=2,
        max_length=5,
    )
    commercial_confidence: float = Field(ge=0, le=1)
    commercial_confidence_reason: str


class ReviewTraceIssue(StrictReviewContract):
    code: str = Field(min_length=1, max_length=128)
    reference_id: str | None = Field(default=None, max_length=128)


class ReviewTraceEvent(StrictReviewContract):
    sequence: int = Field(ge=1)
    phase: ReviewTracePhase
    code: ReviewTraceEventCode
    outcome_code: str | None = Field(default=None, max_length=128)
    issues: tuple[ReviewTraceIssue, ...] = ()


class ReviewDecisionTrace(StrictReviewContract):
    trace_id: str = Field(min_length=1, max_length=160)
    cluster_id: str
    cluster_name: str | None = None
    source_known: bool
    final_outcome: ReviewTraceOutcome
    review_id: str | None = None
    events: tuple[ReviewTraceEvent, ...]


class ReviewCandidateSnapshot(StrictReviewContract):
    """One immutable review item plus its bounded mutable status fields."""

    run_id: str
    review_id: str
    cluster_id: CanonicalIdentifier
    ordinal: int = Field(ge=0)
    version: int = Field(ge=1)
    status: ReviewRecordStatus
    cluster_name: str
    cluster_pageviews: int = Field(ge=0)
    article_count: int = Field(ge=0)
    topic_confidence: float = Field(ge=0, le=1)
    evidence: tuple[ReviewEvidenceSnapshot, ...] = Field(min_length=2)
    recommendation: ReviewRecommendationSnapshot
    edit_attempted: bool = False
    edited_recommendation: ReviewRecommendationSnapshot | None = None
    edit_drop_code: EditDropCode | None = None
    trace: ReviewDecisionTrace
    terminal_command_id: str | None = None
    reject_reason_code: RejectReasonCode | None = None


class ProviderSkipSnapshot(StrictReviewContract):
    cluster_id: CanonicalIdentifier
    cluster_name: str
    reason: str


class ValidationDropSnapshot(StrictReviewContract):
    cluster_id: str
    cluster_name: str | None = None
    source_known: bool
    phase: Literal["initial", "revision"]
    drop_code: str = Field(min_length=1, max_length=128)
    issue_codes: tuple[str, ...]


class ReviewWorkflowMetricsSnapshot(StrictReviewContract):
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
    validation_issue_counts_by_code: tuple["ReviewCodeCountSnapshot", ...]
    drop_counts_by_code: tuple["ReviewCodeCountSnapshot", ...]


class ReviewCodeCountSnapshot(StrictReviewContract):
    """One deterministic immutable metrics count entry."""

    code: str
    count: int = Field(ge=0)


class PendingAudienceReview(StrictReviewContract):
    """Deterministic interrupt payload and future Phase 2 DTO source."""

    run_id: str
    review_id: str
    cluster_id: CanonicalIdentifier
    version: int = Field(ge=1)
    ordinal: int = Field(ge=0)
    queue_size: int = Field(ge=1)
    cluster_name: str
    cluster_pageviews: int = Field(ge=0)
    topic_confidence: float = Field(ge=0, le=1)
    evidence: tuple[ReviewEvidenceSnapshot, ...]
    recommendation: ReviewRecommendationSnapshot
    expires_at: str


class AnalystRejectedOutcome(StrictReviewContract):
    run_id: str
    review_id: str
    cluster_id: CanonicalIdentifier
    reason_code: RejectReasonCode


class ExpiredReviewOutcome(StrictReviewContract):
    run_id: str
    review_id: str
    cluster_id: CanonicalIdentifier


class EditValidationDroppedOutcome(StrictReviewContract):
    """Safe terminal result for one consumed analyst edit allowance."""

    run_id: str
    review_id: str
    cluster_id: CanonicalIdentifier
    drop_code: EditDropCode


class ReviewRunResult(StrictReviewContract):
    """Safe process-local run view; private notes are deliberately absent."""

    run_id: str
    thread_id: str
    created_at: str | None = None
    expires_at: str | None = None
    status: ReviewRunStatus
    is_complete: bool
    failure_code: ReviewFailureCode | None = None
    pending_review: PendingAudienceReview | None = None
    review_candidates: tuple[ReviewCandidateSnapshot, ...] = ()
    published_audiences: tuple[ReviewRecommendationSnapshot, ...] = ()
    rejected_reviews: tuple[AnalystRejectedOutcome, ...] = ()
    expired_reviews: tuple[ExpiredReviewOutcome, ...] = ()
    edit_validation_drops: tuple[EditValidationDroppedOutcome, ...] = ()
    provider_skips: tuple[ProviderSkipSnapshot, ...] = ()
    validation_drops: tuple[ValidationDropSnapshot, ...] = ()
    metrics: ReviewWorkflowMetricsSnapshot
    traces: tuple[ReviewDecisionTrace, ...] = ()


class ReviewCommandBase(StrictReviewContract):
    run_id: str
    review_id: str
    cluster_id: CanonicalIdentifier
    expected_version: int = Field(ge=1)
    command_id: str

    @field_validator("run_id", "review_id", "command_id")
    @classmethod
    def validate_canonical_uuid(cls, value: str) -> str:
        parsed = UUID(value)
        if str(parsed) != value:
            raise ValueError("identifier must be a canonical UUID")
        return value

    @field_validator("run_id", "command_id")
    @classmethod
    def validate_uuid_v4(cls, value: str) -> str:
        if UUID(value).version != 4:
            raise ValueError("identifier must be UUIDv4")
        return value

    @field_validator("review_id")
    @classmethod
    def validate_review_uuid_v5(cls, value: str) -> str:
        if UUID(value).version != 5:
            raise ValueError("review_id must be UUIDv5")
        return value


class ApproveReviewCommand(ReviewCommandBase):
    type: Literal["approve"]


class RejectReviewCommand(ReviewCommandBase):
    type: Literal["reject"]
    reason_code: RejectReasonCode
    private_note: str | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_private_note_input(cls, value: object) -> object:
        """Normalize notes and redact them before any validation failure."""
        if not isinstance(value, dict):
            return value
        raw_note = value.get("private_note")
        reason_code = value.get("reason_code")
        normalized_note: str | None
        error_message: str | None = None
        if raw_note is None:
            normalized_note = None
        elif not isinstance(raw_note, str):
            normalized_note = None
            error_message = "private_note must be a string"
        else:
            normalized_note = None
            if any(category(character).startswith("C") for character in raw_note):
                error_message = (
                    "private_note cannot contain control characters"
                )
            else:
                normalized_note = " ".join(raw_note.split()) or None
                if (
                    normalized_note is not None
                    and len(normalized_note) > MAX_PRIVATE_REJECT_NOTE_LENGTH
                ):
                    error_message = (
                        "private_note exceeds the maximum allowed length"
                    )
        if (
            error_message is None
            and reason_code in {RejectReasonCode.OTHER, RejectReasonCode.OTHER.value}
            and normalized_note is None
        ):
            error_message = (
                "private_note is required when reason_code is other"
            )
        if error_message is not None:
            value["private_note"] = "[redacted]"
            raise ValueError(error_message)
        safe_value = dict(value)
        safe_value["private_note"] = normalized_note
        return safe_value


class EditRecommendationReviewCommand(ReviewCommandBase):
    """One bounded, private analyst-directed recommendation regeneration."""

    type: Literal["edit_recommendation"]
    feedback: str
    fields_to_change: tuple[AnalystEditableField, ...] = Field(
        min_length=1,
        max_length=5,
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_private_feedback(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        safe_value = dict(value)
        raw_feedback = safe_value.get("feedback")
        error_message: str | None = None
        normalized_feedback: str | None = None
        if not isinstance(raw_feedback, str):
            error_message = "feedback must be a string"
        elif any(category(character).startswith("C") for character in raw_feedback):
            error_message = "feedback cannot contain control characters"
        else:
            normalized_feedback = " ".join(normalize("NFC", raw_feedback).split())
            if len(normalized_feedback) < MIN_ANALYST_EDIT_FEEDBACK_LENGTH:
                error_message = "feedback is shorter than the minimum length"
            elif len(normalized_feedback) > MAX_ANALYST_EDIT_FEEDBACK_LENGTH:
                error_message = "feedback exceeds the maximum allowed length"
        if error_message is not None:
            safe_value["feedback"] = "[redacted]"
            value.clear()
            value.update(safe_value)
            raise ValueError(error_message)
        safe_value["feedback"] = normalized_feedback
        if isinstance(safe_value.get("fields_to_change"), list):
            safe_value["fields_to_change"] = tuple(
                safe_value["fields_to_change"]
            )
        return safe_value

    @field_validator("fields_to_change")
    @classmethod
    def canonicalize_editable_fields(
        cls,
        fields_to_change: tuple[AnalystEditableField, ...],
    ) -> tuple[AnalystEditableField, ...]:
        if len(fields_to_change) != len(set(fields_to_change)):
            raise ValueError("fields_to_change must contain unique values")
        selected = set(fields_to_change)
        return tuple(
            field
            for field in ANALYST_EDITABLE_FIELD_ORDER
            if field in selected
        )


class ExpireReviewRunCommand(StrictReviewContract):
    type: Literal["expire_run"]
    run_id: str
    expired_at: str


ReviewCommand = Annotated[
    ApproveReviewCommand | RejectReviewCommand | EditRecommendationReviewCommand,
    Field(discriminator="type"),
]
REVIEW_COMMAND_ADAPTER = TypeAdapter(ReviewCommand)


class ReviewCommandReceipt(StrictReviewContract):
    run_id: str
    review_id: str
    cluster_id: CanonicalIdentifier
    command_id: str
    command_type: Literal["approve", "reject", "edit_recommendation"]
    accepted: Literal[True] = True
    idempotent_replay: bool = False
    resulting_status: Literal[
        "published",
        "rejected",
        "edit_validation_dropped",
    ]
    run_status: ReviewRunStatus


class AppliedReviewCommandSnapshot(StrictReviewContract):
    """Safe checkpoint proof for the latest committed analyst command."""

    command_id: str
    command_digest: CommandDigest
    command_type: Literal["approve", "reject", "edit_recommendation"]
    review_id: str
    cluster_id: CanonicalIdentifier
    review_version: int = Field(ge=1)
    resulting_status: Literal[
        "published",
        "rejected",
        "edit_validation_dropped",
    ]


class ActiveAnalystEditSnapshot(StrictReviewContract):
    """Private checkpointed edit intent required for safe recovery."""

    command_id: str
    command_digest: CommandDigest
    run_id: str
    review_id: str
    cluster_id: CanonicalIdentifier
    accepted_version: int = Field(ge=1)
    resulting_version: int = Field(ge=2)
    feedback: str
    fields_to_change: tuple[AnalystEditableField, ...]


def new_run_id() -> str:
    """Return a canonical server-generated UUIDv4 run identifier."""
    return str(uuid4())


def new_command_id() -> str:
    """Return a canonical UUIDv4 command identifier."""
    return str(uuid4())


def review_id_for(run_id: str, cluster_id: str) -> str:
    """Derive a stable review UUID from immutable run and cluster identity."""
    parsed_run_id = UUID(run_id)
    if parsed_run_id.version != 4 or str(parsed_run_id) != run_id:
        raise ValueError("run_id must be a canonical UUIDv4")
    return str(uuid5(AUDIENCE_REVIEW_NAMESPACE, f"{run_id}:{cluster_id}"))


def review_thread_id(run_id: str) -> str:
    """Map one run to its sole LangGraph checkpoint thread."""
    return f"{REVIEW_THREAD_PREFIX}{run_id}"


def parse_review_command(
    value: (
        ApproveReviewCommand
        | RejectReviewCommand
        | EditRecommendationReviewCommand
        | Mapping[str, object]
    ),
) -> ApproveReviewCommand | RejectReviewCommand | EditRecommendationReviewCommand:
    """Strictly parse the sole Phase 1A command trust boundary."""
    if isinstance(
        value,
        (
            ApproveReviewCommand,
            RejectReviewCommand,
            EditRecommendationReviewCommand,
        ),
    ):
        raw_value = value.model_dump(mode="json")
    elif isinstance(value, Mapping):
        raw_value = dict(value)
    else:
        raise TypeError("review command must be a mapping or command model")
    try:
        serialized = dumps(
            raw_value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        raise TypeError("review command must contain JSON values") from None
    if raw_value.get("type") == "reject":
        # Validate the private-input model outside the union wrapper so its
        # redacted input, rather than the original union input, is retained in
        # structured ValidationError data.
        reject_value = dict(raw_value)
        try:
            reject_value["reason_code"] = RejectReasonCode(
                reject_value.get("reason_code")
            )
        except (TypeError, ValueError):
            pass
        reject = RejectReviewCommand.model_validate(reject_value)
        return REVIEW_COMMAND_ADAPTER.validate_python(reject)
    if raw_value.get("type") == "edit_recommendation":
        edit_value = dict(raw_value)
        raw_fields = edit_value.get("fields_to_change")
        if isinstance(raw_fields, (list, tuple)):
            try:
                edit_value["fields_to_change"] = tuple(
                    AnalystEditableField(field) for field in raw_fields
                )
            except (TypeError, ValueError):
                edit_value["fields_to_change"] = tuple(raw_fields)
        try:
            edit = EditRecommendationReviewCommand.model_validate(edit_value)
        except ValidationError as exc:
            raise ValidationError.from_exception_data(
                "EditRecommendationReviewCommand",
                exc.errors(
                    include_url=False,
                    include_input=False,
                ),
                hide_input=True,
            ) from None
        return REVIEW_COMMAND_ADAPTER.validate_python(edit)
    safe_union_value = dict(raw_value)
    for private_field in ("private_note", "feedback"):
        if private_field in safe_union_value:
            safe_union_value[private_field] = "[redacted]"
    serialized = dumps(
        safe_union_value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return REVIEW_COMMAND_ADAPTER.validate_json(serialized)
