"""Provider-neutral interface for typed audience-generation responses."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..models.audience_generation import (
    AudienceDecision,
    AudienceGenerationResponse,
    CompactClusterContext,
    CreateAudienceDecision,
)
from ..models.audience_review import (
    ANALYST_EDITABLE_FIELD_ORDER,
    MAX_ANALYST_EDIT_FEEDBACK_LENGTH,
    MIN_ANALYST_EDIT_FEEDBACK_LENGTH,
    AnalystEditableField,
)


class AudienceProviderError(RuntimeError):
    """Safe public error for any audience-provider failure."""


@dataclass(frozen=True, slots=True)
class AudienceTokenUsage:
    """Actual token counts reported by an audience-generation provider."""

    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass(frozen=True, slots=True)
class AudienceProviderResult:
    """Typed audience decisions and provider request metadata."""

    response: AudienceGenerationResponse
    model: str
    response_id: str
    elapsed_seconds: float
    usage: AudienceTokenUsage


@dataclass(frozen=True, slots=True)
class AudienceRevisionIssue:
    """One deterministic validation issue supplied for revision."""

    code: str
    reference_id: str | None = None


@dataclass(frozen=True, slots=True)
class AudienceRevisionRequest:
    """Compact source context and exact issues for one revised decision."""

    context: CompactClusterContext
    previous_decisions: tuple[AudienceDecision, ...]
    validation_issues: tuple[AudienceRevisionIssue, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "previous_decisions",
            tuple(self.previous_decisions),
        )
        object.__setattr__(
            self,
            "validation_issues",
            tuple(self.validation_issues),
        )


class AnalystEditProviderRequest(BaseModel):
    """Strict single-cluster input for one analyst-directed regeneration."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        hide_input_in_errors=True,
    )

    expected_cluster_id: str
    context: CompactClusterContext
    original_decision: CreateAudienceDecision
    feedback: str
    fields_to_change: tuple[AnalystEditableField, ...] = Field(
        min_length=1,
        max_length=5,
    )

    @field_validator("feedback")
    @classmethod
    def require_normalized_feedback(cls, feedback: str) -> str:
        if not (
            MIN_ANALYST_EDIT_FEEDBACK_LENGTH
            <= len(feedback)
            <= MAX_ANALYST_EDIT_FEEDBACK_LENGTH
        ):
            raise ValueError("feedback length is outside the allowed range")
        if feedback != " ".join(feedback.split()):
            raise ValueError("feedback must already be normalized")
        return feedback

    @field_validator("fields_to_change")
    @classmethod
    def require_canonical_field_order(
        cls,
        fields_to_change: tuple[AnalystEditableField, ...],
    ) -> tuple[AnalystEditableField, ...]:
        if len(fields_to_change) != len(set(fields_to_change)):
            raise ValueError("fields_to_change must be unique")
        selected = set(fields_to_change)
        canonical = tuple(
            field for field in ANALYST_EDITABLE_FIELD_ORDER if field in selected
        )
        if fields_to_change != canonical:
            raise ValueError("fields_to_change must use canonical order")
        return fields_to_change

    @model_validator(mode="after")
    def require_original_cluster_identity(self) -> "AnalystEditProviderRequest":
        if self.original_decision.cluster_id != self.expected_cluster_id:
            raise ValueError("original decision cluster must match expected cluster")
        return self


class AnalystEditGenerationResponse(BaseModel):
    """Bounded output that retains invalid cardinality for classification."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    decisions: list[AudienceDecision] = Field(default_factory=list, max_length=6)


@dataclass(frozen=True, slots=True)
class AnalystEditProviderResult:
    """Safe edit result without model or response-envelope metadata."""

    status: Literal[
        "completed",
        "refused",
        "missing_output",
        "provider_failed",
    ]
    response: AnalystEditGenerationResponse | None
    elapsed_seconds: float
    usage: AudienceTokenUsage | None

    def __post_init__(self) -> None:
        if self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be non-negative")
        if self.status == "completed":
            if self.response is None or self.usage is None:
                raise ValueError("completed edit result requires response and usage")
        elif self.response is not None:
            raise ValueError("non-completed edit result cannot contain a response")


class AudienceGenerationProvider(Protocol):
    """Injected async provider for cluster-level structured generation."""

    async def generate(
        self,
        cluster_contexts: Sequence[CompactClusterContext],
    ) -> AudienceProviderResult:
        """Return typed decisions and metadata for the supplied contexts."""
        ...

    async def revise(
        self,
        revision_requests: Sequence[AudienceRevisionRequest],
    ) -> AudienceProviderResult:
        """Return replacements only for the supplied invalid decisions."""
        ...

    async def regenerate_from_analyst_edit(
        self,
        request: AnalystEditProviderRequest,
    ) -> AnalystEditProviderResult:
        """Return exactly one bounded analyst-directed regeneration result."""
        ...
