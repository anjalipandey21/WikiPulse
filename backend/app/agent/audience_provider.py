"""Provider-neutral interface for typed audience-generation responses."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from ..models.audience_generation import (
    AudienceDecision,
    AudienceGenerationResponse,
    CompactClusterContext,
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
