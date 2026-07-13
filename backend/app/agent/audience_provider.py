"""Provider-neutral interface for typed audience-generation responses."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from ..models.audience_generation import (
    AudienceGenerationResponse,
    CompactClusterContext,
)


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


class AudienceGenerationProvider(Protocol):
    """Injected async provider for cluster-level structured generation."""

    async def generate(
        self,
        cluster_contexts: Sequence[CompactClusterContext],
    ) -> AudienceProviderResult:
        """Return typed decisions and metadata for the supplied contexts."""
        ...
