"""Provider-neutral interface for typed audience-generation responses."""

from collections.abc import Sequence
from typing import Protocol

from ..models.audience_generation import (
    AudienceGenerationResponse,
    CompactClusterContext,
)


class AudienceGenerationProvider(Protocol):
    """Injected async provider for cluster-level structured generation."""

    async def generate(
        self,
        cluster_contexts: Sequence[CompactClusterContext],
    ) -> AudienceGenerationResponse:
        """Return typed decisions for the supplied compact cluster contexts."""
        ...
