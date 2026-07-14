"""Safe public classification for audience-analysis failures."""

from dataclasses import dataclass

from fastapi.exceptions import RequestValidationError

from ..agent.audience_finalization import AudienceSourceIntegrityError
from ..agent.audience_provider import AudienceProviderError
from ..agent.audience_workflow import AudienceWorkflowInvariantError
from ..audience_analysis import AudienceAnalysisInvariantError
from ..services.wikimedia_client import WikimediaPageviewsError
from ..services.wikipedia_summary_client import WikipediaSummaryError
from ..topic_analysis import TopicAnalysisInvariantError


@dataclass(frozen=True, slots=True)
class SafeAnalysisError:
    """Stable status and public message without an exception detail."""

    status_code: int
    code: str
    message: str


def classify_analysis_exception(exc: Exception) -> SafeAnalysisError:
    """Translate an internal failure into the existing safe API contract."""
    if isinstance(exc, WikimediaPageviewsError):
        return SafeAnalysisError(
            502,
            "wikimedia_pageviews_unavailable",
            "Wikipedia pageview data is temporarily unavailable.",
        )
    if isinstance(exc, WikipediaSummaryError):
        return SafeAnalysisError(
            502,
            "wikipedia_summaries_unavailable",
            "Wikipedia summaries are temporarily unavailable.",
        )
    if isinstance(exc, AudienceSourceIntegrityError):
        return SafeAnalysisError(
            500,
            exc.code,
            "Audience source data failed deterministic validation.",
        )
    if isinstance(exc, AudienceProviderError):
        return SafeAnalysisError(
            502,
            "audience_provider_unavailable",
            "Audience generation is temporarily unavailable.",
        )
    if isinstance(
        exc,
        (
            TopicAnalysisInvariantError,
            AudienceWorkflowInvariantError,
            AudienceAnalysisInvariantError,
        ),
    ):
        return SafeAnalysisError(
            500,
            "analysis_invariant_failed",
            "Audience analysis produced an inconsistent internal result.",
        )
    if isinstance(exc, RequestValidationError):
        return SafeAnalysisError(
            422,
            "request_validation_failed",
            "The request was not valid.",
        )
    return SafeAnalysisError(
        500,
        "internal_error",
        "An unexpected internal error occurred.",
    )
