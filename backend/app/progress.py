"""Internal fixed-code progress reporting for live analysis runs."""

from typing import Literal, Protocol, TypeAlias


AnalysisProgressStage: TypeAlias = Literal[
    "waiting_for_slot",
    "fetching_pageviews",
    "selecting_articles",
    "enriching_summaries",
    "modeling_topics",
    "routing_commercial_clusters",
    "preparing_audience_evidence",
    "generating_audience_decisions",
    "validating_audience_decisions",
    "revising_audience_decisions",
    "validating_revised_decisions",
    "finalizing_audience_results",
    "assembling_response",
]


class AnalysisProgressReporter(Protocol):
    """Receive one fixed progress code at an actual stage boundary."""

    async def __call__(self, stage: AnalysisProgressStage, /) -> None:
        """Record the newly started analysis stage."""
        ...


async def report_progress(
    reporter: AnalysisProgressReporter | None,
    stage: AnalysisProgressStage,
) -> None:
    """Report a stage when live progress is enabled for this invocation."""
    if reporter is not None:
        await reporter(stage)
