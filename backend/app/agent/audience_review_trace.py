"""Public-safe trace projection for the checkpointed review workflow."""

from collections.abc import Mapping

from ..models.audience_review import (
    ReviewDecisionTrace,
    ReviewTraceEvent,
    ReviewTraceEventCode,
    ReviewTraceIssue,
    ReviewTraceOutcome,
    ReviewTracePhase,
)
from .audience_finalization import AudiencePreparation
from .audience_trace import build_audience_decision_traces
from .audience_workflow import AudienceWorkflowInvariantError, AudienceWorkflowResult


def build_review_trace_projection(
    preparation: AudiencePreparation,
    workflow: AudienceWorkflowResult,
    review_ids_by_cluster: Mapping[str, str],
) -> tuple[ReviewDecisionTrace, ...]:
    """Project automatic reports without claiming candidates were published.

    The standard trace projector remains untouched. Terminal provider skips and
    validation drops retain their completed automatic events. For validated
    create decisions, the review projection verifies and omits only the
    standard path's terminal publication assertion, because publication has not
    happened in analyst-review mode.
    """
    automatic = build_audience_decision_traces(preparation, workflow)
    segment_trace_ids = set(automatic.segment_trace_ids)
    traces: list[ReviewDecisionTrace] = []
    for trace in automatic.traces:
        events = tuple(_copy_event(event) for event in trace.events)
        review_id = review_ids_by_cluster.get(trace.cluster_id)
        if trace.trace_id in segment_trace_ids:
            if not events or events[-1].code != "audience_published":
                raise AudienceWorkflowInvariantError(
                    "validated review candidate lacks publication boundary"
                )
            events = events[:-1]
            final_outcome: ReviewTraceOutcome = "queued"
        else:
            final_outcome = trace.final_outcome
        traces.append(
            ReviewDecisionTrace(
                trace_id=(
                    f"audience-review-trace:{review_id}"
                    if review_id is not None
                    else trace.trace_id
                ),
                cluster_id=trace.cluster_id,
                cluster_name=trace.cluster_name,
                source_known=trace.source_known,
                final_outcome=final_outcome,
                review_id=review_id,
                events=events,
            )
        )
    return tuple(traces)


def append_review_trace_event(
    trace: ReviewDecisionTrace,
    *,
    phase: ReviewTracePhase,
    code: ReviewTraceEventCode,
    final_outcome: ReviewTraceOutcome,
    outcome_code: str | None = None,
) -> ReviewDecisionTrace:
    """Return a new immutable trace with one truthful public-safe event."""
    event = ReviewTraceEvent(
        sequence=len(trace.events) + 1,
        phase=phase,
        code=code,
        outcome_code=outcome_code,
    )
    return trace.model_copy(
        update={
            "events": trace.events + (event,),
            "final_outcome": final_outcome,
        }
    )


def _copy_event(event: object) -> ReviewTraceEvent:
    return ReviewTraceEvent(
        sequence=event.sequence,
        phase=event.phase,
        code=event.code,
        outcome_code=event.outcome_code,
        issues=tuple(
            ReviewTraceIssue(
                code=issue.code,
                reference_id=issue.reference_id,
            )
            for issue in event.issues
        ),
    )
