"""Truthful post-run traces for bounded audience decisions."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from ..models import AudienceSegment
from .audience_finalization import (
    AudienceDecisionIssue,
    AudiencePreparation,
    AudienceValidationReport,
    InvalidAudienceDecision,
    PreparedAudienceCluster,
)
from .audience_workflow import (
    REVISION_PROVIDER_FAILURE,
    AudienceWorkflowInvariantError,
    AudienceWorkflowResult,
    DroppedAudienceDecision,
)


AudienceTracePhase = Literal["initial", "revision", "final"]
AudienceTraceEventCode = Literal[
    "generation_requested",
    "decision_received",
    "validation_passed",
    "validation_failed",
    "revision_requested",
    "revision_failed",
    "audience_published",
    "provider_skipped",
    "decision_dropped",
]
AudienceTraceOutcome = Literal[
    "published",
    "provider_skipped",
    "validation_dropped",
]


class AudienceTraceInvariantError(AudienceWorkflowInvariantError):
    """Raised when a completed workflow cannot be projected consistently."""


@dataclass(frozen=True, slots=True)
class AudienceTraceIssue:
    """Immutable public-safe snapshot of one validation issue."""

    code: str
    reference_id: str | None = None


@dataclass(frozen=True, slots=True)
class AudienceTraceEvent:
    """One observable workflow event without model reasoning or raw output."""

    sequence: int
    phase: AudienceTracePhase
    code: AudienceTraceEventCode
    outcome_code: str | None = None
    issues: tuple[AudienceTraceIssue, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "issues", tuple(self.issues))


@dataclass(frozen=True, slots=True)
class AudienceDecisionTrace:
    """Immutable completed journey for one known or unmatched outcome."""

    trace_id: str
    cluster_id: str
    cluster_name: str | None
    source_known: bool
    final_outcome: AudienceTraceOutcome
    events: tuple[AudienceTraceEvent, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(self.events))


@dataclass(frozen=True, slots=True)
class AudienceTraceProjection:
    """Ordered traces and positional IDs aligned with final output tuples."""

    traces: tuple[AudienceDecisionTrace, ...]
    segment_trace_ids: tuple[str, ...]
    provider_skip_trace_ids: tuple[str, ...]
    drop_trace_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "traces", tuple(self.traces))
        object.__setattr__(
            self,
            "segment_trace_ids",
            tuple(self.segment_trace_ids),
        )
        object.__setattr__(
            self,
            "provider_skip_trace_ids",
            tuple(self.provider_skip_trace_ids),
        )
        object.__setattr__(
            self,
            "drop_trace_ids",
            tuple(self.drop_trace_ids),
        )


@dataclass(frozen=True, slots=True)
class _ValidationPartition:
    segment_ids: tuple[str, ...]
    skip_ids: tuple[str, ...]
    invalid_by_id: dict[str, InvalidAudienceDecision]


def build_audience_decision_traces(
    preparation: AudiencePreparation,
    workflow: AudienceWorkflowResult,
) -> AudienceTraceProjection:
    """Project actual completed workflow outcomes into immutable safe traces."""
    prepared_ids = tuple(
        prepared.cluster_id for prepared in preparation.clusters
    )
    if not prepared_ids:
        if (
            workflow.segments
            or workflow.provider_skips
            or workflow.dropped_decisions
            or workflow.initial_validation_report is not None
            or workflow.revision_validation_report is not None
        ):
            raise AudienceTraceInvariantError(
                "empty preparation has audience workflow outcomes"
            )
        return AudienceTraceProjection((), (), (), ())

    if len(set(prepared_ids)) != len(prepared_ids):
        raise AudienceTraceInvariantError(
            "prepared cluster IDs must be unique"
        )
    initial_report = workflow.initial_validation_report
    if initial_report is None:
        raise AudienceTraceInvariantError(
            "completed non-empty workflow requires initial validation"
        )
    initial = _partition_validation_report(initial_report, prepared_ids)
    revision_ids = tuple(
        cluster_id
        for cluster_id in prepared_ids
        if cluster_id in initial.invalid_by_id
    )
    revision = (
        _partition_validation_report(
            workflow.revision_validation_report,
            revision_ids,
        )
        if workflow.revision_validation_report is not None
        else None
    )
    if revision_ids and workflow.metrics.revision_count != 1:
        raise AudienceTraceInvariantError(
            "known invalid decisions require one recorded revision"
        )
    if not revision_ids and workflow.metrics.revision_count != 0:
        raise AudienceTraceInvariantError(
            "revision recorded without a known invalid decision"
        )

    segment_ids = _segment_cluster_ids(workflow.segments)
    _validate_ordered_subset(segment_ids, prepared_ids, "segments")
    source_drops: dict[str, DroppedAudienceDecision] = {}
    for dropped in workflow.dropped_decisions:
        if dropped.source_cluster is None:
            continue
        if (
            dropped.cluster_id not in prepared_ids
            or dropped.cluster_id in source_drops
        ):
            raise AudienceTraceInvariantError(
                "source drops do not map uniquely to preparation"
            )
        source_drops[dropped.cluster_id] = dropped
    if set(segment_ids) & set(source_drops):
        raise AudienceTraceInvariantError(
            "cluster cannot be both published and dropped"
        )
    provider_skip_ids = tuple(
        cluster_id
        for cluster_id in prepared_ids
        if cluster_id not in segment_ids and cluster_id not in source_drops
    )
    if len(provider_skip_ids) != len(workflow.provider_skips):
        raise AudienceTraceInvariantError(
            "provider skips do not complete the final cluster partition"
        )

    trace_ids_by_cluster = {
        cluster_id: _trace_id(index)
        for index, cluster_id in enumerate(prepared_ids)
    }
    known_traces = tuple(
        _build_known_trace(
            prepared,
            trace_ids_by_cluster[prepared.cluster_id],
            initial,
            revision,
            segment_ids,
            provider_skip_ids,
            source_drops,
        )
        for prepared in preparation.clusters
    )

    unmatched_traces: list[AudienceDecisionTrace] = []
    drop_trace_ids: list[str] = []
    next_trace_index = len(known_traces)
    for dropped in workflow.dropped_decisions:
        if dropped.source_cluster is not None:
            drop_trace_ids.append(trace_ids_by_cluster[dropped.cluster_id])
            continue
        trace_id = _trace_id(next_trace_index)
        next_trace_index += 1
        drop_trace_ids.append(trace_id)
        unmatched_traces.append(_build_unmatched_trace(dropped, trace_id))

    return AudienceTraceProjection(
        traces=known_traces + tuple(unmatched_traces),
        segment_trace_ids=tuple(
            trace_ids_by_cluster[cluster_id] for cluster_id in segment_ids
        ),
        provider_skip_trace_ids=tuple(
            trace_ids_by_cluster[cluster_id]
            for cluster_id in provider_skip_ids
        ),
        drop_trace_ids=tuple(drop_trace_ids),
    )


def _partition_validation_report(
    report: AudienceValidationReport,
    source_ids: tuple[str, ...],
) -> _ValidationPartition:
    segment_ids = _segment_cluster_ids(report.valid_segments)
    _validate_ordered_subset(segment_ids, source_ids, "validated segments")
    invalid_by_id: dict[str, InvalidAudienceDecision] = {}
    for invalid in report.invalid_decisions:
        if invalid.source_cluster is None:
            continue
        if (
            invalid.cluster_id not in source_ids
            or invalid.cluster_id in invalid_by_id
        ):
            raise AudienceTraceInvariantError(
                "invalid decisions do not map uniquely to validation sources"
            )
        invalid_by_id[invalid.cluster_id] = invalid
    if set(segment_ids) & set(invalid_by_id):
        raise AudienceTraceInvariantError(
            "validation source has conflicting outcomes"
        )
    skip_ids = tuple(
        cluster_id
        for cluster_id in source_ids
        if cluster_id not in segment_ids and cluster_id not in invalid_by_id
    )
    if len(skip_ids) != len(report.provider_skips):
        raise AudienceTraceInvariantError(
            "provider skips do not complete the validation partition"
        )
    return _ValidationPartition(segment_ids, skip_ids, invalid_by_id)


def _build_known_trace(
    prepared: PreparedAudienceCluster,
    trace_id: str,
    initial: _ValidationPartition,
    revision: _ValidationPartition | None,
    final_segment_ids: tuple[str, ...],
    final_skip_ids: tuple[str, ...],
    source_drops: dict[str, DroppedAudienceDecision],
) -> AudienceDecisionTrace:
    cluster_id = prepared.cluster_id
    final_outcome = _get_final_outcome(
        cluster_id,
        final_segment_ids,
        final_skip_ids,
        source_drops,
    )
    events: list[AudienceTraceEvent] = []
    _append_event(events, "initial", "generation_requested")

    if cluster_id in initial.segment_ids:
        _append_event(events, "initial", "decision_received")
        _append_event(events, "initial", "validation_passed")
        _require_outcome(final_outcome, "published")
    elif cluster_id in initial.skip_ids:
        _append_event(events, "initial", "decision_received")
        _append_event(events, "initial", "validation_passed")
        _require_outcome(final_outcome, "provider_skipped")
    else:
        invalid = initial.invalid_by_id[cluster_id]
        if invalid.decisions:
            _append_event(events, "initial", "decision_received")
        _append_event(
            events,
            "initial",
            "validation_failed",
            issues=invalid.issues,
        )
        _append_event(
            events,
            "revision",
            "revision_requested",
            issues=invalid.issues,
        )
        if revision is None:
            dropped = source_drops.get(cluster_id)
            if (
                dropped is None
                or dropped.drop_code != REVISION_PROVIDER_FAILURE
            ):
                raise AudienceTraceInvariantError(
                    "missing revision report requires a revision failure drop"
                )
            _append_event(
                events,
                "revision",
                "revision_failed",
                outcome_code=dropped.drop_code,
            )
            _require_outcome(final_outcome, "validation_dropped")
        elif cluster_id in revision.segment_ids:
            _append_event(events, "revision", "decision_received")
            _append_event(events, "revision", "validation_passed")
            _require_outcome(final_outcome, "published")
        elif cluster_id in revision.skip_ids:
            _append_event(events, "revision", "decision_received")
            _append_event(events, "revision", "validation_passed")
            _require_outcome(final_outcome, "provider_skipped")
        else:
            revised_invalid = revision.invalid_by_id[cluster_id]
            if revised_invalid.decisions:
                _append_event(events, "revision", "decision_received")
            _append_event(
                events,
                "revision",
                "validation_failed",
                issues=revised_invalid.issues,
            )
            _require_outcome(final_outcome, "validation_dropped")

    if final_outcome == "published":
        _append_event(events, "final", "audience_published")
    elif final_outcome == "provider_skipped":
        _append_event(events, "final", "provider_skipped")
    else:
        dropped = source_drops[cluster_id]
        _append_event(
            events,
            "final",
            "decision_dropped",
            outcome_code=dropped.drop_code,
            issues=dropped.issues,
        )
    return AudienceDecisionTrace(
        trace_id=trace_id,
        cluster_id=cluster_id,
        cluster_name=prepared.context.name,
        source_known=True,
        final_outcome=final_outcome,
        events=tuple(events),
    )


def _build_unmatched_trace(
    dropped: DroppedAudienceDecision,
    trace_id: str,
) -> AudienceDecisionTrace:
    events: list[AudienceTraceEvent] = []
    if dropped.decisions:
        _append_event(events, dropped.phase, "decision_received")
    _append_event(
        events,
        dropped.phase,
        "validation_failed",
        issues=dropped.issues,
    )
    _append_event(
        events,
        "final",
        "decision_dropped",
        outcome_code=dropped.drop_code,
        issues=dropped.issues,
    )
    return AudienceDecisionTrace(
        trace_id=trace_id,
        cluster_id=dropped.cluster_id,
        cluster_name=None,
        source_known=False,
        final_outcome="validation_dropped",
        events=tuple(events),
    )


def _get_final_outcome(
    cluster_id: str,
    segment_ids: tuple[str, ...],
    skip_ids: tuple[str, ...],
    source_drops: dict[str, DroppedAudienceDecision],
) -> AudienceTraceOutcome:
    outcomes = (
        cluster_id in segment_ids,
        cluster_id in skip_ids,
        cluster_id in source_drops,
    )
    if sum(outcomes) != 1:
        raise AudienceTraceInvariantError(
            "prepared cluster must have exactly one final trace outcome"
        )
    if outcomes[0]:
        return "published"
    if outcomes[1]:
        return "provider_skipped"
    return "validation_dropped"


def _segment_cluster_ids(
    segments: Sequence[AudienceSegment],
) -> tuple[str, ...]:
    cluster_ids: list[str] = []
    for segment in segments:
        if len(segment.topic_cluster_ids) != 1:
            raise AudienceTraceInvariantError(
                "audience segment must reference one source cluster"
            )
        cluster_ids.append(segment.topic_cluster_ids[0])
    return tuple(cluster_ids)


def _validate_ordered_subset(
    values: tuple[str, ...],
    source: tuple[str, ...],
    label: str,
) -> None:
    if len(set(values)) != len(values):
        raise AudienceTraceInvariantError(f"{label} contain duplicate IDs")
    ranks = {cluster_id: index for index, cluster_id in enumerate(source)}
    try:
        expected = tuple(sorted(values, key=ranks.__getitem__))
    except KeyError as exc:
        raise AudienceTraceInvariantError(
            f"{label} contain an unknown cluster ID"
        ) from exc
    if values != expected:
        raise AudienceTraceInvariantError(
            f"{label} do not preserve source order"
        )


def _append_event(
    events: list[AudienceTraceEvent],
    phase: AudienceTracePhase,
    code: AudienceTraceEventCode,
    *,
    outcome_code: str | None = None,
    issues: tuple[AudienceDecisionIssue, ...] = (),
) -> None:
    events.append(
        AudienceTraceEvent(
            sequence=len(events) + 1,
            phase=phase,
            code=code,
            outcome_code=outcome_code,
            issues=tuple(
                AudienceTraceIssue(issue.code, issue.reference_id)
                for issue in issues
            ),
        )
    )


def _require_outcome(
    actual: AudienceTraceOutcome,
    expected: AudienceTraceOutcome,
) -> None:
    if actual != expected:
        raise AudienceTraceInvariantError(
            "validation outcome conflicts with final workflow outcome"
        )


def _trace_id(index: int) -> str:
    return f"audience-trace-{index + 1:03d}"
