"""Bounded generation, validation, and revision of audience decisions."""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from ..models import AudienceSegment, TopicCluster
from ..models.audience_generation import AudienceDecision
from .audience_finalization import (
    AudienceDecisionIssue,
    AudiencePreparation,
    AudienceValidationReport,
    InvalidAudienceDecision,
    PreparedAudienceCluster,
    ProviderSkippedCluster,
    finalize_audience_decisions,
)
from .audience_provider import (
    AudienceGenerationProvider,
    AudienceProviderError,
    AudienceProviderResult,
    AudienceRevisionIssue,
    AudienceRevisionRequest,
)


MAX_REVISIONS = 1

UNMATCHED_INITIAL_DECISION = "unmatched_initial_decision"
REVISION_PROVIDER_FAILURE = "revision_provider_failure"
UNRESOLVED_AFTER_REVISION = "unresolved_after_revision"


class AudienceWorkflowInvariantError(RuntimeError):
    """Raised when validated outcomes cannot map to their prepared sources."""


@dataclass(frozen=True, slots=True)
class DroppedAudienceDecision:
    """A validation outcome explicitly excluded from the final portfolio."""

    cluster_id: str
    source_cluster: TopicCluster | None
    decisions: tuple[AudienceDecision, ...]
    issues: tuple[AudienceDecisionIssue, ...]
    phase: Literal["initial", "revision"]
    drop_code: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "decisions", tuple(self.decisions))
        object.__setattr__(self, "issues", tuple(self.issues))


@dataclass(frozen=True, slots=True)
class AudienceWorkflowMetrics:
    """Immutable counts and reported provider usage for one bounded run."""

    initial_decision_count: int
    initial_valid_decision_count: int
    initial_invalid_report_count: int
    revision_count: int
    revision_requested_cluster_count: int
    revision_decision_count: int
    revision_valid_decision_count: int
    final_valid_decision_count: int
    final_segment_count: int
    final_provider_skip_count: int
    dropped_source_cluster_count: int
    dropped_unmatched_decision_count: int
    provider_call_count: int
    provider_input_tokens: int
    provider_output_tokens: int
    provider_total_tokens: int
    provider_elapsed_seconds: float
    validation_issue_count: int
    validation_issue_counts_by_code: Mapping[str, int]
    drop_counts_by_code: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "validation_issue_counts_by_code",
            MappingProxyType(dict(self.validation_issue_counts_by_code)),
        )
        object.__setattr__(
            self,
            "drop_counts_by_code",
            MappingProxyType(dict(self.drop_counts_by_code)),
        )


@dataclass(frozen=True, slots=True)
class AudienceWorkflowResult:
    """Validated final outcomes, explicit drops, and bounded-run evidence."""

    segments: tuple[AudienceSegment, ...]
    provider_skips: tuple[ProviderSkippedCluster, ...]
    dropped_decisions: tuple[DroppedAudienceDecision, ...]
    initial_provider_result: AudienceProviderResult | None
    revision_provider_result: AudienceProviderResult | None
    initial_validation_report: AudienceValidationReport | None
    revision_validation_report: AudienceValidationReport | None
    metrics: AudienceWorkflowMetrics
    is_publishable: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "segments", tuple(self.segments))
        object.__setattr__(self, "provider_skips", tuple(self.provider_skips))
        object.__setattr__(
            self,
            "dropped_decisions",
            tuple(self.dropped_decisions),
        )


async def run_audience_workflow(
    preparation: AudiencePreparation,
    provider: AudienceGenerationProvider,
) -> AudienceWorkflowResult:
    """Generate audiences and perform no more than one targeted revision."""
    if not preparation.clusters:
        return _build_result(
            initial_provider_result=None,
            initial_report=None,
            revision_provider_result=None,
            revision_report=None,
            revision_count=0,
            revision_requested_cluster_count=0,
            segments=(),
            provider_skips=(),
            dropped_decisions=(),
        )

    initial_provider_result = await provider.generate(preparation.contexts)
    initial_report = finalize_audience_decisions(
        preparation,
        initial_provider_result.response,
    )
    initial_segments = _index_segments(
        initial_report.valid_segments,
        preparation,
    )
    initial_skips = _index_skips(
        initial_report.provider_skips,
        preparation,
    )
    known_invalid = _index_known_invalid(
        initial_report.invalid_decisions,
        preparation,
    )
    unmatched_drops = [
        _drop_from_invalid(
            invalid,
            phase="initial",
            drop_code=UNMATCHED_INITIAL_DECISION,
        )
        for invalid in initial_report.invalid_decisions
        if invalid.source_cluster is None
    ]

    if not known_invalid:
        segments, provider_skips = _merge_outcomes(
            preparation,
            initial_segments,
            initial_skips,
            {},
            {},
            {},
        )
        return _build_result(
            initial_provider_result=initial_provider_result,
            initial_report=initial_report,
            revision_provider_result=None,
            revision_report=None,
            revision_count=0,
            revision_requested_cluster_count=0,
            segments=segments,
            provider_skips=provider_skips,
            dropped_decisions=tuple(unmatched_drops),
        )

    revision_prepared = tuple(
        prepared
        for prepared in preparation.clusters
        if prepared.cluster_id in known_invalid
    )
    revision_requests = tuple(
        _build_revision_request(prepared, known_invalid[prepared.cluster_id])
        for prepared in revision_prepared
    )
    revision_count = MAX_REVISIONS
    revision_provider_result: AudienceProviderResult | None = None
    revision_report: AudienceValidationReport | None = None
    revised_segments: dict[str, AudienceSegment] = {}
    revised_skips: dict[str, ProviderSkippedCluster] = {}
    source_drops: dict[str, DroppedAudienceDecision] = {}

    try:
        revision_provider_result = await provider.revise(revision_requests)
    except AudienceProviderError:
        source_drops = {
            prepared.cluster_id: _drop_from_invalid(
                known_invalid[prepared.cluster_id],
                phase="revision",
                drop_code=REVISION_PROVIDER_FAILURE,
            )
            for prepared in revision_prepared
        }
    else:
        revision_preparation = _subset_preparation(
            preparation,
            revision_prepared,
        )
        revision_report = finalize_audience_decisions(
            revision_preparation,
            revision_provider_result.response,
        )
        revised_segments = _index_segments(
            revision_report.valid_segments,
            revision_preparation,
        )
        revised_skips = _index_skips(
            revision_report.provider_skips,
            revision_preparation,
        )
        for invalid in revision_report.invalid_decisions:
            dropped = _drop_from_invalid(
                invalid,
                phase="revision",
                drop_code=UNRESOLVED_AFTER_REVISION,
            )
            if invalid.source_cluster is None:
                unmatched_drops.append(dropped)
            else:
                source_drops[invalid.cluster_id] = dropped

    segments, provider_skips = _merge_outcomes(
        preparation,
        initial_segments,
        initial_skips,
        revised_segments,
        revised_skips,
        source_drops,
    )
    ordered_source_drops = tuple(
        source_drops[prepared.cluster_id]
        for prepared in preparation.clusters
        if prepared.cluster_id in source_drops
    )
    return _build_result(
        initial_provider_result=initial_provider_result,
        initial_report=initial_report,
        revision_provider_result=revision_provider_result,
        revision_report=revision_report,
        revision_count=revision_count,
        revision_requested_cluster_count=len(revision_requests),
        segments=segments,
        provider_skips=provider_skips,
        dropped_decisions=ordered_source_drops + tuple(unmatched_drops),
    )


def _build_revision_request(
    prepared: PreparedAudienceCluster,
    invalid: InvalidAudienceDecision,
) -> AudienceRevisionRequest:
    return AudienceRevisionRequest(
        context=prepared.context,
        previous_decisions=invalid.decisions,
        validation_issues=tuple(
            AudienceRevisionIssue(
                code=issue.code,
                reference_id=issue.reference_id,
            )
            for issue in invalid.issues
        ),
    )


def _subset_preparation(
    preparation: AudiencePreparation,
    prepared_clusters: Sequence[PreparedAudienceCluster],
) -> AudiencePreparation:
    return AudiencePreparation(
        clusters=tuple(prepared_clusters),
        total_analyzed_views=preparation.total_analyzed_views,
        reference_cluster_ids=preparation.reference_cluster_ids,
    )


def _index_known_invalid(
    invalid_decisions: Sequence[InvalidAudienceDecision],
    preparation: AudiencePreparation,
) -> dict[str, InvalidAudienceDecision]:
    prepared_ids = {prepared.cluster_id for prepared in preparation.clusters}
    known_invalid: dict[str, InvalidAudienceDecision] = {}
    for invalid in invalid_decisions:
        if invalid.source_cluster is None:
            continue
        if (
            invalid.cluster_id not in prepared_ids
            or invalid.cluster_id in known_invalid
        ):
            raise AudienceWorkflowInvariantError(
                "invalid source decision does not map uniquely to preparation"
            )
        known_invalid[invalid.cluster_id] = invalid
    return known_invalid


def _index_segments(
    segments: Sequence[AudienceSegment],
    preparation: AudiencePreparation,
) -> dict[str, AudienceSegment]:
    prepared_ids = {prepared.cluster_id for prepared in preparation.clusters}
    indexed: dict[str, AudienceSegment] = {}
    for segment in segments:
        if len(segment.topic_cluster_ids) != 1:
            raise AudienceWorkflowInvariantError(
                "audience segment must resolve to exactly one source cluster"
            )
        cluster_id = segment.topic_cluster_ids[0]
        if cluster_id not in prepared_ids or cluster_id in indexed:
            raise AudienceWorkflowInvariantError(
                "audience segment does not map uniquely to preparation"
            )
        indexed[cluster_id] = segment
    return indexed


def _index_skips(
    skips: Sequence[ProviderSkippedCluster],
    preparation: AudiencePreparation,
) -> dict[str, ProviderSkippedCluster]:
    cluster_ids_by_identity = {
        id(prepared.cluster): prepared.cluster_id
        for prepared in preparation.clusters
    }
    indexed: dict[str, ProviderSkippedCluster] = {}
    for skipped in skips:
        cluster_id = cluster_ids_by_identity.get(id(skipped.cluster))
        if cluster_id is None or cluster_id in indexed:
            raise AudienceWorkflowInvariantError(
                "provider skip does not map uniquely to preparation"
            )
        indexed[cluster_id] = skipped
    return indexed


def _merge_outcomes(
    preparation: AudiencePreparation,
    initial_segments: Mapping[str, AudienceSegment],
    initial_skips: Mapping[str, ProviderSkippedCluster],
    revised_segments: Mapping[str, AudienceSegment],
    revised_skips: Mapping[str, ProviderSkippedCluster],
    source_drops: Mapping[str, DroppedAudienceDecision],
) -> tuple[tuple[AudienceSegment, ...], tuple[ProviderSkippedCluster, ...]]:
    segments: list[AudienceSegment] = []
    provider_skips: list[ProviderSkippedCluster] = []

    for prepared in preparation.clusters:
        cluster_id = prepared.cluster_id
        outcomes = [
            initial_segments.get(cluster_id),
            initial_skips.get(cluster_id),
            revised_segments.get(cluster_id),
            revised_skips.get(cluster_id),
            source_drops.get(cluster_id),
        ]
        present_outcomes = [outcome for outcome in outcomes if outcome is not None]
        if len(present_outcomes) != 1:
            raise AudienceWorkflowInvariantError(
                "each prepared cluster must have exactly one final outcome"
            )

        if cluster_id in initial_segments:
            segments.append(initial_segments[cluster_id])
        elif cluster_id in initial_skips:
            provider_skips.append(initial_skips[cluster_id])
        elif cluster_id in revised_segments:
            segments.append(revised_segments[cluster_id])
        elif cluster_id in revised_skips:
            provider_skips.append(revised_skips[cluster_id])

    return tuple(segments), tuple(provider_skips)


def _drop_from_invalid(
    invalid: InvalidAudienceDecision,
    *,
    phase: Literal["initial", "revision"],
    drop_code: str,
) -> DroppedAudienceDecision:
    return DroppedAudienceDecision(
        cluster_id=invalid.cluster_id,
        source_cluster=invalid.source_cluster,
        decisions=invalid.decisions,
        issues=invalid.issues,
        phase=phase,
        drop_code=drop_code,
    )


def _build_result(
    *,
    initial_provider_result: AudienceProviderResult | None,
    initial_report: AudienceValidationReport | None,
    revision_provider_result: AudienceProviderResult | None,
    revision_report: AudienceValidationReport | None,
    revision_count: int,
    revision_requested_cluster_count: int,
    segments: Sequence[AudienceSegment],
    provider_skips: Sequence[ProviderSkippedCluster],
    dropped_decisions: Sequence[DroppedAudienceDecision],
) -> AudienceWorkflowResult:
    provider_results = tuple(
        result
        for result in (initial_provider_result, revision_provider_result)
        if result is not None
    )
    issue_counts = Counter(
        issue.code
        for dropped in dropped_decisions
        for issue in dropped.issues
    )
    drop_counts = Counter(
        dropped.drop_code for dropped in dropped_decisions
    )
    metrics = AudienceWorkflowMetrics(
        initial_decision_count=(
            len(initial_provider_result.response.decisions)
            if initial_provider_result is not None
            else 0
        ),
        initial_valid_decision_count=(
            initial_report.metrics.valid_decision_count
            if initial_report is not None
            else 0
        ),
        initial_invalid_report_count=(
            len(initial_report.invalid_decisions)
            if initial_report is not None
            else 0
        ),
        revision_count=revision_count,
        revision_requested_cluster_count=revision_requested_cluster_count,
        revision_decision_count=(
            len(revision_provider_result.response.decisions)
            if revision_provider_result is not None
            else 0
        ),
        revision_valid_decision_count=(
            revision_report.metrics.valid_decision_count
            if revision_report is not None
            else 0
        ),
        final_valid_decision_count=len(segments) + len(provider_skips),
        final_segment_count=len(segments),
        final_provider_skip_count=len(provider_skips),
        dropped_source_cluster_count=sum(
            dropped.source_cluster is not None
            for dropped in dropped_decisions
        ),
        dropped_unmatched_decision_count=sum(
            dropped.source_cluster is None
            for dropped in dropped_decisions
        ),
        provider_call_count=(
            (1 if initial_provider_result is not None else 0) + revision_count
        ),
        provider_input_tokens=sum(
            result.usage.input_tokens for result in provider_results
        ),
        provider_output_tokens=sum(
            result.usage.output_tokens for result in provider_results
        ),
        provider_total_tokens=sum(
            result.usage.total_tokens for result in provider_results
        ),
        provider_elapsed_seconds=sum(
            result.elapsed_seconds for result in provider_results
        ),
        validation_issue_count=sum(issue_counts.values()),
        validation_issue_counts_by_code={
            code: issue_counts[code] for code in sorted(issue_counts)
        },
        drop_counts_by_code={
            code: drop_counts[code] for code in sorted(drop_counts)
        },
    )
    return AudienceWorkflowResult(
        segments=tuple(segments),
        provider_skips=tuple(provider_skips),
        dropped_decisions=tuple(dropped_decisions),
        initial_provider_result=initial_provider_result,
        revision_provider_result=revision_provider_result,
        initial_validation_report=initial_report,
        revision_validation_report=revision_report,
        metrics=metrics,
        is_publishable=True,
    )
