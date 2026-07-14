"""Bounded generation, validation, and revision of audience decisions."""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from ..models import AudienceSegment, TopicCluster
from ..models.audience_generation import AudienceDecision
from ..progress import AnalysisProgressReporter, AnalysisProgressStage
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


@dataclass(frozen=True, slots=True)
class _AudienceWorkflowContext:
    """Run-scoped dependencies kept outside serializable graph state."""

    provider: AudienceGenerationProvider


class _AudienceWorkflowState(TypedDict, total=False):
    """Private state for the bounded audience workflow graph."""

    preparation: AudiencePreparation
    initial_provider_result: AudienceProviderResult
    initial_report: AudienceValidationReport
    initial_segments: dict[str, AudienceSegment]
    initial_skips: dict[str, ProviderSkippedCluster]
    known_invalid: dict[str, InvalidAudienceDecision]
    revision_prepared: tuple[PreparedAudienceCluster, ...]
    revision_requests: tuple[AudienceRevisionRequest, ...]
    revision_provider_result: AudienceProviderResult | None
    revision_report: AudienceValidationReport | None
    revised_segments: dict[str, AudienceSegment]
    revised_skips: dict[str, ProviderSkippedCluster]
    source_drops: dict[str, DroppedAudienceDecision]
    unmatched_drops: tuple[DroppedAudienceDecision, ...]
    revision_count: int
    revision_requested_cluster_count: int
    revision_failed: bool
    result: AudienceWorkflowResult


async def run_audience_workflow(
    preparation: AudiencePreparation,
    provider: AudienceGenerationProvider,
    *,
    progress_reporter: AnalysisProgressReporter | None = None,
) -> AudienceWorkflowResult:
    """Run the once-compiled graph with at most one targeted revision."""
    if progress_reporter is not None:
        return await _stream_audience_workflow(
            preparation,
            provider,
            progress_reporter,
        )

    state = await _AUDIENCE_WORKFLOW_GRAPH.ainvoke(
        {"preparation": preparation},
        context=_AudienceWorkflowContext(provider=provider),
    )
    return _require_workflow_result(state)


_NODE_PROGRESS_STAGES: Mapping[str, AnalysisProgressStage] = MappingProxyType(
    {
        "build_empty_result": "finalizing_audience_results",
        "generate_initial": "generating_audience_decisions",
        "validate_initial": "validating_audience_decisions",
        "revise_once": "revising_audience_decisions",
        "validate_revision": "validating_revised_decisions",
        "merge_and_build_result": "finalizing_audience_results",
    }
)


async def _stream_audience_workflow(
    preparation: AudiencePreparation,
    provider: AudienceGenerationProvider,
    progress_reporter: AnalysisProgressReporter,
) -> AudienceWorkflowResult:
    """Project allowlisted node starts while keeping graph state private."""
    root_output: object = None
    included_names = (
        _AUDIENCE_WORKFLOW_GRAPH.name,
        *_NODE_PROGRESS_STAGES,
    )
    async for event in _AUDIENCE_WORKFLOW_GRAPH.astream_events(
        {"preparation": preparation},
        context=_AudienceWorkflowContext(provider=provider),
        version="v2",
        include_names=included_names,
        output_keys=("result",),
    ):
        event_name = event.get("name")
        if event.get("event") == "on_chain_start":
            stage = _NODE_PROGRESS_STAGES.get(event_name)
            if stage is not None:
                await progress_reporter(stage)
            continue
        if event.get("event") == "on_chain_end" and not event.get(
            "parent_ids"
        ):
            data = event.get("data")
            if isinstance(data, Mapping):
                root_output = data.get("output")

    return _require_workflow_result(root_output)


def _require_workflow_result(state: object) -> AudienceWorkflowResult:
    if not isinstance(state, Mapping):
        result = None
    else:
        result = state.get("result")
    if not isinstance(result, AudienceWorkflowResult):
        raise AudienceWorkflowInvariantError(
            "audience workflow graph did not produce a result"
        )
    return result


def _route_preparation(
    state: _AudienceWorkflowState,
) -> Literal["build_empty_result", "generate_initial"]:
    if state["preparation"].clusters:
        return "generate_initial"
    return "build_empty_result"


def _build_empty_result(
    _state: _AudienceWorkflowState,
) -> _AudienceWorkflowState:
    return {
        "result": _build_result(
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
    }


async def _generate_initial(
    state: _AudienceWorkflowState,
    runtime: Runtime[_AudienceWorkflowContext],
) -> _AudienceWorkflowState:
    provider = runtime.context.provider
    initial_provider_result = await provider.generate(
        state["preparation"].contexts
    )
    return {"initial_provider_result": initial_provider_result}


def _validate_initial(
    state: _AudienceWorkflowState,
) -> _AudienceWorkflowState:
    preparation = state["preparation"]
    initial_provider_result = state["initial_provider_result"]
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
    revision_prepared = tuple(
        prepared
        for prepared in preparation.clusters
        if prepared.cluster_id in known_invalid
    )
    revision_requests = tuple(
        _build_revision_request(prepared, known_invalid[prepared.cluster_id])
        for prepared in revision_prepared
    )
    return {
        "initial_report": initial_report,
        "initial_segments": initial_segments,
        "initial_skips": initial_skips,
        "known_invalid": known_invalid,
        "revision_prepared": revision_prepared,
        "revision_requests": revision_requests,
        "revision_provider_result": None,
        "revision_report": None,
        "revised_segments": {},
        "revised_skips": {},
        "source_drops": {},
        "unmatched_drops": tuple(unmatched_drops),
        "revision_count": 0,
        "revision_requested_cluster_count": 0,
        "revision_failed": False,
    }


def _route_after_initial_validation(
    state: _AudienceWorkflowState,
) -> Literal["revise_once", "merge_and_build_result"]:
    if state["known_invalid"]:
        return "revise_once"
    return "merge_and_build_result"


async def _revise_once(
    state: _AudienceWorkflowState,
    runtime: Runtime[_AudienceWorkflowContext],
) -> _AudienceWorkflowState:
    revision_prepared = state["revision_prepared"]
    revision_requests = state["revision_requests"]
    try:
        revision_provider_result = await runtime.context.provider.revise(
            revision_requests
        )
    except AudienceProviderError:
        source_drops = {
            prepared.cluster_id: _drop_from_invalid(
                state["known_invalid"][prepared.cluster_id],
                phase="revision",
                drop_code=REVISION_PROVIDER_FAILURE,
            )
            for prepared in revision_prepared
        }
        return {
            "revision_count": MAX_REVISIONS,
            "revision_requested_cluster_count": len(revision_requests),
            "source_drops": source_drops,
            "revision_failed": True,
        }
    return {
        "revision_count": MAX_REVISIONS,
        "revision_requested_cluster_count": len(revision_requests),
        "revision_provider_result": revision_provider_result,
        "revision_failed": False,
    }


def _route_after_revision(
    state: _AudienceWorkflowState,
) -> Literal["validate_revision", "merge_and_build_result"]:
    if state["revision_failed"]:
        return "merge_and_build_result"
    return "validate_revision"


def _validate_revision(
    state: _AudienceWorkflowState,
) -> _AudienceWorkflowState:
    revision_provider_result = state["revision_provider_result"]
    if revision_provider_result is None:
        raise AudienceWorkflowInvariantError(
            "revision validation requires a provider result"
        )
    revision_preparation = _subset_preparation(
        state["preparation"],
        state["revision_prepared"],
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
    source_drops = dict(state["source_drops"])
    unmatched_drops = list(state["unmatched_drops"])
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
    return {
        "revision_report": revision_report,
        "revised_segments": revised_segments,
        "revised_skips": revised_skips,
        "source_drops": source_drops,
        "unmatched_drops": tuple(unmatched_drops),
    }


def _merge_and_build_result(
    state: _AudienceWorkflowState,
) -> _AudienceWorkflowState:
    preparation = state["preparation"]
    source_drops = state["source_drops"]
    segments, provider_skips = _merge_outcomes(
        preparation,
        state["initial_segments"],
        state["initial_skips"],
        state["revised_segments"],
        state["revised_skips"],
        source_drops,
    )
    ordered_source_drops = tuple(
        source_drops[prepared.cluster_id]
        for prepared in preparation.clusters
        if prepared.cluster_id in source_drops
    )
    result = _build_result(
        initial_provider_result=state["initial_provider_result"],
        initial_report=state["initial_report"],
        revision_provider_result=state["revision_provider_result"],
        revision_report=state["revision_report"],
        revision_count=state["revision_count"],
        revision_requested_cluster_count=(
            state["revision_requested_cluster_count"]
        ),
        segments=segments,
        provider_skips=provider_skips,
        dropped_decisions=(
            ordered_source_drops + state["unmatched_drops"]
        ),
    )
    return {"result": result}


def _build_audience_workflow_graph() -> CompiledStateGraph:
    graph = StateGraph(
        _AudienceWorkflowState,
        context_schema=_AudienceWorkflowContext,
    )
    graph.add_node("build_empty_result", _build_empty_result)
    graph.add_node("generate_initial", _generate_initial)
    graph.add_node("validate_initial", _validate_initial)
    graph.add_node("revise_once", _revise_once)
    graph.add_node("validate_revision", _validate_revision)
    graph.add_node("merge_and_build_result", _merge_and_build_result)
    graph.add_conditional_edges(START, _route_preparation)
    graph.add_edge("build_empty_result", END)
    graph.add_edge("generate_initial", "validate_initial")
    graph.add_conditional_edges(
        "validate_initial",
        _route_after_initial_validation,
    )
    graph.add_conditional_edges("revise_once", _route_after_revision)
    graph.add_edge("validate_revision", "merge_and_build_result")
    graph.add_edge("merge_and_build_result", END)
    return graph.compile()


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


_AUDIENCE_WORKFLOW_GRAPH = _build_audience_workflow_graph()
