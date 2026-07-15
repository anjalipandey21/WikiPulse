"""Separate checkpointed graph for bounded analyst approval and rejection."""

import asyncio
from contextvars import Context
from dataclasses import dataclass, fields
from json import dumps
from typing import Literal, Protocol, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import interrupt

from ..models import Article, AudienceSegment, TopicCluster
from ..models.audience_generation import (
    AudienceGenerationResponse,
    CompactArticleContext,
    CompactClusterContext,
    CreateAudienceDecision,
    SkipClusterDecision,
)
from ..models.audience_review import (
    ActiveAnalystEditSnapshot,
    AppliedReviewCommandSnapshot,
    AnalystEditableField,
    AnalystRejectedOutcome,
    ApproveReviewCommand,
    EditDropCode,
    EditRecommendationReviewCommand,
    EditValidationDroppedOutcome,
    ExpiredReviewOutcome,
    ExpireReviewRunCommand,
    PendingAudienceReview,
    ProviderSkipSnapshot,
    RejectReviewCommand,
    REVIEW_VERSION,
    ReviewArticleSnapshot,
    ReviewCandidateSnapshot,
    ReviewClusterSnapshot,
    ReviewCodeCountSnapshot,
    ReviewCompactArticleSnapshot,
    ReviewCompactClusterSnapshot,
    ReviewConflictCode,
    ReviewDailyViewSnapshot,
    ReviewDecisionTrace,
    ReviewEvidenceSnapshot,
    ReviewPreparationSnapshot,
    ReviewReferenceOwnershipSnapshot,
    ReviewRecommendationSnapshot,
    ReviewResolutionSnapshot,
    ReviewSourceClusterSnapshot,
    RejectReasonCode,
    ReviewRunResult,
    ReviewWorkflowMetricsSnapshot,
    ValidationDropSnapshot,
    review_id_for,
)
from .audience_finalization import (
    CROSS_CLUSTER_SUPPORTING_REFERENCE,
    DUPLICATE_SUPPORTING_REFERENCE,
    TOO_FEW_SUPPORTING_REFERENCES,
    TOO_MANY_SUPPORTING_REFERENCES,
    UNKNOWN_SUPPORTING_REFERENCE,
    AudiencePreparation,
    PreparedAudienceCluster,
    finalize_audience_decisions,
)
from .audience_provider import (
    AnalystEditProviderRequest,
    AnalystEditProviderResult,
    AudienceGenerationProvider,
)
from .audience_review_trace import (
    append_review_trace_event,
    build_review_trace_projection,
)
from .audience_workflow import AudienceWorkflowResult, run_audience_workflow


class AudienceReviewConflictError(RuntimeError):
    """Stable conflict raised without leaking graph or provider internals."""

    def __init__(self, code: ReviewConflictCode) -> None:
        self.code = code
        super().__init__(code.value)


@dataclass(frozen=True, slots=True)
class AudienceReviewWorkflowContext:
    """Runtime-only provider dependency, never part of checkpointed state."""

    provider: AudienceGenerationProvider
    edit_executor: "AnalystEditExecutor | None" = None


class AnalystEditExecutor(Protocol):
    """Runtime-only single-flight boundary used by the edit graph node."""

    async def execute(
        self,
        request: AnalystEditProviderRequest,
        *,
        command_id: str,
        command_digest: str,
    ) -> AnalystEditProviderResult:
        ...


@dataclass(frozen=True, slots=True)
class _SafeRejectCommand:
    """Validated reject command after its private note was discarded."""

    type: Literal["reject"]
    run_id: str
    review_id: str
    cluster_id: str
    expected_version: int
    command_id: str
    command_digest: str
    reason_code: RejectReasonCode


@dataclass(frozen=True, slots=True)
class _SafeApproveCommand:
    """Validated approve command plus its safe canonical digest."""

    type: Literal["approve"]
    run_id: str
    review_id: str
    cluster_id: str
    expected_version: int
    command_id: str
    command_digest: str


@dataclass(frozen=True, slots=True)
class _SafeEditCommand:
    """Validated private edit command retained only in safe graph state."""

    type: Literal["edit_recommendation"]
    run_id: str
    review_id: str
    cluster_id: str
    expected_version: int
    command_id: str
    command_digest: str
    feedback: str
    fields_to_change: tuple[AnalystEditableField, ...]


class AudienceReviewState(TypedDict, total=False):
    """Only JSON primitives and strict review model dumps enter checkpoints."""

    run_id: str
    expires_at: str
    preparation_snapshot: dict[str, object]
    records: list[dict[str, object]]
    current_index: int
    provider_skips: list[dict[str, object]]
    validation_drops: list[dict[str, object]]
    metrics: dict[str, object]
    traces: list[dict[str, object]]
    pending_command: dict[str, object]
    active_edit: dict[str, object] | None
    last_applied_command: dict[str, object] | None
    completed: bool
    expired: bool
    failed: bool
    failure_code: str | None


def snapshot_preparation(
    preparation: AudiencePreparation,
) -> ReviewPreparationSnapshot:
    """Copy calculated source data into a frozen, JSON-safe graph input."""
    reference_owners = dict(preparation.reference_cluster_ids)
    clusters = []
    for prepared in preparation.clusters:
        resolution_reference_ids = tuple(prepared.resolution_map)
        if any(
            reference_id not in reference_owners
            for reference_id in resolution_reference_ids
        ):
            raise ValueError("resolved reference must have a preserved owner")
        if any(
            reference_id not in prepared.resolution_map
            or reference_id not in reference_owners
            for reference_id in prepared.evidence_reference_ids
        ):
            raise ValueError("evidence reference must be resolved and owned")
        clusters.append(
            ReviewClusterSnapshot(
                cluster_id=prepared.cluster_id,
                cluster_pageviews=prepared.cluster_pageviews,
                source=ReviewSourceClusterSnapshot(
                    id=prepared.cluster.id,
                    name=prepared.cluster.name,
                    description=prepared.cluster.description,
                    articles=tuple(
                        _snapshot_article(article)
                        for article in prepared.cluster.articles
                    ),
                    keywords=tuple(prepared.cluster.keywords),
                    total_views=prepared.cluster.total_views,
                    article_count=prepared.cluster.article_count,
                    confidence_score=prepared.cluster.confidence_score,
                ),
                context=ReviewCompactClusterSnapshot(
                    cluster_id=prepared.context.cluster_id,
                    name=prepared.context.name,
                    keywords=tuple(prepared.context.keywords),
                    total_views=prepared.context.total_views,
                    article_count=prepared.context.article_count,
                    topic_confidence=prepared.context.topic_confidence,
                    articles=tuple(
                        ReviewCompactArticleSnapshot(
                            reference_id=article.reference_id,
                            title=article.title,
                            weekly_views=article.weekly_views,
                            summary=article.summary,
                        )
                        for article in prepared.context.articles
                    ),
                ),
                evidence_reference_ids=tuple(
                    prepared.evidence_reference_ids
                ),
                resolution=tuple(
                    ReviewResolutionSnapshot(
                        reference_id=reference_id,
                        owning_cluster_id=reference_owners[reference_id],
                        article=_snapshot_article(article),
                    )
                    for reference_id, article in prepared.resolution_map.items()
                ),
            )
        )
    return ReviewPreparationSnapshot(
        clusters=tuple(clusters),
        total_analyzed_views=preparation.total_analyzed_views,
        reference_owners=tuple(
            ReviewReferenceOwnershipSnapshot(
                reference_id=reference_id,
                cluster_id=cluster_id,
            )
            for reference_id, cluster_id in reference_owners.items()
        ),
    )


def build_review_initial_state(
    preparation: AudiencePreparation,
    *,
    run_id: str,
    expires_at: str,
) -> AudienceReviewState:
    """Create the complete serializable input to the outer review graph."""
    return {
        "run_id": run_id,
        "expires_at": expires_at,
        "preparation_snapshot": snapshot_preparation(preparation).model_dump(
            mode="json"
        ),
        "records": [],
        "current_index": 0,
        "provider_skips": [],
        "validation_drops": [],
        "metrics": _empty_metrics().model_dump(mode="json"),
        "traces": [],
        "active_edit": None,
        "last_applied_command": None,
        "completed": False,
        "expired": False,
        "failed": False,
        "failure_code": None,
    }


def build_audience_review_graph(
    checkpointer: BaseCheckpointSaver,
) -> CompiledStateGraph:
    """Compile the isolated review topology with the supplied saver."""
    graph = StateGraph(
        AudienceReviewState,
        context_schema=AudienceReviewWorkflowContext,
    )
    graph.add_node("run_automatic_and_project", _run_automatic_and_project)
    graph.add_node("mark_review_pending", _mark_review_pending)
    graph.add_node("await_analyst_command", _await_analyst_command)
    graph.add_node("apply_analyst_command", _apply_analyst_command)
    graph.add_node(
        "regenerate_and_finalize_edit",
        _regenerate_and_finalize_edit,
    )
    graph.add_node("finalize_review_run", _finalize_review_run)
    graph.add_edge(START, "run_automatic_and_project")
    graph.add_conditional_edges(
        "run_automatic_and_project",
        _route_after_projection,
    )
    graph.add_edge("mark_review_pending", "await_analyst_command")
    graph.add_edge("await_analyst_command", "apply_analyst_command")
    graph.add_conditional_edges(
        "apply_analyst_command",
        _route_after_command,
    )
    graph.add_conditional_edges(
        "regenerate_and_finalize_edit",
        _route_after_command,
    )
    graph.add_edge("finalize_review_run", END)
    return graph.compile(checkpointer=checkpointer)


async def _run_automatic_and_project(
    state: AudienceReviewState,
    runtime: Runtime[AudienceReviewWorkflowContext],
) -> AudienceReviewState:
    """Run the existing graph opaquely, then return only safe snapshots."""
    try:
        preparation = _restore_preparation(state["preparation_snapshot"])
    except Exception:
        return _safe_failed_projection("review_projection_failed")
    # A fresh context prevents the outer checkpointer configuration from being
    # inherited by the existing, intentionally opaque automatic graph.
    try:
        workflow = await asyncio.create_task(
            run_audience_workflow(preparation, runtime.context.provider),
            context=Context(),
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        return _safe_failed_projection("automatic_workflow_failed")
    try:
        return _project_workflow(state["run_id"], preparation, workflow)
    except Exception:
        return _safe_failed_projection("review_projection_failed")


def _safe_failed_projection(
    failure_code: Literal[
        "automatic_workflow_failed",
        "review_projection_failed",
    ],
) -> AudienceReviewState:
    return {
        "records": [],
        "current_index": 0,
        "provider_skips": [],
        "validation_drops": [],
        "traces": [],
        "failed": True,
        "failure_code": failure_code,
    }


def _project_workflow(
    run_id: str,
    preparation: AudiencePreparation,
    workflow: AudienceWorkflowResult,
) -> AudienceReviewState:
    segments_by_cluster = {
        segment.topic_cluster_ids[0]: segment for segment in workflow.segments
    }
    review_ids = {
        prepared.cluster_id: review_id_for(run_id, prepared.cluster_id)
        for prepared in preparation.clusters
        if prepared.cluster_id in segments_by_cluster
    }
    traces = build_review_trace_projection(preparation, workflow, review_ids)
    traces_by_review_id = {
        trace.review_id: trace for trace in traces if trace.review_id is not None
    }
    records: list[dict[str, object]] = []
    for prepared in preparation.clusters:
        segment = segments_by_cluster.get(prepared.cluster_id)
        if segment is None:
            continue
        review_id = review_ids[prepared.cluster_id]
        evidence = tuple(
            ReviewEvidenceSnapshot(
                reference_id=reference_id,
                article=_snapshot_article(prepared.resolution_map[reference_id]),
            )
            for reference_id in prepared.evidence_reference_ids
        )
        supporting_titles = {
            article.normalized_title for article in segment.supporting_articles
        }
        supporting_reference_ids = tuple(
            item.reference_id
            for item in evidence
            if item.article.normalized_title in supporting_titles
        )
        recommendation = _snapshot_recommendation(
            segment,
            supporting_reference_ids,
        )
        record = ReviewCandidateSnapshot(
            run_id=run_id,
            review_id=review_id,
            cluster_id=prepared.cluster_id,
            ordinal=len(records),
            version=REVIEW_VERSION,
            status="queued",
            cluster_name=prepared.context.name,
            cluster_pageviews=prepared.cluster_pageviews,
            article_count=prepared.context.article_count,
            topic_confidence=prepared.context.topic_confidence,
            evidence=evidence,
            recommendation=recommendation,
            trace=traces_by_review_id[review_id],
        )
        records.append(record.model_dump(mode="json"))

    provider_skip_cluster_ids = tuple(
        trace.cluster_id
        for trace in traces
        if trace.final_outcome == "provider_skipped"
    )
    if len(provider_skip_cluster_ids) != len(workflow.provider_skips):
        raise ValueError("provider skip projection is inconsistent")
    provider_skips = tuple(
        ProviderSkipSnapshot(
            cluster_id=cluster_id,
            cluster_name=skip.cluster.name,
            reason=skip.reason,
        )
        for cluster_id, skip in zip(
            provider_skip_cluster_ids,
            workflow.provider_skips,
            strict=True,
        )
    )
    validation_drops = tuple(
        ValidationDropSnapshot(
            cluster_id=dropped.cluster_id,
            cluster_name=(
                dropped.source_cluster.name
                if dropped.source_cluster is not None
                else None
            ),
            source_known=dropped.source_cluster is not None,
            phase=dropped.phase,
            drop_code=dropped.drop_code,
            issue_codes=tuple(issue.code for issue in dropped.issues),
        )
        for dropped in workflow.dropped_decisions
    )
    return {
        "records": records,
        "current_index": 0,
        "provider_skips": [item.model_dump(mode="json") for item in provider_skips],
        "validation_drops": [
            item.model_dump(mode="json") for item in validation_drops
        ],
        "metrics": _snapshot_metrics(workflow).model_dump(mode="json"),
        "traces": [trace.model_dump(mode="json") for trace in traces],
    }


def _route_after_projection(
    state: AudienceReviewState,
) -> Literal["mark_review_pending", "finalize_review_run"]:
    if state.get("failed") or not state["records"]:
        return "finalize_review_run"
    return "mark_review_pending"


def _mark_review_pending(state: AudienceReviewState) -> AudienceReviewState:
    records = _load_records(state)
    index = state["current_index"]
    record = records[index]
    if record.status != "queued":
        raise AudienceReviewConflictError(ReviewConflictCode.REVIEW_NOT_PENDING)
    trace = append_review_trace_event(
        record.trace,
        phase="review",
        code="review_requested",
        final_outcome="pending_review",
    )
    records[index] = record.model_copy(
        update={"status": "pending_review", "trace": trace}
    )
    return {
        "records": _dump_records(records),
        "traces": _merge_candidate_traces(state, records),
    }


def _await_analyst_command(state: AudienceReviewState) -> AudienceReviewState:
    """Interrupt-only node: no calls or external effects precede interrupt()."""
    payload = pending_review_from_state(state).model_dump(mode="json")
    resumed_value = interrupt(payload)
    return {"pending_command": _safe_checkpoint_command(resumed_value)}


def _apply_analyst_command(state: AudienceReviewState) -> AudienceReviewState:
    records = _load_records(state)
    current = records[state["current_index"]]
    raw_command = state["pending_command"]
    if raw_command.get("type") == "expire_run":
        command = ExpireReviewRunCommand.model_validate_json(dumps(raw_command))
        _require_equal(command.run_id, state["run_id"], ReviewConflictCode.RUN_ID_MISMATCH)
        return _expire_records(state, records)

    command = _parse_analyst_command(raw_command)
    _validate_current_command(state, current, command)
    if isinstance(command, _SafeEditCommand):
        if current.edit_attempted:
            raise AudienceReviewConflictError(
                ReviewConflictCode.REVIEW_NOT_PENDING
            )
        editing_trace = append_review_trace_event(
            current.trace,
            phase="edit",
            code="analyst_edit_requested",
            final_outcome="editing",
        )
        resulting_version = current.version + 1
        records[state["current_index"]] = current.model_copy(
            update={
                "status": "editing",
                "version": resulting_version,
                "edit_attempted": True,
                "trace": editing_trace,
            }
        )
        active_edit = ActiveAnalystEditSnapshot(
            command_id=command.command_id,
            command_digest=command.command_digest,
            run_id=command.run_id,
            review_id=command.review_id,
            cluster_id=command.cluster_id,
            accepted_version=current.version,
            resulting_version=resulting_version,
            feedback=command.feedback,
            fields_to_change=command.fields_to_change,
        )
        return {
            "records": _dump_records(records),
            "traces": _merge_candidate_traces(state, records),
            "active_edit": active_edit.model_dump(mode="json"),
        }
    if isinstance(command, _SafeApproveCommand):
        approved_trace = append_review_trace_event(
            current.trace,
            phase="review",
            code="analyst_approved",
            final_outcome="pending_review",
        )
        published_trace = append_review_trace_event(
            approved_trace,
            phase="final",
            code="audience_published",
            final_outcome="published",
        )
        records[state["current_index"]] = current.model_copy(
            update={
                "status": "published",
                "terminal_command_id": command.command_id,
                "trace": published_trace,
            }
        )
    else:
        rejected_trace = append_review_trace_event(
            current.trace,
            phase="final",
            code="analyst_rejected",
            outcome_code=command.reason_code.value,
            final_outcome="analyst_rejected",
        )
        records[state["current_index"]] = current.model_copy(
            update={
                "status": "rejected",
                "terminal_command_id": command.command_id,
                "reject_reason_code": command.reason_code,
                "trace": rejected_trace,
            }
        )
    resulting_status = (
        "published" if isinstance(command, _SafeApproveCommand) else "rejected"
    )
    applied = AppliedReviewCommandSnapshot(
        command_id=command.command_id,
        command_digest=command.command_digest,
        command_type=command.type,
        review_id=command.review_id,
        cluster_id=command.cluster_id,
        review_version=current.version,
        resulting_status=resulting_status,
    )
    return {
        "records": _dump_records(records),
        "current_index": state["current_index"] + 1,
        "traces": _merge_candidate_traces(state, records),
        "last_applied_command": applied.model_dump(mode="json"),
    }


async def _regenerate_and_finalize_edit(
    state: AudienceReviewState,
    runtime: Runtime[AudienceReviewWorkflowContext],
) -> AudienceReviewState:
    """Execute one single-flight edit and checkpoint only its safe outcome."""
    try:
        active = ActiveAnalystEditSnapshot.model_validate_json(
            dumps(state.get("active_edit"))
        )
        records = _load_records(state)
        current = records[state["current_index"]]
        if (
            current.status != "editing"
            or not current.edit_attempted
            or current.review_id != active.review_id
            or current.cluster_id != active.cluster_id
            or current.version != active.resulting_version
        ):
            return _terminalize_edit_drop(
                state,
                records,
                current,
                active,
                "edit_internal_failure",
            )
        preparation = _restore_preparation(state["preparation_snapshot"])
        prepared = next(
            item
            for item in preparation.clusters
            if item.cluster_id == current.cluster_id
        )
        request = AnalystEditProviderRequest(
            expected_cluster_id=current.cluster_id,
            context=prepared.context,
            original_decision=_restore_original_decision(current),
            feedback=active.feedback,
            fields_to_change=active.fields_to_change,
        )
        if runtime.context.edit_executor is None:
            result = AnalystEditProviderResult(
                status="provider_failed",
                response=None,
                elapsed_seconds=0,
                usage=None,
            )
        else:
            result = await runtime.context.edit_executor.execute(
                request,
                command_id=active.command_id,
                command_digest=active.command_digest,
            )
        return _finalize_edit_result(
            state,
            records,
            current,
            active,
            preparation,
            prepared,
            result,
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        try:
            active = ActiveAnalystEditSnapshot.model_validate_json(
                dumps(state.get("active_edit"))
            )
            records = _load_records(state)
            current = records[state["current_index"]]
            return _terminalize_edit_drop(
                state,
                records,
                current,
                active,
                "edit_internal_failure",
            )
        except Exception:
            return {
                "failed": True,
                "failure_code": "review_projection_failed",
            }


_REFERENCE_ISSUE_CODES = frozenset(
    {
        CROSS_CLUSTER_SUPPORTING_REFERENCE,
        DUPLICATE_SUPPORTING_REFERENCE,
        TOO_FEW_SUPPORTING_REFERENCES,
        TOO_MANY_SUPPORTING_REFERENCES,
        UNKNOWN_SUPPORTING_REFERENCE,
    }
)


def _finalize_edit_result(
    state: AudienceReviewState,
    records: list[ReviewCandidateSnapshot],
    current: ReviewCandidateSnapshot,
    active: ActiveAnalystEditSnapshot,
    preparation: AudiencePreparation,
    prepared: PreparedAudienceCluster,
    result: AnalystEditProviderResult,
) -> AudienceReviewState:
    if result.status != "completed":
        drop_code: EditDropCode = {
            "provider_failed": "edit_provider_failed",
            "refused": "edit_provider_refused",
            "missing_output": "edit_provider_missing_output",
        }[result.status]
        return _terminalize_edit_drop(
            state,
            records,
            current,
            active,
            drop_code,
        )
    if result.response is None:
        return _terminalize_edit_drop(
            state,
            records,
            current,
            active,
            "edit_provider_missing_output",
        )
    decisions = result.response.decisions
    if not decisions:
        return _terminalize_edit_drop(
            state,
            records,
            current,
            active,
            "edit_zero_decisions",
        )
    received_trace = append_review_trace_event(
        current.trace,
        phase="edit",
        code="edited_decision_received",
        final_outcome="editing",
    )
    current = current.model_copy(update={"trace": received_trace})
    records[state["current_index"]] = current
    if len(decisions) != 1:
        return _terminalize_edit_drop(
            state,
            records,
            current,
            active,
            "edit_multiple_decisions",
        )
    decision = decisions[0]
    if isinstance(decision, SkipClusterDecision):
        return _terminalize_edit_drop(
            state,
            records,
            current,
            active,
            "edit_provider_skip_not_allowed",
        )
    if decision.cluster_id != current.cluster_id:
        return _terminalize_edit_drop(
            state,
            records,
            current,
            active,
            "edit_wrong_cluster",
        )
    single_preparation = AudiencePreparation(
        clusters=(prepared,),
        total_analyzed_views=preparation.total_analyzed_views,
        reference_cluster_ids=preparation.reference_cluster_ids,
    )
    report = finalize_audience_decisions(
        single_preparation,
        AudienceGenerationResponse(decisions=[decision]),
    )
    if (
        len(report.valid_segments) != 1
        or report.provider_skips
        or report.invalid_decisions
    ):
        issue_codes = {
            issue.code
            for invalid in report.invalid_decisions
            for issue in invalid.issues
        }
        drop_code = (
            "edit_unsupported_references"
            if issue_codes & _REFERENCE_ISSUE_CODES
            else "edit_validation_failed"
        )
        return _terminalize_edit_drop(
            state,
            records,
            current,
            active,
            drop_code,
        )
    edited = _snapshot_recommendation(
        report.valid_segments[0],
        tuple(
            reference_id
            for reference_id in prepared.evidence_reference_ids
            if reference_id
            in set(decision.supporting_article_reference_ids)
        ),
    )
    if not _edit_conforms(
        current.recommendation,
        edited,
        active.fields_to_change,
    ):
        return _terminalize_edit_drop(
            state,
            records,
            current,
            active,
            "edit_intent_conformance_failed",
        )
    validated_trace = append_review_trace_event(
        current.trace,
        phase="edit",
        code="edited_decision_validated",
        final_outcome="editing",
    )
    published_trace = append_review_trace_event(
        validated_trace,
        phase="final",
        code="edited_audience_published",
        final_outcome="published",
    )
    updated = current.model_copy(
        update={
            "status": "published",
            "edited_recommendation": edited,
            "terminal_command_id": active.command_id,
            "trace": published_trace,
        }
    )
    records[state["current_index"]] = updated
    applied = AppliedReviewCommandSnapshot(
        command_id=active.command_id,
        command_digest=active.command_digest,
        command_type="edit_recommendation",
        review_id=active.review_id,
        cluster_id=active.cluster_id,
        review_version=active.resulting_version,
        resulting_status="published",
    )
    return {
        "records": _dump_records(records),
        "current_index": state["current_index"] + 1,
        "traces": _merge_candidate_traces(state, records),
        "active_edit": None,
        "pending_command": {},
        "last_applied_command": applied.model_dump(mode="json"),
    }


def _terminalize_edit_drop(
    state: AudienceReviewState,
    records: list[ReviewCandidateSnapshot],
    current: ReviewCandidateSnapshot,
    active: ActiveAnalystEditSnapshot,
    drop_code: EditDropCode,
) -> AudienceReviewState:
    dropped_trace = append_review_trace_event(
        current.trace,
        phase="final",
        code="analyst_edit_dropped",
        outcome_code=drop_code,
        final_outcome="edit_validation_dropped",
    )
    updated = current.model_copy(
        update={
            "status": "edit_validation_dropped",
            "edit_drop_code": drop_code,
            "terminal_command_id": active.command_id,
            "trace": dropped_trace,
        }
    )
    records[state["current_index"]] = updated
    applied = AppliedReviewCommandSnapshot(
        command_id=active.command_id,
        command_digest=active.command_digest,
        command_type="edit_recommendation",
        review_id=active.review_id,
        cluster_id=active.cluster_id,
        review_version=active.resulting_version,
        resulting_status="edit_validation_dropped",
    )
    return {
        "records": _dump_records(records),
        "current_index": state["current_index"] + 1,
        "traces": _merge_candidate_traces(state, records),
        "active_edit": None,
        "pending_command": {},
        "last_applied_command": applied.model_dump(mode="json"),
    }


def _restore_original_decision(
    current: ReviewCandidateSnapshot,
) -> CreateAudienceDecision:
    recommendation = current.recommendation
    return CreateAudienceDecision(
        decision="create_audience",
        cluster_id=current.cluster_id,
        name=recommendation.name,
        description=recommendation.description,
        supporting_article_reference_ids=list(
            recommendation.supporting_article_reference_ids
        ),
        buying_power=recommendation.buying_power,
        buying_power_reason=recommendation.buying_power_reason,
        brand_categories=list(recommendation.brand_categories),
        commercial_confidence=recommendation.commercial_confidence,
        commercial_confidence_reason=(
            recommendation.commercial_confidence_reason
        ),
    )


def _snapshot_recommendation(
    segment: AudienceSegment,
    supporting_reference_ids: tuple[str, ...],
) -> ReviewRecommendationSnapshot:
    return ReviewRecommendationSnapshot(
        audience_id=segment.id,
        name=segment.name,
        description=segment.description,
        topic_cluster_ids=tuple(segment.topic_cluster_ids),
        size_index=segment.size_index,
        buying_power=segment.buying_power,
        buying_power_reason=segment.buying_power_reason,
        brand_categories=tuple(segment.brand_categories),
        supporting_article_reference_ids=supporting_reference_ids,
        supporting_articles=tuple(
            _snapshot_article(article) for article in segment.supporting_articles
        ),
        commercial_confidence=segment.commercial_confidence,
        commercial_confidence_reason=segment.commercial_confidence_reason,
    )


def _edit_conforms(
    original: ReviewRecommendationSnapshot,
    edited: ReviewRecommendationSnapshot,
    fields_to_change: tuple[AnalystEditableField, ...],
) -> bool:
    if (
        original.audience_id != edited.audience_id
        or original.topic_cluster_ids != edited.topic_cluster_ids
        or original.size_index != edited.size_index
    ):
        return False
    selected = set(fields_to_change)
    exact_groups = {
        AnalystEditableField.AUDIENCE_POSITIONING: (
            original.name,
            original.description,
        ) == (edited.name, edited.description),
        AnalystEditableField.SUPPORTING_EVIDENCE: (
            original.supporting_article_reference_ids
            == edited.supporting_article_reference_ids
        ),
        AnalystEditableField.BUYING_POWER: (
            original.buying_power,
            original.buying_power_reason,
        ) == (edited.buying_power, edited.buying_power_reason),
        AnalystEditableField.BRAND_CATEGORIES: (
            original.brand_categories == edited.brand_categories
        ),
        AnalystEditableField.COMMERCIAL_CONFIDENCE: (
            original.commercial_confidence,
            original.commercial_confidence_reason,
        ) == (
            edited.commercial_confidence,
            edited.commercial_confidence_reason,
        ),
    }
    if any(
        not unchanged
        for group, unchanged in exact_groups.items()
        if group not in selected
    ):
        return False
    meaningful_changes = {
        AnalystEditableField.AUDIENCE_POSITIONING: not exact_groups[
            AnalystEditableField.AUDIENCE_POSITIONING
        ],
        AnalystEditableField.SUPPORTING_EVIDENCE: (
            frozenset(original.supporting_article_reference_ids)
            != frozenset(edited.supporting_article_reference_ids)
        ),
        AnalystEditableField.BUYING_POWER: not exact_groups[
            AnalystEditableField.BUYING_POWER
        ],
        AnalystEditableField.BRAND_CATEGORIES: (
            frozenset(value.casefold() for value in original.brand_categories)
            != frozenset(value.casefold() for value in edited.brand_categories)
        ),
        AnalystEditableField.COMMERCIAL_CONFIDENCE: not exact_groups[
            AnalystEditableField.COMMERCIAL_CONFIDENCE
        ],
    }
    return any(meaningful_changes[group] for group in selected)


def _route_after_command(
    state: AudienceReviewState,
) -> Literal[
    "regenerate_and_finalize_edit",
    "mark_review_pending",
    "finalize_review_run",
]:
    records = _load_records(state)
    if (
        state["current_index"] < len(records)
        and records[state["current_index"]].status == "editing"
    ):
        return "regenerate_and_finalize_edit"
    if state.get("expired") or state["current_index"] >= len(state["records"]):
        return "finalize_review_run"
    return "mark_review_pending"


def _finalize_review_run(_state: AudienceReviewState) -> AudienceReviewState:
    return {"completed": True}


def pending_review_from_state(state: AudienceReviewState) -> PendingAudienceReview:
    records = _load_records(state)
    record = records[state["current_index"]]
    if record.status != "pending_review":
        raise AudienceReviewConflictError(ReviewConflictCode.REVIEW_NOT_PENDING)
    return PendingAudienceReview(
        run_id=record.run_id,
        review_id=record.review_id,
        cluster_id=record.cluster_id,
        version=record.version,
        ordinal=record.ordinal,
        queue_size=len(records),
        cluster_name=record.cluster_name,
        cluster_pageviews=record.cluster_pageviews,
        topic_confidence=record.topic_confidence,
        evidence=record.evidence,
        recommendation=record.recommendation,
        expires_at=state["expires_at"],
    )


def build_review_run_result(
    state: AudienceReviewState,
    *,
    thread_id: str,
    failed: bool = False,
) -> ReviewRunResult:
    """Create a safe registry/API-facing view with no private analyst note."""
    records = _load_records(state)
    pending = next(
        (record for record in records if record.status == "pending_review"),
        None,
    )
    if failed or state.get("failed"):
        status = "failed"
    elif pending is not None:
        status = "pending_review"
    elif any(record.status == "editing" for record in records):
        status = "editing"
    elif state.get("expired"):
        status = "expired"
    elif state.get("completed"):
        status = "completed"
    else:
        status = "running"
    pending_dto = (
        pending_review_from_state(state) if pending is not None else None
    )
    return ReviewRunResult(
        run_id=state["run_id"],
        thread_id=thread_id,
        status=status,
        is_complete=status in {"completed", "expired", "failed"},
        failure_code=state.get("failure_code"),
        pending_review=pending_dto,
        review_candidates=tuple(records),
        published_audiences=tuple(
            record.edited_recommendation or record.recommendation
            for record in records
            if record.status == "published"
        ),
        rejected_reviews=tuple(
            AnalystRejectedOutcome(
                run_id=record.run_id,
                review_id=record.review_id,
                cluster_id=record.cluster_id,
                reason_code=record.reject_reason_code,
            )
            for record in records
            if record.status == "rejected"
            and record.reject_reason_code is not None
        ),
        expired_reviews=tuple(
            ExpiredReviewOutcome(
                run_id=record.run_id,
                review_id=record.review_id,
                cluster_id=record.cluster_id,
            )
            for record in records
            if record.status == "expired"
        ),
        edit_validation_drops=tuple(
            EditValidationDroppedOutcome(
                run_id=record.run_id,
                review_id=record.review_id,
                cluster_id=record.cluster_id,
                drop_code=record.edit_drop_code,
            )
            for record in records
            if record.status == "edit_validation_dropped"
            and record.edit_drop_code is not None
        ),
        provider_skips=tuple(
            ProviderSkipSnapshot.model_validate_json(dumps(item))
            for item in state["provider_skips"]
        ),
        validation_drops=tuple(
            ValidationDropSnapshot.model_validate_json(dumps(item))
            for item in state["validation_drops"]
        ),
        metrics=ReviewWorkflowMetricsSnapshot.model_validate_json(
            dumps(state["metrics"])
        ),
        traces=tuple(
            ReviewDecisionTrace.model_validate_json(dumps(item))
            for item in state["traces"]
        ),
    )


def _expire_records(
    state: AudienceReviewState,
    records: list[ReviewCandidateSnapshot],
) -> AudienceReviewState:
    updated: list[ReviewCandidateSnapshot] = []
    for record in records:
        if record.status in {"pending_review", "queued"}:
            trace = append_review_trace_event(
                record.trace,
                phase="final",
                code="review_expired",
                final_outcome="expired",
            )
            updated.append(record.model_copy(update={"status": "expired", "trace": trace}))
        else:
            updated.append(record)
    return {
        "records": _dump_records(updated),
        "current_index": len(updated),
        "traces": _merge_candidate_traces(state, updated),
        "expired": True,
    }


def _validate_current_command(
    state: AudienceReviewState,
    current: ReviewCandidateSnapshot,
    command: _SafeApproveCommand | _SafeRejectCommand | _SafeEditCommand,
) -> None:
    _require_equal(command.run_id, state["run_id"], ReviewConflictCode.RUN_ID_MISMATCH)
    _require_equal(command.review_id, current.review_id, ReviewConflictCode.REVIEW_ID_MISMATCH)
    _require_equal(command.cluster_id, current.cluster_id, ReviewConflictCode.CLUSTER_ID_MISMATCH)
    _require_equal(command.expected_version, current.version, ReviewConflictCode.STALE_VERSION)
    if current.status != "pending_review":
        raise AudienceReviewConflictError(ReviewConflictCode.REVIEW_NOT_PENDING)


def _parse_analyst_command(
    raw: dict[str, object],
) -> _SafeApproveCommand | _SafeRejectCommand | _SafeEditCommand:
    validation_payload = dict(raw)
    command_digest = _parse_command_digest(
        validation_payload.pop("command_digest", None)
    )
    if raw.get("type") == "approve":
        command = ApproveReviewCommand.model_validate_json(
            dumps(validation_payload)
        )
        return _SafeApproveCommand(
            type="approve",
            run_id=command.run_id,
            review_id=command.review_id,
            cluster_id=command.cluster_id,
            expected_version=command.expected_version,
            command_id=command.command_id,
            command_digest=command_digest,
        )
    if raw.get("type") == "reject":
        if raw.get("reason_code") == "other":
            validation_payload["private_note"] = "accepted outside graph state"
        command = RejectReviewCommand.model_validate_json(
            dumps(validation_payload)
        )
        return _SafeRejectCommand(
            type="reject",
            run_id=command.run_id,
            review_id=command.review_id,
            cluster_id=command.cluster_id,
            expected_version=command.expected_version,
            command_id=command.command_id,
            command_digest=command_digest,
            reason_code=command.reason_code,
        )
    if raw.get("type") == "edit_recommendation":
        command = EditRecommendationReviewCommand.model_validate_json(
            dumps(validation_payload)
        )
        return _SafeEditCommand(
            type="edit_recommendation",
            run_id=command.run_id,
            review_id=command.review_id,
            cluster_id=command.cluster_id,
            expected_version=command.expected_version,
            command_id=command.command_id,
            command_digest=command_digest,
            feedback=command.feedback,
            fields_to_change=command.fields_to_change,
        )
    raise AudienceReviewConflictError(ReviewConflictCode.REVIEW_NOT_PENDING)


def _safe_checkpoint_command(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise AudienceReviewConflictError(ReviewConflictCode.REVIEW_NOT_PENDING)
    if raw.get("type") == "expire_run":
        return ExpireReviewRunCommand.model_validate_json(
            dumps(raw)
        ).model_dump(mode="json")
    command = _parse_analyst_command(raw)
    safe_command = {
        "type": command.type,
        "run_id": command.run_id,
        "review_id": command.review_id,
        "cluster_id": command.cluster_id,
        "expected_version": command.expected_version,
        "command_id": command.command_id,
        "command_digest": command.command_digest,
    }
    if isinstance(command, _SafeRejectCommand):
        safe_command["reason_code"] = command.reason_code.value
    elif isinstance(command, _SafeEditCommand):
        safe_command["feedback"] = command.feedback
        safe_command["fields_to_change"] = [
            field.value for field in command.fields_to_change
        ]
    return safe_command


def _parse_command_digest(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AudienceReviewConflictError(ReviewConflictCode.COMMAND_ID_REUSED)
    return value


def _require_equal(actual: object, expected: object, code: ReviewConflictCode) -> None:
    if actual != expected:
        raise AudienceReviewConflictError(code)


def _load_records(state: AudienceReviewState) -> list[ReviewCandidateSnapshot]:
    return [
        ReviewCandidateSnapshot.model_validate_json(dumps(item))
        for item in state["records"]
    ]


def _dump_records(records: list[ReviewCandidateSnapshot]) -> list[dict[str, object]]:
    return [record.model_dump(mode="json") for record in records]


def _merge_candidate_traces(
    state: AudienceReviewState,
    records: list[ReviewCandidateSnapshot],
) -> list[dict[str, object]]:
    by_review_id = {record.review_id: record.trace for record in records}
    merged = []
    for raw_trace in state["traces"]:
        trace = ReviewDecisionTrace.model_validate_json(dumps(raw_trace))
        merged.append(
            by_review_id.get(trace.review_id, trace).model_dump(mode="json")
        )
    return merged


def _restore_preparation(raw: dict[str, object]) -> AudiencePreparation:
    snapshot = ReviewPreparationSnapshot.model_validate_json(dumps(raw))
    prepared_clusters = []
    reference_cluster_ids = {
        entry.reference_id: entry.cluster_id
        for entry in snapshot.reference_owners
    }
    for prepared_snapshot in snapshot.clusters:
        source = prepared_snapshot.source
        articles = [_restore_article(article) for article in source.articles]
        cluster = TopicCluster(
            id=source.id,
            name=source.name,
            description=source.description,
            articles=articles,
            keywords=list(source.keywords),
            total_views=source.total_views,
            article_count=source.article_count,
            confidence_score=source.confidence_score,
        )
        context = CompactClusterContext(
            cluster_id=prepared_snapshot.context.cluster_id,
            name=prepared_snapshot.context.name,
            keywords=list(prepared_snapshot.context.keywords),
            total_views=prepared_snapshot.context.total_views,
            article_count=prepared_snapshot.context.article_count,
            topic_confidence=prepared_snapshot.context.topic_confidence,
            articles=[
                CompactArticleContext(
                    reference_id=article.reference_id,
                    title=article.title,
                    weekly_views=article.weekly_views,
                    summary=article.summary,
                )
                for article in prepared_snapshot.context.articles
            ],
        )
        resolution_map = {
            entry.reference_id: _restore_article(entry.article)
            for entry in prepared_snapshot.resolution
        }
        if any(
            reference_cluster_ids.get(entry.reference_id)
            != entry.owning_cluster_id
            for entry in prepared_snapshot.resolution
        ):
            raise ValueError("resolved reference ownership changed")
        prepared_clusters.append(
            PreparedAudienceCluster(
                cluster=cluster,
                context=context,
                cluster_id=prepared_snapshot.cluster_id,
                cluster_pageviews=prepared_snapshot.cluster_pageviews,
                evidence_reference_ids=(
                    prepared_snapshot.evidence_reference_ids
                ),
                resolution_map=resolution_map,
            )
        )
    return AudiencePreparation(
        clusters=tuple(prepared_clusters),
        total_analyzed_views=snapshot.total_analyzed_views,
        reference_cluster_ids=reference_cluster_ids,
    )


def _restore_article(article: ReviewArticleSnapshot) -> Article:
    return Article(
        title=article.title,
        normalized_title=article.normalized_title,
        url=article.url,
        weekly_views=article.weekly_views,
        daily_views={entry.day: entry.pageviews for entry in article.daily_views},
        summary=article.summary,
        analysis_start_date=article.analysis_start_date,
        analysis_end_date=article.analysis_end_date,
    )


def _snapshot_article(article: Article) -> ReviewArticleSnapshot:
    return ReviewArticleSnapshot(
        title=article.title,
        normalized_title=article.normalized_title,
        url=article.url,
        weekly_views=article.weekly_views,
        daily_views=tuple(
            ReviewDailyViewSnapshot(day=day, pageviews=pageviews)
            for day, pageviews in sorted(article.daily_views.items())
        ),
        summary=article.summary,
        analysis_start_date=article.analysis_start_date,
        analysis_end_date=article.analysis_end_date,
    )


def _snapshot_metrics(workflow: AudienceWorkflowResult) -> ReviewWorkflowMetricsSnapshot:
    values = {
        field.name: getattr(workflow.metrics, field.name)
        for field in fields(workflow.metrics)
    }
    values["validation_issue_counts_by_code"] = tuple(
        ReviewCodeCountSnapshot(code=code, count=count)
        for code, count in sorted(
            workflow.metrics.validation_issue_counts_by_code.items()
        )
    )
    values["drop_counts_by_code"] = tuple(
        ReviewCodeCountSnapshot(code=code, count=count)
        for code, count in sorted(workflow.metrics.drop_counts_by_code.items())
    )
    return ReviewWorkflowMetricsSnapshot.model_validate(values)


def _empty_metrics() -> ReviewWorkflowMetricsSnapshot:
    return ReviewWorkflowMetricsSnapshot(
        initial_decision_count=0,
        initial_valid_decision_count=0,
        initial_invalid_report_count=0,
        revision_count=0,
        revision_requested_cluster_count=0,
        revision_decision_count=0,
        revision_valid_decision_count=0,
        final_valid_decision_count=0,
        final_segment_count=0,
        final_provider_skip_count=0,
        dropped_source_cluster_count=0,
        dropped_unmatched_decision_count=0,
        provider_call_count=0,
        provider_input_tokens=0,
        provider_output_tokens=0,
        provider_total_tokens=0,
        provider_elapsed_seconds=0,
        validation_issue_count=0,
        validation_issue_counts_by_code=(),
        drop_counts_by_code=(),
    )
