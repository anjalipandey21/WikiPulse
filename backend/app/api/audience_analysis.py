"""FastAPI endpoint and public mapping for complete audience analysis."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass
import logging

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from ..agent.audience_finalization import ProviderSkippedCluster
from ..agent.audience_provider import AudienceGenerationProvider
from ..agent.audience_trace import (
    AudienceDecisionTrace,
    AudienceTraceInvariantError,
    build_audience_decision_traces,
)
from ..audience_analysis import AudienceAnalysisResult, analyze_audiences
from ..clustering.semantic_clustering import ArticleEncoder
from ..models import Article, AudienceSegment, TopicCluster
from ..models.audience_api import (
    ApiErrorResponse,
    ApiErrorDetailResponse,
    ArticleResponse,
    AudienceAnalysisMetricsResponse,
    AudienceAnalysisResponse,
    AudienceAnalysisErrorEvent,
    AudienceAnalysisProgressEvent,
    AudienceAnalysisResultEvent,
    AudienceAnalysisStreamEvent,
    AudienceDecisionTraceResponse,
    AudienceDecisionIssueResponse,
    AudienceFunnelMetricsResponse,
    AudienceSegmentResponse,
    AudienceTraceEventResponse,
    AudienceWorkflowMetricsResponse,
    CommercialSkippedClusterResponse,
    DroppedAudienceDecisionResponse,
    ProviderSkippedClusterResponse,
    RejectedArticleResponse,
    TopicAnalysisMetricsResponse,
    TopicClusterResponse,
)
from ..progress import AnalysisProgressStage
from ..topic_analysis import PageviewClient, SummaryClient
from .analysis_errors import classify_analysis_exception


logger = logging.getLogger(__name__)

STREAM_QUEUE_MAXSIZE = 16
STREAM_MEDIA_TYPE = "application/x-ndjson"
_STREAM_END = object()


@dataclass(frozen=True, slots=True)
class AudienceAnalysisResources:
    """Application-owned dependencies reused across analysis requests."""

    pageview_client: PageviewClient
    summary_client: SummaryClient
    encoder: ArticleEncoder
    audience_provider: AudienceGenerationProvider
    analysis_lock: asyncio.Lock


router = APIRouter(prefix="/api", tags=["audience-analysis"])


def get_audience_analysis_resources(
    request: Request,
) -> AudienceAnalysisResources:
    """Resolve lifespan-managed dependencies from application state."""
    resources = getattr(
        request.app.state,
        "audience_analysis_resources",
        None,
    )
    if resources is None:
        raise RuntimeError("audience analysis resources are unavailable")
    return resources


@router.post(
    "/audience-analysis",
    response_model=AudienceAnalysisResponse,
    responses={
        422: {"model": ApiErrorResponse},
        500: {"model": ApiErrorResponse},
        502: {"model": ApiErrorResponse},
    },
)
async def run_audience_analysis(
    resources: AudienceAnalysisResources = Depends(
        get_audience_analysis_resources
    ),
) -> AudienceAnalysisResponse:
    """Run one serialized complete WikiPulse audience analysis."""
    async with resources.analysis_lock:
        result = await analyze_audiences(
            resources.pageview_client,
            resources.summary_client,
            resources.encoder,
            resources.audience_provider,
        )
    return map_audience_analysis_response(result)


@router.post(
    "/audience-analysis/stream",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "NDJSON progress followed by one terminal event.",
            "content": {STREAM_MEDIA_TYPE: {}},
        },
        422: {"model": ApiErrorResponse},
        500: {"model": ApiErrorResponse},
        502: {"model": ApiErrorResponse},
    },
)
async def stream_audience_analysis(
    resources: AudienceAnalysisResources = Depends(
        get_audience_analysis_resources
    ),
) -> StreamingResponse:
    """Stream safe stage codes and one exact terminal analysis response."""
    return StreamingResponse(
        _stream_analysis_events(resources),
        media_type=STREAM_MEDIA_TYPE,
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


class _StreamEventEmitter:
    """Assign request-local sequences and write to one bounded queue."""

    def __init__(
        self,
        queue: asyncio.Queue[AudienceAnalysisStreamEvent | object],
    ) -> None:
        self._queue = queue
        self._sequence = 0

    async def progress(self, stage: AnalysisProgressStage) -> None:
        await self._put(
            AudienceAnalysisProgressEvent(
                sequence=self._next_sequence(),
                stage=stage,
            )
        )

    async def result(self, result: AudienceAnalysisResponse) -> None:
        await self._put(
            AudienceAnalysisResultEvent(
                sequence=self._next_sequence(),
                result=result,
            )
        )

    async def error(
        self,
        *,
        status_code: int,
        code: str,
        message: str,
    ) -> None:
        await self._put(
            AudienceAnalysisErrorEvent(
                sequence=self._next_sequence(),
                status_code=status_code,
                error=ApiErrorDetailResponse(code=code, message=message),
            )
        )

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    async def _put(self, event: AudienceAnalysisStreamEvent) -> None:
        await self._queue.put(event)


async def _stream_analysis_events(
    resources: AudienceAnalysisResources,
) -> AsyncIterator[bytes]:
    queue: asyncio.Queue[AudienceAnalysisStreamEvent | object] = asyncio.Queue(
        maxsize=STREAM_QUEUE_MAXSIZE
    )
    emitter = _StreamEventEmitter(queue)
    producer = asyncio.create_task(
        _produce_analysis_events(resources, emitter, queue)
    )
    try:
        while True:
            item = await queue.get()
            if item is _STREAM_END:
                break
            if not isinstance(
                item,
                (
                    AudienceAnalysisProgressEvent,
                    AudienceAnalysisResultEvent,
                    AudienceAnalysisErrorEvent,
                ),
            ):
                continue
            yield f"{item.model_dump_json()}\n".encode("utf-8")
            if isinstance(
                item,
                (AudienceAnalysisResultEvent, AudienceAnalysisErrorEvent),
            ):
                break
    finally:
        if not producer.done():
            producer.cancel()
        with suppress(asyncio.CancelledError):
            await producer


async def _produce_analysis_events(
    resources: AudienceAnalysisResources,
    emitter: _StreamEventEmitter,
    queue: asyncio.Queue[AudienceAnalysisStreamEvent | object],
) -> None:
    try:
        await emitter.progress("waiting_for_slot")
        async with resources.analysis_lock:
            result = await analyze_audiences(
                resources.pageview_client,
                resources.summary_client,
                resources.encoder,
                resources.audience_provider,
                progress_reporter=emitter.progress,
            )
        await emitter.progress("assembling_response")
        response = map_audience_analysis_response(result)
        await emitter.result(response)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        public_error = classify_analysis_exception(exc)
        logger.log(
            logging.WARNING
            if public_error.status_code < 500
            or public_error.status_code == 502
            else logging.ERROR,
            "Streaming audience analysis failed: %s",
            type(exc).__name__,
        )
        await emitter.error(
            status_code=public_error.status_code,
            code=public_error.code,
            message=public_error.message,
        )
    finally:
        try:
            queue.put_nowait(_STREAM_END)
        except asyncio.QueueFull:
            pass


def map_audience_analysis_response(
    result: AudienceAnalysisResult,
) -> AudienceAnalysisResponse:
    """Map internal stage results to the explicit public API contract."""
    trace_projection = build_audience_decision_traces(
        result.preparation,
        result.audience_workflow,
    )
    traces_by_id = {
        trace.trace_id: trace for trace in trace_projection.traces
    }
    topic_metrics = result.topic_analysis.metrics
    funnel_metrics = result.metrics
    workflow_metrics = result.audience_workflow.metrics

    return AudienceAnalysisResponse(
        topics=[_map_topic(cluster) for cluster in result.topic_analysis.topics],
        audience_segments=[
            _map_segment(segment, trace_id)
            for segment, trace_id in zip(
                result.segments,
                trace_projection.segment_trace_ids,
                strict=True,
            )
        ],
        unclustered_articles=[
            _map_article(article)
            for article in result.topic_analysis.unclustered_articles
        ],
        rejected_articles=[
            RejectedArticleResponse(
                article=_map_article(rejected.article),
                reason=rejected.reason,
            )
            for rejected in result.topic_analysis.rejected_articles
        ],
        commercial_skips=[
            CommercialSkippedClusterResponse(
                cluster_id=skipped.cluster.id,
                cluster_name=skipped.cluster.name,
                reason=skipped.reason,
            )
            for skipped in result.commercial_skips
        ],
        provider_skips=[
            _map_provider_skip(
                skipped,
                traces_by_id[trace_id],
            )
            for skipped, trace_id in zip(
                result.provider_skips,
                trace_projection.provider_skip_trace_ids,
                strict=True,
            )
        ],
        validation_drops=[
            DroppedAudienceDecisionResponse(
                trace_id=trace_id,
                cluster_id=dropped.cluster_id,
                source_known=dropped.source_cluster is not None,
                phase=dropped.phase,
                drop_code=dropped.drop_code,
                issues=[
                    AudienceDecisionIssueResponse(
                        code=issue.code,
                        reference_id=issue.reference_id,
                    )
                    for issue in dropped.issues
                ],
            )
            for dropped, trace_id in zip(
                result.dropped_decisions,
                trace_projection.drop_trace_ids,
                strict=True,
            )
        ],
        audience_traces=[
            _map_trace(trace) for trace in trace_projection.traces
        ],
        is_publishable=result.is_publishable,
        metrics=AudienceAnalysisMetricsResponse(
            topic_analysis=TopicAnalysisMetricsResponse(
                fetched_article_count=topic_metrics.fetched_article_count,
                rejected_article_count=topic_metrics.rejected_article_count,
                eligible_article_count=topic_metrics.eligible_article_count,
                top_n_omitted_article_count=(
                    topic_metrics.top_n_omitted_article_count
                ),
                selected_article_count=topic_metrics.selected_article_count,
                summary_available_article_count=(
                    topic_metrics.summary_available_article_count
                ),
                summary_missing_article_count=(
                    topic_metrics.summary_missing_article_count
                ),
                topic_cluster_count=topic_metrics.topic_cluster_count,
                clustered_article_count=topic_metrics.clustered_article_count,
                unclustered_article_count=(
                    topic_metrics.unclustered_article_count
                ),
                selected_pageviews=topic_metrics.selected_pageviews,
            ),
            audience_funnel=AudienceFunnelMetricsResponse(
                topic_cluster_count=funnel_metrics.topic_cluster_count,
                commercial_eligible_cluster_count=(
                    funnel_metrics.commercial_eligible_cluster_count
                ),
                commercial_skipped_cluster_count=(
                    funnel_metrics.commercial_skipped_cluster_count
                ),
                prepared_cluster_count=funnel_metrics.prepared_cluster_count,
                final_segment_count=funnel_metrics.final_segment_count,
                provider_skipped_cluster_count=(
                    funnel_metrics.provider_skipped_cluster_count
                ),
                validation_dropped_source_cluster_count=(
                    funnel_metrics.validation_dropped_source_cluster_count
                ),
                unmatched_provider_output_count=(
                    funnel_metrics.unmatched_provider_output_count
                ),
                commercial_eligible_pageviews=(
                    funnel_metrics.commercial_eligible_pageviews
                ),
                represented_audience_pageviews=(
                    funnel_metrics.represented_audience_pageviews
                ),
                commercial_skip_counts_by_reason=dict(
                    funnel_metrics.commercial_skip_counts_by_reason
                ),
            ),
            workflow=AudienceWorkflowMetricsResponse(
                initial_decision_count=workflow_metrics.initial_decision_count,
                initial_valid_decision_count=(
                    workflow_metrics.initial_valid_decision_count
                ),
                initial_invalid_report_count=(
                    workflow_metrics.initial_invalid_report_count
                ),
                revision_count=workflow_metrics.revision_count,
                revision_requested_cluster_count=(
                    workflow_metrics.revision_requested_cluster_count
                ),
                revision_decision_count=workflow_metrics.revision_decision_count,
                revision_valid_decision_count=(
                    workflow_metrics.revision_valid_decision_count
                ),
                final_valid_decision_count=(
                    workflow_metrics.final_valid_decision_count
                ),
                final_segment_count=workflow_metrics.final_segment_count,
                final_provider_skip_count=(
                    workflow_metrics.final_provider_skip_count
                ),
                dropped_source_cluster_count=(
                    workflow_metrics.dropped_source_cluster_count
                ),
                dropped_unmatched_decision_count=(
                    workflow_metrics.dropped_unmatched_decision_count
                ),
                provider_call_count=workflow_metrics.provider_call_count,
                provider_input_tokens=workflow_metrics.provider_input_tokens,
                provider_output_tokens=workflow_metrics.provider_output_tokens,
                provider_total_tokens=workflow_metrics.provider_total_tokens,
                provider_elapsed_seconds=(
                    workflow_metrics.provider_elapsed_seconds
                ),
                validation_issue_count=workflow_metrics.validation_issue_count,
                validation_issue_counts_by_code=dict(
                    workflow_metrics.validation_issue_counts_by_code
                ),
                drop_counts_by_code=dict(workflow_metrics.drop_counts_by_code),
            ),
        ),
    )


def _map_article(article: Article) -> ArticleResponse:
    return ArticleResponse(
        title=article.title,
        normalized_title=article.normalized_title,
        url=article.url,
        weekly_views=article.weekly_views,
        daily_views=dict(article.daily_views),
        summary=article.summary,
        analysis_start_date=article.analysis_start_date,
        analysis_end_date=article.analysis_end_date,
    )


def _map_topic(cluster: TopicCluster) -> TopicClusterResponse:
    return TopicClusterResponse(
        id=cluster.id,
        name=cluster.name,
        description=cluster.description,
        articles=[_map_article(article) for article in cluster.articles],
        keywords=list(cluster.keywords),
        total_views=cluster.total_views,
        article_count=cluster.article_count,
        confidence_score=cluster.confidence_score,
    )


def _map_segment(
    segment: AudienceSegment,
    trace_id: str,
) -> AudienceSegmentResponse:
    return AudienceSegmentResponse(
        trace_id=trace_id,
        id=segment.id,
        name=segment.name,
        description=segment.description,
        topic_cluster_ids=list(segment.topic_cluster_ids),
        size_index=segment.size_index,
        buying_power=segment.buying_power,
        buying_power_reason=segment.buying_power_reason,
        brand_categories=list(segment.brand_categories),
        supporting_articles=[
            _map_article(article) for article in segment.supporting_articles
        ],
        commercial_confidence=segment.commercial_confidence,
        commercial_confidence_reason=segment.commercial_confidence_reason,
    )


def _map_provider_skip(
    skipped: ProviderSkippedCluster,
    trace: AudienceDecisionTrace,
) -> ProviderSkippedClusterResponse:
    if not trace.source_known or trace.cluster_name is None:
        raise AudienceTraceInvariantError(
            "provider skip must map to a known trace source"
        )
    return ProviderSkippedClusterResponse(
        trace_id=trace.trace_id,
        cluster_id=trace.cluster_id,
        cluster_name=trace.cluster_name,
        reason=skipped.reason,
    )


def _map_trace(
    trace: AudienceDecisionTrace,
) -> AudienceDecisionTraceResponse:
    return AudienceDecisionTraceResponse(
        trace_id=trace.trace_id,
        cluster_id=trace.cluster_id,
        cluster_name=trace.cluster_name,
        source_known=trace.source_known,
        final_outcome=trace.final_outcome,
        events=[
            AudienceTraceEventResponse(
                sequence=event.sequence,
                phase=event.phase,
                code=event.code,
                outcome_code=event.outcome_code,
                issues=[
                    AudienceDecisionIssueResponse(
                        code=issue.code,
                        reference_id=issue.reference_id,
                    )
                    for issue in event.issues
                ],
            )
            for event in trace.events
        ],
    )
