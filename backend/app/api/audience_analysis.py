"""FastAPI endpoint and public mapping for complete audience analysis."""

import asyncio
from dataclasses import dataclass

from fastapi import APIRouter, Depends, Request

from ..agent.audience_provider import AudienceGenerationProvider
from ..audience_analysis import AudienceAnalysisResult, analyze_audiences
from ..clustering.semantic_clustering import ArticleEncoder
from ..models import Article, AudienceSegment, TopicCluster
from ..models.audience_api import (
    ApiErrorResponse,
    ArticleResponse,
    AudienceAnalysisMetricsResponse,
    AudienceAnalysisResponse,
    AudienceDecisionIssueResponse,
    AudienceFunnelMetricsResponse,
    AudienceSegmentResponse,
    AudienceWorkflowMetricsResponse,
    CommercialSkippedClusterResponse,
    DroppedAudienceDecisionResponse,
    ProviderSkippedClusterResponse,
    RejectedArticleResponse,
    TopicAnalysisMetricsResponse,
    TopicClusterResponse,
)
from ..topic_analysis import PageviewClient, SummaryClient


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


def map_audience_analysis_response(
    result: AudienceAnalysisResult,
) -> AudienceAnalysisResponse:
    """Map internal stage results to the explicit public API contract."""
    prepared_ids_by_identity = {
        id(prepared.cluster): prepared.cluster_id
        for prepared in result.preparation.clusters
    }
    topic_metrics = result.topic_analysis.metrics
    funnel_metrics = result.metrics
    workflow_metrics = result.audience_workflow.metrics

    return AudienceAnalysisResponse(
        topics=[_map_topic(cluster) for cluster in result.topic_analysis.topics],
        audience_segments=[
            _map_segment(segment) for segment in result.segments
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
            ProviderSkippedClusterResponse(
                cluster_id=prepared_ids_by_identity[id(skipped.cluster)],
                cluster_name=skipped.cluster.name,
                reason=skipped.reason,
            )
            for skipped in result.provider_skips
        ],
        validation_drops=[
            DroppedAudienceDecisionResponse(
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
            for dropped in result.dropped_decisions
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


def _map_segment(segment: AudienceSegment) -> AudienceSegmentResponse:
    return AudienceSegmentResponse(
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
