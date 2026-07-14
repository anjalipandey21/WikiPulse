"""End-to-end orchestration for WikiPulse audience analysis."""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from types import MappingProxyType

from .agent.audience_finalization import (
    AudiencePreparation,
    ProviderSkippedCluster,
    prepare_audience_clusters,
)
from .agent.audience_provider import AudienceGenerationProvider
from .agent.audience_workflow import (
    AudienceWorkflowResult,
    DroppedAudienceDecision,
    run_audience_workflow,
)
from .clustering.keyword_extraction import DEFAULT_TOP_K
from .clustering.semantic_clustering import (
    DEFAULT_SIMILARITY_THRESHOLD,
    MIN_CLUSTER_SIZE,
    ArticleEncoder,
)
from .filtering.commercial_safety import (
    CommercialSafetyResult,
    SkippedTopicCluster,
    route_commercial_clusters,
)
from .models import AudienceSegment
from .progress import AnalysisProgressReporter, report_progress
from .topic_analysis import (
    DEFAULT_TOP_N,
    PageviewClient,
    SummaryClient,
    TopicAnalysisResult,
    analyze_topics,
)


ROUTING_PARTITION_MISMATCH = "routing_partition_mismatch"
PREPARATION_PARTITION_MISMATCH = "preparation_partition_mismatch"
WORKFLOW_PARTITION_MISMATCH = "workflow_partition_mismatch"


class AudienceAnalysisInvariantError(RuntimeError):
    """Raised when one stage changes the cross-stage cluster partition."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class AudienceAnalysisMetrics:
    """Immutable deterministic funnel metrics across audience-analysis stages."""

    topic_cluster_count: int
    commercial_eligible_cluster_count: int
    commercial_skipped_cluster_count: int
    prepared_cluster_count: int
    final_segment_count: int
    provider_skipped_cluster_count: int
    validation_dropped_source_cluster_count: int
    unmatched_provider_output_count: int
    commercial_eligible_pageviews: int
    represented_audience_pageviews: int
    commercial_skip_counts_by_reason: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "commercial_skip_counts_by_reason",
            MappingProxyType(dict(self.commercial_skip_counts_by_reason)),
        )


@dataclass(frozen=True, slots=True)
class AudienceAnalysisResult:
    """Each retained stage result plus cross-stage deterministic metrics."""

    topic_analysis: TopicAnalysisResult
    commercial_routing: CommercialSafetyResult
    preparation: AudiencePreparation
    audience_workflow: AudienceWorkflowResult
    metrics: AudienceAnalysisMetrics

    @property
    def segments(self) -> tuple[AudienceSegment, ...]:
        """Return final segments without copying the workflow output."""
        return self.audience_workflow.segments

    @property
    def commercial_skips(self) -> tuple[SkippedTopicCluster, ...]:
        """Return deterministic commercial-routing exclusions."""
        return self.commercial_routing.skipped_clusters

    @property
    def provider_skips(self) -> tuple[ProviderSkippedCluster, ...]:
        """Return accepted provider skip decisions."""
        return self.audience_workflow.provider_skips

    @property
    def dropped_decisions(self) -> tuple[DroppedAudienceDecision, ...]:
        """Return explicit validation drops after bounded revision."""
        return self.audience_workflow.dropped_decisions

    @property
    def is_publishable(self) -> bool:
        """Return whether the bounded workflow produced publishable output."""
        return self.audience_workflow.is_publishable


async def analyze_audiences(
    pageview_client: PageviewClient,
    summary_client: SummaryClient,
    encoder: ArticleEncoder,
    audience_provider: AudienceGenerationProvider,
    *,
    today_utc: date | None = None,
    top_n: int = DEFAULT_TOP_N,
    keyword_top_k: int = DEFAULT_TOP_K,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    progress_reporter: AnalysisProgressReporter | None = None,
) -> AudienceAnalysisResult:
    """Compose topic analysis, routing, preparation, and bounded generation."""
    topic_arguments = {
        "today_utc": today_utc,
        "top_n": top_n,
        "keyword_top_k": keyword_top_k,
        "similarity_threshold": similarity_threshold,
        "min_cluster_size": min_cluster_size,
    }
    if progress_reporter is None:
        topic_result = await analyze_topics(
            pageview_client,
            summary_client,
            encoder,
            **topic_arguments,
        )
    else:
        topic_result = await analyze_topics(
            pageview_client,
            summary_client,
            encoder,
            progress_reporter=progress_reporter,
            **topic_arguments,
        )
    await report_progress(progress_reporter, "routing_commercial_clusters")
    routing_result = route_commercial_clusters(topic_result.topics)
    _validate_routing_partition(topic_result, routing_result)

    await report_progress(progress_reporter, "preparing_audience_evidence")
    preparation = prepare_audience_clusters(
        routing_result.eligible_clusters,
        total_analyzed_views=topic_result.metrics.selected_pageviews,
    )
    _validate_preparation_partition(
        topic_result,
        routing_result,
        preparation,
    )

    if progress_reporter is None:
        workflow_result = await run_audience_workflow(
            preparation,
            audience_provider,
        )
    else:
        workflow_result = await run_audience_workflow(
            preparation,
            audience_provider,
            progress_reporter=progress_reporter,
        )
    segment_cluster_ids = _validate_workflow_partition(
        preparation,
        workflow_result,
    )
    metrics = _build_metrics(
        topic_result,
        routing_result,
        preparation,
        workflow_result,
        segment_cluster_ids,
    )
    return AudienceAnalysisResult(
        topic_analysis=topic_result,
        commercial_routing=routing_result,
        preparation=preparation,
        audience_workflow=workflow_result,
        metrics=metrics,
    )


def _validate_routing_partition(
    topic_result: TopicAnalysisResult,
    routing_result: CommercialSafetyResult,
) -> None:
    topics_by_id = {cluster.id: cluster for cluster in topic_result.topics}
    routed_clusters = (
        routing_result.eligible_clusters
        + tuple(skipped.cluster for skipped in routing_result.skipped_clusters)
    )
    routed_ids = tuple(cluster.id for cluster in routed_clusters)

    if (
        len(topics_by_id) != len(topic_result.topics)
        or len(routed_clusters) != len(topic_result.topics)
        or len(set(routed_ids)) != len(routed_ids)
        or set(routed_ids) != set(topics_by_id)
        or any(topics_by_id[cluster.id] is not cluster for cluster in routed_clusters)
    ):
        raise AudienceAnalysisInvariantError(ROUTING_PARTITION_MISMATCH)


def _validate_preparation_partition(
    topic_result: TopicAnalysisResult,
    routing_result: CommercialSafetyResult,
    preparation: AudiencePreparation,
) -> None:
    eligible_ids = tuple(cluster.id for cluster in routing_result.eligible_clusters)
    prepared_ids = tuple(prepared.cluster_id for prepared in preparation.clusters)
    if (
        preparation.total_analyzed_views
        != topic_result.metrics.selected_pageviews
        or len(preparation.clusters) != len(routing_result.eligible_clusters)
        or prepared_ids != eligible_ids
        or any(
            prepared.cluster is not cluster
            for prepared, cluster in zip(
                preparation.clusters,
                routing_result.eligible_clusters,
                strict=True,
            )
        )
    ):
        raise AudienceAnalysisInvariantError(PREPARATION_PARTITION_MISMATCH)


def _validate_workflow_partition(
    preparation: AudiencePreparation,
    workflow_result: AudienceWorkflowResult,
) -> tuple[str, ...]:
    prepared_ids = tuple(prepared.cluster_id for prepared in preparation.clusters)
    prepared_ids_by_identity = {
        id(prepared.cluster): prepared.cluster_id for prepared in preparation.clusters
    }
    segment_ids = _segment_cluster_ids(workflow_result.segments)

    provider_skip_ids: list[str] = []
    for skipped in workflow_result.provider_skips:
        cluster_id = prepared_ids_by_identity.get(id(skipped.cluster))
        if cluster_id is None:
            raise AudienceAnalysisInvariantError(WORKFLOW_PARTITION_MISMATCH)
        provider_skip_ids.append(cluster_id)

    source_drop_ids: list[str] = []
    unmatched_drop_count = 0
    seen_unmatched_drop = False
    for dropped in workflow_result.dropped_decisions:
        if dropped.source_cluster is None:
            seen_unmatched_drop = True
            unmatched_drop_count += 1
            continue
        if seen_unmatched_drop:
            raise AudienceAnalysisInvariantError(WORKFLOW_PARTITION_MISMATCH)
        cluster_id = prepared_ids_by_identity.get(id(dropped.source_cluster))
        if cluster_id is None or dropped.cluster_id != cluster_id:
            raise AudienceAnalysisInvariantError(WORKFLOW_PARTITION_MISMATCH)
        source_drop_ids.append(cluster_id)

    final_source_ids = segment_ids + tuple(provider_skip_ids) + tuple(source_drop_ids)
    if (
        len(final_source_ids) != len(prepared_ids)
        or len(set(final_source_ids)) != len(final_source_ids)
        or set(final_source_ids) != set(prepared_ids)
        or not _is_in_preparation_order(segment_ids, prepared_ids)
        or not _is_in_preparation_order(provider_skip_ids, prepared_ids)
        or not _is_in_preparation_order(source_drop_ids, prepared_ids)
        or workflow_result.metrics.final_segment_count
        != len(workflow_result.segments)
        or workflow_result.metrics.final_provider_skip_count
        != len(workflow_result.provider_skips)
        or workflow_result.metrics.dropped_source_cluster_count
        != len(source_drop_ids)
        or workflow_result.metrics.dropped_unmatched_decision_count
        != unmatched_drop_count
        or workflow_result.metrics.final_valid_decision_count
        != len(workflow_result.segments) + len(workflow_result.provider_skips)
    ):
        raise AudienceAnalysisInvariantError(WORKFLOW_PARTITION_MISMATCH)

    return segment_ids


def _segment_cluster_ids(
    segments: Sequence[AudienceSegment],
) -> tuple[str, ...]:
    cluster_ids: list[str] = []
    for segment in segments:
        if len(segment.topic_cluster_ids) != 1:
            raise AudienceAnalysisInvariantError(WORKFLOW_PARTITION_MISMATCH)
        cluster_ids.append(segment.topic_cluster_ids[0])
    return tuple(cluster_ids)


def _is_in_preparation_order(
    cluster_ids: Sequence[str],
    prepared_ids: Sequence[str],
) -> bool:
    ranks = {cluster_id: index for index, cluster_id in enumerate(prepared_ids)}
    try:
        return list(cluster_ids) == sorted(cluster_ids, key=ranks.__getitem__)
    except KeyError:
        return False


def _build_metrics(
    topic_result: TopicAnalysisResult,
    routing_result: CommercialSafetyResult,
    preparation: AudiencePreparation,
    workflow_result: AudienceWorkflowResult,
    segment_cluster_ids: Sequence[str],
) -> AudienceAnalysisMetrics:
    skip_counts = Counter(
        skipped.reason for skipped in routing_result.skipped_clusters
    )
    prepared_by_id = {
        prepared.cluster_id: prepared for prepared in preparation.clusters
    }
    return AudienceAnalysisMetrics(
        topic_cluster_count=len(topic_result.topics),
        commercial_eligible_cluster_count=len(routing_result.eligible_clusters),
        commercial_skipped_cluster_count=len(routing_result.skipped_clusters),
        prepared_cluster_count=len(preparation.clusters),
        final_segment_count=len(workflow_result.segments),
        provider_skipped_cluster_count=len(workflow_result.provider_skips),
        validation_dropped_source_cluster_count=(
            workflow_result.metrics.dropped_source_cluster_count
        ),
        unmatched_provider_output_count=(
            workflow_result.metrics.dropped_unmatched_decision_count
        ),
        commercial_eligible_pageviews=sum(
            prepared.cluster_pageviews for prepared in preparation.clusters
        ),
        represented_audience_pageviews=sum(
            prepared_by_id[cluster_id].cluster_pageviews
            for cluster_id in segment_cluster_ids
        ),
        commercial_skip_counts_by_reason={
            reason: skip_counts[reason] for reason in sorted(skip_counts)
        },
    )
