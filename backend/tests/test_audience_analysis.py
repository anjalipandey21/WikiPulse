"""Focused tests for complete audience-analysis orchestration."""

from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import date
import unittest
from unittest.mock import AsyncMock, patch

from app.agent.audience_finalization import (
    AudienceSourceIntegrityError,
    CLUSTER_VIEWS_EXCEED_TOTAL,
)
from app.agent.audience_provider import (
    AudienceProviderError,
    AudienceProviderResult,
    AudienceRevisionRequest,
    AudienceTokenUsage,
)
from app.agent.audience_workflow import (
    REVISION_PROVIDER_FAILURE,
    AudienceWorkflowResult,
    run_audience_workflow,
)
from app.audience_analysis import (
    ROUTING_PARTITION_MISMATCH,
    WORKFLOW_PARTITION_MISMATCH,
    AudienceAnalysisInvariantError,
    analyze_audiences,
    prepare_audience_analysis,
)
from app.filtering.commercial_safety import (
    LOW_COHESION_REASON,
    CommercialSafetyResult,
)
from app.models import Article, TopicCluster
from app.models.audience_generation import (
    AudienceGenerationResponse,
    CompactClusterContext,
    CreateAudienceDecision,
    SkipClusterDecision,
)
from app.progress import AnalysisProgressStage
from app.topic_analysis import TopicAnalysisMetrics, TopicAnalysisResult


def make_article(title: str, views: int) -> Article:
    observed_date = date(2026, 7, 12)
    return Article(
        title=title,
        normalized_title=title,
        url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
        weekly_views=views,
        daily_views={observed_date: views},
        summary=f"{title} is a useful source article for this topic.",
        analysis_start_date=date(2026, 7, 6),
        analysis_end_date=observed_date,
    )


def make_cluster(
    cluster_id: str,
    views: tuple[int, int],
    *,
    confidence: float = 0.8,
) -> TopicCluster:
    articles = [
        make_article(f"{cluster_id} Alpha", views[0]),
        make_article(f"{cluster_id} Beta", views[1]),
    ]
    return TopicCluster(
        id=cluster_id,
        name=f"Topic {cluster_id}",
        articles=articles,
        keywords=[cluster_id, "example topic"],
        total_views=sum(views),
        article_count=2,
        confidence_score=confidence,
    )


def make_topic_result(
    clusters: Sequence[TopicCluster],
    *,
    selected_pageviews: int,
) -> TopicAnalysisResult:
    selected_article_count = sum(len(cluster.articles) for cluster in clusters)
    return TopicAnalysisResult(
        topics=tuple(clusters),
        unclustered_articles=(),
        rejected_articles=(),
        metrics=TopicAnalysisMetrics(
            fetched_article_count=selected_article_count,
            rejected_article_count=0,
            eligible_article_count=selected_article_count,
            top_n_omitted_article_count=0,
            selected_article_count=selected_article_count,
            summary_available_article_count=selected_article_count,
            summary_missing_article_count=0,
            topic_cluster_count=len(clusters),
            clustered_article_count=selected_article_count,
            unclustered_article_count=0,
            selected_pageviews=selected_pageviews,
        ),
    )


def make_create_decision(cluster_id: str) -> CreateAudienceDecision:
    return CreateAudienceDecision(
        decision="create_audience",
        cluster_id=cluster_id,
        name=f"{cluster_id.title()} Followers",
        description=(
            "People following this coherent topic and its related developments."
        ),
        supporting_article_reference_ids=[
            f"{cluster_id}:a0",
            f"{cluster_id}:a1",
        ],
        buying_power="medium",
        buying_power_reason=(
            "The audience includes broad consumer groups with repeat spending."
        ),
        brand_categories=["Media", "Consumer technology"],
        commercial_confidence=0.76,
        commercial_confidence_reason=(
            "The source articles provide coherent commercial evidence."
        ),
    )


def make_skip_decision(cluster_id: str) -> SkipClusterDecision:
    return SkipClusterDecision(
        decision="skip_cluster",
        cluster_id=cluster_id,
        reason=(
            "The source topic does not support a sufficiently specific audience."
        ),
    )


def make_provider_result(
    response: AudienceGenerationResponse,
    *,
    phase: str,
) -> AudienceProviderResult:
    return AudienceProviderResult(
        response=response,
        model="mock-model",
        response_id=f"response-{phase}",
        elapsed_seconds=0.1,
        usage=AudienceTokenUsage(10, 5, 15),
    )


class FakeAudienceProvider:
    def __init__(
        self,
        initial_response: AudienceGenerationResponse | None,
        *,
        initial_error: AudienceProviderError | None = None,
        revision_error: AudienceProviderError | None = None,
        on_generate: Callable[[], None] | None = None,
    ) -> None:
        self.initial_result = (
            make_provider_result(initial_response, phase="initial")
            if initial_response is not None
            else None
        )
        self.initial_error = initial_error
        self.revision_error = revision_error
        self.on_generate = on_generate
        self.generate_call_count = 0
        self.revise_call_count = 0
        self.generated_contexts: tuple[CompactClusterContext, ...] | None = None

    async def generate(
        self,
        cluster_contexts: Sequence[CompactClusterContext],
    ) -> AudienceProviderResult:
        self.generate_call_count += 1
        self.generated_contexts = tuple(cluster_contexts)
        if self.on_generate is not None:
            self.on_generate()
        if self.initial_error is not None:
            raise self.initial_error
        if self.initial_result is None:
            raise AssertionError("generate should not have been called")
        return self.initial_result

    async def revise(
        self,
        revision_requests: Sequence[AudienceRevisionRequest],
    ) -> AudienceProviderResult:
        self.revise_call_count += 1
        if self.revision_error is not None:
            raise self.revision_error
        raise AssertionError("revision should not have been called")


class AudienceAnalysisTests(unittest.IsolatedAsyncioTestCase):
    async def test_standard_and_review_modes_share_identical_preparation(self) -> None:
        cluster = make_cluster("eligible", (400, 300))
        topic_result = make_topic_result([cluster], selected_pageviews=700)
        provider = FakeAudienceProvider(
            AudienceGenerationResponse(
                decisions=[make_create_decision("eligible")]
            )
        )

        with patch(
            "app.audience_analysis.analyze_topics",
            new=AsyncMock(side_effect=[topic_result, topic_result]),
        ):
            review_input = await prepare_audience_analysis(
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
            )
            standard = await analyze_audiences(
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                provider,
            )

        self.assertEqual(review_input.topic_analysis, standard.topic_analysis)
        self.assertEqual(
            review_input.commercial_routing,
            standard.commercial_routing,
        )
        self.assertEqual(review_input.preparation, standard.preparation)

    async def test_reports_routing_preparation_and_graph_boundaries(self) -> None:
        cluster = make_cluster("eligible", (400, 300))
        topic_result = make_topic_result(
            [cluster],
            selected_pageviews=700,
        )
        provider = FakeAudienceProvider(
            AudienceGenerationResponse(
                decisions=[make_create_decision("eligible")]
            )
        )
        progress: list[AnalysisProgressStage] = []

        async def report(stage: AnalysisProgressStage) -> None:
            progress.append(stage)

        with patch(
            "app.audience_analysis.analyze_topics",
            new=AsyncMock(return_value=topic_result),
        ) as analyze_topics_mock:
            await analyze_audiences(
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                provider,
                progress_reporter=report,
            )

        self.assertIs(
            analyze_topics_mock.await_args.kwargs["progress_reporter"],
            report,
        )
        self.assertEqual(
            progress,
            [
                "routing_commercial_clusters",
                "preparing_audience_evidence",
                "generating_audience_decisions",
                "validating_audience_decisions",
                "finalizing_audience_results",
            ],
        )

    async def test_composes_stages_and_uses_prepared_pageview_snapshots(
        self,
    ) -> None:
        lower = make_cluster("lower", (200, 100))
        rejected = make_cluster("rejected", (120, 80), confidence=0.4)
        higher = make_cluster("higher", (400, 300))
        topic_result = make_topic_result(
            [lower, rejected, higher],
            selected_pageviews=1_500,
        )
        response = AudienceGenerationResponse(
            decisions=[
                make_skip_decision("lower"),
                make_create_decision("higher"),
            ]
        )
        provider = FakeAudienceProvider(
            response,
            on_generate=lambda: setattr(higher, "total_views", 9_999),
        )

        with patch(
            "app.audience_analysis.analyze_topics",
            new=AsyncMock(return_value=topic_result),
        ) as analyze_topics_mock:
            result = await analyze_audiences(
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                provider,
                top_n=50,
            )

        analyze_topics_mock.assert_awaited_once()
        self.assertIs(result.topic_analysis, topic_result)
        self.assertEqual(
            [cluster.id for cluster in result.commercial_routing.eligible_clusters],
            ["higher", "lower"],
        )
        self.assertIs(result.preparation.clusters[0].cluster, higher)
        self.assertIs(result.preparation.clusters[1].cluster, lower)
        self.assertIs(
            result.audience_workflow.initial_provider_result,
            provider.initial_result,
        )
        self.assertIs(result.segments, result.audience_workflow.segments)
        self.assertIs(
            result.commercial_skips,
            result.commercial_routing.skipped_clusters,
        )
        self.assertIs(result.provider_skips, result.audience_workflow.provider_skips)
        self.assertIs(
            result.dropped_decisions,
            result.audience_workflow.dropped_decisions,
        )
        self.assertEqual(
            [segment.topic_cluster_ids[0] for segment in result.segments],
            ["higher"],
        )
        self.assertEqual(
            [skipped.cluster.id for skipped in result.provider_skips],
            ["lower"],
        )
        self.assertEqual(
            [context.cluster_id for context in provider.generated_contexts or ()],
            ["higher", "lower"],
        )
        self.assertEqual(provider.generate_call_count, 1)
        self.assertEqual(provider.revise_call_count, 0)

        metrics = result.metrics
        self.assertEqual(metrics.topic_cluster_count, 3)
        self.assertEqual(metrics.commercial_eligible_cluster_count, 2)
        self.assertEqual(metrics.commercial_skipped_cluster_count, 1)
        self.assertEqual(metrics.prepared_cluster_count, 2)
        self.assertEqual(metrics.final_segment_count, 1)
        self.assertEqual(metrics.provider_skipped_cluster_count, 1)
        self.assertEqual(metrics.validation_dropped_source_cluster_count, 0)
        self.assertEqual(metrics.unmatched_provider_output_count, 0)
        self.assertEqual(metrics.commercial_eligible_pageviews, 1_000)
        self.assertEqual(metrics.represented_audience_pageviews, 700)
        self.assertEqual(
            dict(metrics.commercial_skip_counts_by_reason),
            {LOW_COHESION_REASON: 1},
        )
        with self.assertRaises(TypeError):
            metrics.commercial_skip_counts_by_reason[  # type: ignore[index]
                "other"
            ] = 1

    async def test_empty_eligible_result_preserves_skips_without_provider_calls(
        self,
    ) -> None:
        rejected = make_cluster("rejected", (200, 100), confidence=0.4)
        topic_result = make_topic_result(
            [rejected],
            selected_pageviews=300,
        )
        provider = FakeAudienceProvider(None)

        with patch(
            "app.audience_analysis.analyze_topics",
            new=AsyncMock(return_value=topic_result),
        ):
            result = await analyze_audiences(
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                provider,
            )

        self.assertTrue(result.is_publishable)
        self.assertEqual(result.segments, ())
        self.assertEqual(result.preparation.clusters, ())
        self.assertEqual(len(result.commercial_skips), 1)
        self.assertEqual(result.commercial_skips[0].reason, LOW_COHESION_REASON)
        self.assertEqual(provider.generate_call_count, 0)
        self.assertEqual(provider.revise_call_count, 0)
        self.assertEqual(result.metrics.commercial_eligible_cluster_count, 0)
        self.assertEqual(result.metrics.commercial_skipped_cluster_count, 1)

    async def test_completely_empty_analysis_allows_zero_denominator(self) -> None:
        topic_result = make_topic_result([], selected_pageviews=0)
        provider = FakeAudienceProvider(None)

        with patch(
            "app.audience_analysis.analyze_topics",
            new=AsyncMock(return_value=topic_result),
        ):
            result = await analyze_audiences(
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                provider,
            )

        self.assertEqual(result.preparation.total_analyzed_views, 0)
        self.assertEqual(result.audience_workflow.metrics.provider_call_count, 0)
        self.assertEqual(result.metrics.topic_cluster_count, 0)
        self.assertEqual(result.metrics.commercial_eligible_pageviews, 0)
        self.assertEqual(result.metrics.represented_audience_pageviews, 0)
        self.assertEqual(provider.generate_call_count, 0)

    async def test_source_integrity_and_initial_provider_failures_remain_fatal(
        self,
    ) -> None:
        cluster = make_cluster("source", (200, 100))
        invalid_source_result = make_topic_result(
            [cluster],
            selected_pageviews=299,
        )
        provider = FakeAudienceProvider(None)

        with patch(
            "app.audience_analysis.analyze_topics",
            new=AsyncMock(return_value=invalid_source_result),
        ):
            with self.assertRaises(AudienceSourceIntegrityError) as raised:
                await analyze_audiences(
                    object(),  # type: ignore[arg-type]
                    object(),  # type: ignore[arg-type]
                    object(),  # type: ignore[arg-type]
                    provider,
                )

        self.assertEqual(raised.exception.code, CLUSTER_VIEWS_EXCEED_TOTAL)
        self.assertEqual(provider.generate_call_count, 0)

        valid_source_result = make_topic_result(
            [cluster],
            selected_pageviews=500,
        )
        initial_error = AudienceProviderError("safe initial failure")
        failing_provider = FakeAudienceProvider(None, initial_error=initial_error)
        with patch(
            "app.audience_analysis.analyze_topics",
            new=AsyncMock(return_value=valid_source_result),
        ):
            with self.assertRaises(AudienceProviderError) as provider_raised:
                await analyze_audiences(
                    object(),  # type: ignore[arg-type]
                    object(),  # type: ignore[arg-type]
                    object(),  # type: ignore[arg-type]
                    failing_provider,
                )

        self.assertIs(provider_raised.exception, initial_error)
        self.assertEqual(failing_provider.generate_call_count, 1)
        self.assertEqual(failing_provider.revise_call_count, 0)

    async def test_revision_failure_result_is_retained(self) -> None:
        created = make_cluster("created", (300, 200))
        missing = make_cluster("missing", (200, 100))
        topic_result = make_topic_result(
            [missing, created],
            selected_pageviews=1_000,
        )
        provider = FakeAudienceProvider(
            AudienceGenerationResponse(
                decisions=[make_create_decision("created")]
            ),
            revision_error=AudienceProviderError("safe revision failure"),
        )

        with patch(
            "app.audience_analysis.analyze_topics",
            new=AsyncMock(return_value=topic_result),
        ):
            result = await analyze_audiences(
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                provider,
            )

        self.assertTrue(result.is_publishable)
        self.assertEqual(provider.generate_call_count, 1)
        self.assertEqual(provider.revise_call_count, 1)
        self.assertEqual(
            [segment.topic_cluster_ids[0] for segment in result.segments],
            ["created"],
        )
        self.assertEqual(len(result.dropped_decisions), 1)
        self.assertEqual(
            result.dropped_decisions[0].drop_code,
            REVISION_PROVIDER_FAILURE,
        )
        self.assertEqual(
            result.metrics.validation_dropped_source_cluster_count,
            1,
        )
        self.assertEqual(result.metrics.represented_audience_pageviews, 500)

    async def test_rejects_routing_count_and_cluster_id_partition_changes(
        self,
    ) -> None:
        cluster = make_cluster("source", (200, 100))
        rogue = make_cluster("rogue", (200, 100))
        topic_result = make_topic_result([cluster], selected_pageviews=500)

        for malformed_routing in (
            CommercialSafetyResult(eligible_clusters=(), skipped_clusters=()),
            CommercialSafetyResult(
                eligible_clusters=(rogue,),
                skipped_clusters=(),
            ),
        ):
            with self.subTest(routing=malformed_routing):
                provider = FakeAudienceProvider(None)
                with (
                    patch(
                        "app.audience_analysis.analyze_topics",
                        new=AsyncMock(return_value=topic_result),
                    ),
                    patch(
                        "app.audience_analysis.route_commercial_clusters",
                        return_value=malformed_routing,
                    ),
                ):
                    with self.assertRaises(
                        AudienceAnalysisInvariantError
                    ) as raised:
                        await analyze_audiences(
                            object(),  # type: ignore[arg-type]
                            object(),  # type: ignore[arg-type]
                            object(),  # type: ignore[arg-type]
                            provider,
                        )

                self.assertEqual(raised.exception.code, ROUTING_PARTITION_MISMATCH)
                self.assertEqual(provider.generate_call_count, 0)

    async def test_rejects_workflow_count_and_cluster_id_partition_changes(
        self,
    ) -> None:
        cluster = make_cluster("source", (200, 100))
        topic_result = make_topic_result([cluster], selected_pageviews=500)

        for corruption in ("count", "cluster_id"):
            with self.subTest(corruption=corruption):
                provider = FakeAudienceProvider(
                    AudienceGenerationResponse(
                        decisions=[make_create_decision("source")]
                    )
                )

                async def corrupt_workflow(
                    preparation,
                    injected_provider,
                ) -> AudienceWorkflowResult:
                    workflow_result = await run_audience_workflow(
                        preparation,
                        injected_provider,
                    )
                    if corruption == "count":
                        return replace(workflow_result, segments=())
                    segment = workflow_result.segments[0].model_copy(deep=True)
                    segment.topic_cluster_ids = ["outside"]
                    return replace(workflow_result, segments=(segment,))

                with (
                    patch(
                        "app.audience_analysis.analyze_topics",
                        new=AsyncMock(return_value=topic_result),
                    ),
                    patch(
                        "app.audience_analysis.run_audience_workflow",
                        side_effect=corrupt_workflow,
                    ),
                ):
                    with self.assertRaises(
                        AudienceAnalysisInvariantError
                    ) as raised:
                        await analyze_audiences(
                            object(),  # type: ignore[arg-type]
                            object(),  # type: ignore[arg-type]
                            object(),  # type: ignore[arg-type]
                            provider,
                        )

                self.assertEqual(raised.exception.code, WORKFLOW_PARTITION_MISMATCH)


if __name__ == "__main__":
    unittest.main()
