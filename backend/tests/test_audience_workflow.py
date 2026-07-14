"""Focused tests for the bounded audience revision workflow."""

from collections.abc import Sequence
from datetime import date
import unittest
from unittest.mock import patch

from app.agent.audience_finalization import (
    CROSS_CLUSTER_SUPPORTING_REFERENCE,
    MISSING_CLUSTER_DECISION,
    UNKNOWN_DECISION_CLUSTER,
    UNKNOWN_SUPPORTING_REFERENCE,
    finalize_audience_decisions,
    prepare_audience_clusters,
)
from app.agent.audience_provider import (
    AudienceProviderError,
    AudienceProviderResult,
    AudienceRevisionRequest,
    AudienceTokenUsage,
)
from app.agent.audience_workflow import (
    MAX_REVISIONS,
    REVISION_PROVIDER_FAILURE,
    UNRESOLVED_AFTER_REVISION,
    run_audience_workflow,
)
from app.models import Article, TopicCluster
from app.models.audience_generation import (
    AudienceGenerationResponse,
    CompactClusterContext,
    CreateAudienceDecision,
    SkipClusterDecision,
)


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


def make_cluster(cluster_id: str) -> TopicCluster:
    articles = [
        make_article(f"{cluster_id} Alpha", 200),
        make_article(f"{cluster_id} Beta", 100),
    ]
    return TopicCluster(
        id=cluster_id,
        name=f"Topic {cluster_id}",
        articles=articles,
        keywords=[cluster_id, "example topic"],
        total_views=300,
        article_count=2,
        confidence_score=0.8,
    )


def make_create_decision(
    cluster_id: str,
    references: list[str] | None = None,
) -> CreateAudienceDecision:
    return CreateAudienceDecision(
        decision="create_audience",
        cluster_id=cluster_id,
        name=f"{cluster_id.title()} Followers",
        description=(
            "People following this coherent topic and its related developments."
        ),
        supporting_article_reference_ids=(
            references or [f"{cluster_id}:a0", f"{cluster_id}:a1"]
        ),
        buying_power="medium",
        buying_power_reason=(
            "The audience includes broad consumer groups with repeat spending."
        ),
        brand_categories=["Media", "Consumer technology"],
        commercial_confidence=0.76,
        commercial_confidence_reason=(
            "The selected source articles provide coherent commercial evidence."
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


def make_response(
    *decisions: CreateAudienceDecision | SkipClusterDecision,
) -> AudienceGenerationResponse:
    return AudienceGenerationResponse(decisions=list(decisions))


def make_provider_result(
    response: AudienceGenerationResponse,
    *,
    phase: str,
) -> AudienceProviderResult:
    if phase == "initial":
        usage = AudienceTokenUsage(100, 50, 150)
        elapsed_seconds = 0.4
    else:
        usage = AudienceTokenUsage(40, 20, 60)
        elapsed_seconds = 0.2
    return AudienceProviderResult(
        response=response,
        model="mock-model",
        response_id=f"response-{phase}",
        elapsed_seconds=elapsed_seconds,
        usage=usage,
    )


class FakeAudienceProvider:
    def __init__(
        self,
        initial_response: AudienceGenerationResponse | None,
        revision_response: AudienceGenerationResponse | None = None,
        *,
        initial_error: AudienceProviderError | None = None,
        revision_error: AudienceProviderError | None = None,
    ) -> None:
        self.initial_response = initial_response
        self.revision_response = revision_response
        self.initial_error = initial_error
        self.revision_error = revision_error
        self.generated_contexts: tuple[CompactClusterContext, ...] | None = None
        self.revision_requests: tuple[AudienceRevisionRequest, ...] | None = None
        self.generate_call_count = 0
        self.revise_call_count = 0

    async def generate(
        self,
        cluster_contexts: Sequence[CompactClusterContext],
    ) -> AudienceProviderResult:
        self.generate_call_count += 1
        self.generated_contexts = tuple(cluster_contexts)
        if self.initial_error is not None:
            raise self.initial_error
        if self.initial_response is None:
            raise AssertionError("generate should not have been called")
        return make_provider_result(self.initial_response, phase="initial")

    async def revise(
        self,
        revision_requests: Sequence[AudienceRevisionRequest],
    ) -> AudienceProviderResult:
        self.revise_call_count += 1
        self.revision_requests = tuple(revision_requests)
        if self.revision_error is not None:
            raise self.revision_error
        if self.revision_response is None:
            raise AssertionError("revise should not have been called")
        return make_provider_result(self.revision_response, phase="revision")


class AudienceWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_preparation_avoids_all_provider_calls(self) -> None:
        preparation = prepare_audience_clusters(
            [],
            total_analyzed_views=1_000,
        )
        provider = FakeAudienceProvider(None)

        result = await run_audience_workflow(preparation, provider)

        self.assertTrue(result.is_publishable)
        self.assertEqual(result.segments, ())
        self.assertEqual(result.provider_skips, ())
        self.assertEqual(result.dropped_decisions, ())
        self.assertEqual(provider.generate_call_count, 0)
        self.assertEqual(provider.revise_call_count, 0)
        self.assertEqual(result.metrics.provider_call_count, 0)

    async def test_fatal_initial_provider_error_identity_is_preserved(
        self,
    ) -> None:
        preparation = prepare_audience_clusters(
            [make_cluster("source")],
            total_analyzed_views=1_000,
        )
        source_error = AudienceProviderError("safe initial failure")
        provider = FakeAudienceProvider(
            None,
            initial_error=source_error,
        )

        with self.assertRaises(AudienceProviderError) as raised:
            await run_audience_workflow(preparation, provider)

        self.assertIs(raised.exception, source_error)
        self.assertEqual(provider.generate_call_count, 1)
        self.assertEqual(provider.revise_call_count, 0)

    async def test_all_valid_preserves_segment_and_skip_identity(self) -> None:
        clusters = [make_cluster("create"), make_cluster("skip")]
        preparation = prepare_audience_clusters(
            clusters,
            total_analyzed_views=1_000,
        )
        provider = FakeAudienceProvider(
            make_response(
                make_skip_decision("skip"),
                make_create_decision("create"),
            )
        )

        result = await run_audience_workflow(preparation, provider)

        self.assertTrue(result.is_publishable)
        self.assertEqual(provider.generate_call_count, 1)
        self.assertEqual(provider.revise_call_count, 0)
        self.assertIsNotNone(result.initial_validation_report)
        initial_report = result.initial_validation_report
        assert initial_report is not None
        self.assertIs(result.segments[0], initial_report.valid_segments[0])
        self.assertIs(result.provider_skips[0], initial_report.provider_skips[0])
        self.assertEqual(result.dropped_decisions, ())
        self.assertEqual(result.metrics.revision_count, 0)
        self.assertEqual(result.metrics.final_valid_decision_count, 2)
        self.assertEqual(result.metrics.provider_total_tokens, 150)

    async def test_validates_full_preparation_then_revision_subset(
        self,
    ) -> None:
        clusters = [make_cluster("valid"), make_cluster("missing")]
        preparation = prepare_audience_clusters(
            clusters,
            total_analyzed_views=1_000,
        )
        provider = FakeAudienceProvider(
            make_response(make_create_decision("valid")),
            make_response(make_create_decision("missing")),
        )

        with patch(
            "app.agent.audience_workflow.finalize_audience_decisions",
            wraps=finalize_audience_decisions,
        ) as finalize:
            result = await run_audience_workflow(preparation, provider)

        self.assertEqual(finalize.call_count, 2)
        initial_preparation = finalize.call_args_list[0].args[0]
        revision_preparation = finalize.call_args_list[1].args[0]
        self.assertIs(initial_preparation, preparation)
        self.assertEqual(
            [
                prepared.cluster_id
                for prepared in revision_preparation.clusters
            ],
            ["missing"],
        )
        self.assertIs(revision_preparation.clusters[0].cluster, clusters[1])

        initial_report = result.initial_validation_report
        revision_report = result.revision_validation_report
        assert initial_report is not None
        assert revision_report is not None
        self.assertIs(result.segments[0], initial_report.valid_segments[0])
        self.assertIs(result.segments[1], revision_report.valid_segments[0])

    async def test_revises_only_known_invalid_clusters_and_merges_in_order(
        self,
    ) -> None:
        clusters = [
            make_cluster("valid"),
            make_cluster("initial-skip"),
            make_cluster("invalid"),
            make_cluster("missing"),
        ]
        preparation = prepare_audience_clusters(
            clusters,
            total_analyzed_views=2_000,
        )
        initial_response = make_response(
            make_create_decision("valid"),
            make_skip_decision("initial-skip"),
            make_create_decision(
                "invalid",
                ["invalid:a0", "invalid:a9"],
            ),
        )
        revision_response = make_response(
            make_create_decision("valid"),
            make_skip_decision("invalid"),
            make_create_decision("missing"),
        )
        provider = FakeAudienceProvider(initial_response, revision_response)

        result = await run_audience_workflow(preparation, provider)

        self.assertEqual(MAX_REVISIONS, 1)
        self.assertEqual(provider.revise_call_count, 1)
        assert provider.revision_requests is not None
        self.assertEqual(
            [request.context.cluster_id for request in provider.revision_requests],
            ["invalid", "missing"],
        )
        self.assertEqual(
            [
                [
                    (issue.code, issue.reference_id)
                    for issue in request.validation_issues
                ]
                for request in provider.revision_requests
            ],
            [
                [(UNKNOWN_SUPPORTING_REFERENCE, "invalid:a9")],
                [(MISSING_CLUSTER_DECISION, None)],
            ],
        )
        self.assertEqual(
            [len(request.previous_decisions) for request in provider.revision_requests],
            [1, 0],
        )

        initial_report = result.initial_validation_report
        assert initial_report is not None
        self.assertIs(result.segments[0], initial_report.valid_segments[0])
        self.assertEqual(
            [segment.topic_cluster_ids[0] for segment in result.segments],
            ["valid", "missing"],
        )
        self.assertEqual(
            [skipped.cluster.id for skipped in result.provider_skips],
            ["initial-skip", "invalid"],
        )
        self.assertIs(
            result.provider_skips[0],
            initial_report.provider_skips[0],
        )
        self.assertEqual(len(result.dropped_decisions), 1)
        self.assertEqual(result.dropped_decisions[0].cluster_id, "valid")
        self.assertIsNone(result.dropped_decisions[0].source_cluster)
        self.assertEqual(
            result.dropped_decisions[0].drop_code,
            UNRESOLVED_AFTER_REVISION,
        )
        self.assertEqual(
            [issue.code for issue in result.dropped_decisions[0].issues],
            [UNKNOWN_DECISION_CLUSTER],
        )

        metrics = result.metrics
        self.assertEqual(metrics.revision_count, 1)
        self.assertEqual(metrics.revision_requested_cluster_count, 2)
        self.assertEqual(metrics.revision_decision_count, 3)
        self.assertEqual(metrics.revision_valid_decision_count, 2)
        self.assertEqual(metrics.final_valid_decision_count, 4)
        self.assertEqual(metrics.dropped_source_cluster_count, 0)
        self.assertEqual(metrics.dropped_unmatched_decision_count, 1)
        self.assertEqual(metrics.provider_call_count, 2)
        self.assertEqual(metrics.provider_input_tokens, 140)
        self.assertEqual(metrics.provider_output_tokens, 70)
        self.assertEqual(metrics.provider_total_tokens, 210)
        self.assertAlmostEqual(metrics.provider_elapsed_seconds, 0.6)
        with self.assertRaises(TypeError):
            metrics.drop_counts_by_code["other"] = 1  # type: ignore[index]

    async def test_graph_revises_at_most_once_and_drops_still_invalid_results(
        self,
    ) -> None:
        clusters = [
            make_cluster("valid"),
            make_cluster("invalid"),
            make_cluster("missing"),
        ]
        preparation = prepare_audience_clusters(
            clusters,
            total_analyzed_views=1_500,
        )
        provider = FakeAudienceProvider(
            make_response(
                make_create_decision("valid"),
                make_create_decision(
                    "invalid",
                    ["invalid:a0", "invalid:a9"],
                ),
            ),
            make_response(
                make_create_decision(
                    "invalid",
                    ["invalid:a0", "valid:a0"],
                ),
                make_skip_decision("outside"),
            ),
        )

        result = await run_audience_workflow(preparation, provider)

        self.assertTrue(result.is_publishable)
        self.assertEqual(MAX_REVISIONS, 1)
        self.assertEqual(provider.generate_call_count, 1)
        self.assertEqual(provider.revise_call_count, 1)
        self.assertEqual(
            [segment.topic_cluster_ids[0] for segment in result.segments],
            ["valid"],
        )
        self.assertEqual(result.provider_skips, ())
        self.assertEqual(
            [dropped.cluster_id for dropped in result.dropped_decisions],
            ["invalid", "missing", "outside"],
        )
        self.assertEqual(
            [dropped.drop_code for dropped in result.dropped_decisions],
            [UNRESOLVED_AFTER_REVISION] * 3,
        )
        self.assertEqual(result.metrics.dropped_source_cluster_count, 2)
        self.assertEqual(result.metrics.dropped_unmatched_decision_count, 1)
        self.assertEqual(
            dict(result.metrics.validation_issue_counts_by_code),
            {
                CROSS_CLUSTER_SUPPORTING_REFERENCE: 1,
                MISSING_CLUSTER_DECISION: 1,
                UNKNOWN_DECISION_CLUSTER: 1,
            },
        )
        self.assertEqual(
            dict(result.metrics.drop_counts_by_code),
            {UNRESOLVED_AFTER_REVISION: 3},
        )

    async def test_revision_failure_preserves_valid_and_drops_pending(self) -> None:
        clusters = [make_cluster("valid"), make_cluster("missing")]
        preparation = prepare_audience_clusters(
            clusters,
            total_analyzed_views=1_000,
        )
        provider = FakeAudienceProvider(
            make_response(make_create_decision("valid")),
            revision_error=AudienceProviderError("safe revision failure"),
        )

        result = await run_audience_workflow(preparation, provider)

        self.assertTrue(result.is_publishable)
        self.assertEqual(provider.revise_call_count, 1)
        initial_report = result.initial_validation_report
        assert initial_report is not None
        self.assertIs(result.segments[0], initial_report.valid_segments[0])
        self.assertEqual(len(result.dropped_decisions), 1)
        dropped = result.dropped_decisions[0]
        self.assertEqual(dropped.cluster_id, "missing")
        self.assertEqual(dropped.drop_code, REVISION_PROVIDER_FAILURE)
        self.assertEqual(
            [issue.code for issue in dropped.issues],
            [MISSING_CLUSTER_DECISION],
        )
        self.assertIsNone(result.revision_provider_result)
        self.assertIsNone(result.revision_validation_report)
        self.assertEqual(result.metrics.provider_call_count, 2)
        self.assertEqual(result.metrics.provider_total_tokens, 150)
        self.assertEqual(
            dict(result.metrics.drop_counts_by_code),
            {REVISION_PROVIDER_FAILURE: 1},
        )


if __name__ == "__main__":
    unittest.main()
