"""Focused tests for truthful completed audience-decision traces."""

from collections.abc import Sequence
from dataclasses import FrozenInstanceError, fields
from datetime import date
import unittest

from app.agent.audience_finalization import (
    CROSS_CLUSTER_SUPPORTING_REFERENCE,
    MISSING_CLUSTER_DECISION,
    UNKNOWN_DECISION_CLUSTER,
    UNKNOWN_SUPPORTING_REFERENCE,
    prepare_audience_clusters,
)
from app.agent.audience_provider import (
    AudienceProviderError,
    AudienceProviderResult,
    AudienceRevisionRequest,
    AudienceTokenUsage,
)
from app.agent.audience_trace import (
    AudienceTraceEvent,
    build_audience_decision_traces,
)
from app.agent.audience_workflow import (
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
        summary=f"{title} is useful evidence for this topic.",
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
        keywords=[cluster_id, "example"],
        total_views=300,
        article_count=2,
        confidence_score=0.8,
    )


def make_create(
    cluster_id: str,
    references: list[str] | None = None,
) -> CreateAudienceDecision:
    return CreateAudienceDecision(
        decision="create_audience",
        cluster_id=cluster_id,
        name=f"{cluster_id.title()} Followers",
        description="People following this coherent topic and its developments.",
        supporting_article_reference_ids=(
            references or [f"{cluster_id}:a0", f"{cluster_id}:a1"]
        ),
        buying_power="medium",
        buying_power_reason=(
            "The audience includes broad groups with repeat category spending."
        ),
        brand_categories=["Media"],
        commercial_confidence=0.75,
        commercial_confidence_reason=(
            "The selected articles provide coherent commercial evidence."
        ),
    )


def make_skip(cluster_id: str) -> SkipClusterDecision:
    return SkipClusterDecision(
        decision="skip_cluster",
        cluster_id=cluster_id,
        reason="The source topic does not support a sufficiently specific audience.",
    )


def make_response(
    *decisions: CreateAudienceDecision | SkipClusterDecision,
) -> AudienceGenerationResponse:
    return AudienceGenerationResponse(decisions=list(decisions))


def make_provider_result(
    response: AudienceGenerationResponse,
    phase: str,
) -> AudienceProviderResult:
    return AudienceProviderResult(
        response=response,
        model="mock-model",
        response_id=f"response-{phase}",
        elapsed_seconds=0.1,
        usage=AudienceTokenUsage(10, 5, 15),
    )


class FakeProvider:
    def __init__(
        self,
        initial: AudienceGenerationResponse,
        revision: AudienceGenerationResponse | None = None,
        *,
        revision_error: AudienceProviderError | None = None,
    ) -> None:
        self.initial = initial
        self.revision = revision
        self.revision_error = revision_error

    async def generate(
        self,
        _contexts: Sequence[CompactClusterContext],
    ) -> AudienceProviderResult:
        return make_provider_result(self.initial, "initial")

    async def revise(
        self,
        _requests: Sequence[AudienceRevisionRequest],
    ) -> AudienceProviderResult:
        if self.revision_error is not None:
            raise self.revision_error
        if self.revision is None:
            raise AssertionError("unexpected revision")
        return make_provider_result(self.revision, "revision")


class AudienceTraceTests(unittest.IsolatedAsyncioTestCase):
    async def test_initial_outcomes_and_unmatched_output_are_aligned(self) -> None:
        clusters = [make_cluster("create"), make_cluster("skip")]
        preparation = prepare_audience_clusters(
            clusters,
            total_analyzed_views=1_000,
        )
        result = await run_audience_workflow(
            preparation,
            FakeProvider(
                make_response(
                    make_skip("skip"),
                    make_create("create"),
                    make_skip("outside"),
                )
            ),
        )

        projection = build_audience_decision_traces(preparation, result)

        self.assertEqual(
            [trace.cluster_id for trace in projection.traces],
            ["create", "skip", "outside"],
        )
        self.assertEqual(
            projection.segment_trace_ids,
            ("audience-trace-001",),
        )
        self.assertEqual(
            projection.provider_skip_trace_ids,
            ("audience-trace-002",),
        )
        self.assertEqual(
            projection.drop_trace_ids,
            ("audience-trace-003",),
        )
        self.assertEqual(
            [event.code for event in projection.traces[0].events],
            [
                "generation_requested",
                "decision_received",
                "validation_passed",
                "audience_published",
            ],
        )
        self.assertEqual(
            [event.code for event in projection.traces[1].events],
            [
                "generation_requested",
                "decision_received",
                "validation_passed",
                "provider_skipped",
            ],
        )
        self.assertEqual(
            [event.code for event in projection.traces[2].events],
            [
                "decision_received",
                "validation_failed",
                "decision_dropped",
            ],
        )
        self.assertFalse(projection.traces[2].source_known)
        self.assertNotIn(
            "generation_requested",
            [event.code for event in projection.traces[2].events],
        )
        self.assertEqual(
            projection.traces[2].events[1].issues[0].code,
            UNKNOWN_DECISION_CLUSTER,
        )
        with self.assertRaises(FrozenInstanceError):
            projection.traces[0].cluster_id = "changed"  # type: ignore[misc]
        self.assertNotIn("decisions", {field.name for field in fields(AudienceTraceEvent)})
        self.assertNotIn("reasoning", {field.name for field in fields(AudienceTraceEvent)})

    async def test_revision_events_preserve_exact_ordered_issues(self) -> None:
        clusters = [make_cluster("invalid"), make_cluster("missing")]
        preparation = prepare_audience_clusters(
            clusters,
            total_analyzed_views=1_000,
        )
        result = await run_audience_workflow(
            preparation,
            FakeProvider(
                make_response(
                    make_create(
                        "invalid",
                        ["invalid:a0", "invalid:a9"],
                    )
                ),
                make_response(
                    make_skip("invalid"),
                    make_create("missing"),
                ),
            ),
        )

        projection = build_audience_decision_traces(preparation, result)

        self.assertEqual(
            projection.segment_trace_ids,
            ("audience-trace-002",),
        )
        self.assertEqual(
            projection.provider_skip_trace_ids,
            ("audience-trace-001",),
        )
        invalid_trace, missing_trace = projection.traces
        self.assertEqual(
            [event.code for event in invalid_trace.events],
            [
                "generation_requested",
                "decision_received",
                "validation_failed",
                "revision_requested",
                "decision_received",
                "validation_passed",
                "provider_skipped",
            ],
        )
        failed = invalid_trace.events[2]
        requested = invalid_trace.events[3]
        self.assertEqual(
            [(issue.code, issue.reference_id) for issue in failed.issues],
            [(UNKNOWN_SUPPORTING_REFERENCE, "invalid:a9")],
        )
        self.assertEqual(requested.issues, failed.issues)
        self.assertEqual(
            [event.code for event in missing_trace.events],
            [
                "generation_requested",
                "validation_failed",
                "revision_requested",
                "decision_received",
                "validation_passed",
                "audience_published",
            ],
        )
        self.assertEqual(
            missing_trace.events[1].issues[0].code,
            MISSING_CLUSTER_DECISION,
        )

    async def test_revision_failure_and_unknown_output_stay_distinct(self) -> None:
        preparation = prepare_audience_clusters(
            [make_cluster("source")],
            total_analyzed_views=1_000,
        )
        result = await run_audience_workflow(
            preparation,
            FakeProvider(
                make_response(make_skip("outside")),
                revision_error=AudienceProviderError("safe failure"),
            ),
        )

        projection = build_audience_decision_traces(preparation, result)

        self.assertEqual(
            projection.drop_trace_ids,
            ("audience-trace-001", "audience-trace-002"),
        )
        source_trace, unknown_trace = projection.traces
        self.assertEqual(
            [event.code for event in source_trace.events],
            [
                "generation_requested",
                "validation_failed",
                "revision_requested",
                "revision_failed",
                "decision_dropped",
            ],
        )
        self.assertEqual(
            source_trace.events[3].outcome_code,
            REVISION_PROVIDER_FAILURE,
        )
        self.assertEqual(source_trace.final_outcome, "validation_dropped")
        self.assertEqual(
            [event.code for event in unknown_trace.events],
            [
                "decision_received",
                "validation_failed",
                "decision_dropped",
            ],
        )
        self.assertNotIn(
            "revision_requested",
            [event.code for event in unknown_trace.events],
        )

    async def test_still_invalid_and_unknown_revision_outputs_are_traced(
        self,
    ) -> None:
        preparation = prepare_audience_clusters(
            [make_cluster("invalid"), make_cluster("missing")],
            total_analyzed_views=1_000,
        )
        result = await run_audience_workflow(
            preparation,
            FakeProvider(
                make_response(
                    make_create(
                        "invalid",
                        ["invalid:a0", "invalid:a9"],
                    )
                ),
                make_response(
                    make_create(
                        "invalid",
                        ["invalid:a0", "missing:a0"],
                    ),
                    make_skip("outside"),
                ),
            ),
        )

        projection = build_audience_decision_traces(preparation, result)

        self.assertEqual(projection.segment_trace_ids, ())
        self.assertEqual(projection.provider_skip_trace_ids, ())
        self.assertEqual(
            projection.drop_trace_ids,
            (
                "audience-trace-001",
                "audience-trace-002",
                "audience-trace-003",
            ),
        )
        invalid_trace, missing_trace, unknown_trace = projection.traces
        self.assertEqual(
            [event.code for event in invalid_trace.events],
            [
                "generation_requested",
                "decision_received",
                "validation_failed",
                "revision_requested",
                "decision_received",
                "validation_failed",
                "decision_dropped",
            ],
        )
        self.assertEqual(
            [
                (issue.code, issue.reference_id)
                for issue in invalid_trace.events[5].issues
            ],
            [(CROSS_CLUSTER_SUPPORTING_REFERENCE, "missing:a0")],
        )
        self.assertEqual(
            invalid_trace.events[-1].outcome_code,
            UNRESOLVED_AFTER_REVISION,
        )
        self.assertEqual(
            [event.code for event in missing_trace.events],
            [
                "generation_requested",
                "validation_failed",
                "revision_requested",
                "validation_failed",
                "decision_dropped",
            ],
        )
        self.assertEqual(
            [event.phase for event in unknown_trace.events],
            ["revision", "revision", "final"],
        )
        self.assertEqual(
            [event.code for event in unknown_trace.events],
            [
                "decision_received",
                "validation_failed",
                "decision_dropped",
            ],
        )
        self.assertEqual(
            unknown_trace.events[1].issues[0].code,
            UNKNOWN_DECISION_CLUSTER,
        )
        self.assertNotIn(
            "revision_requested",
            [event.code for event in unknown_trace.events],
        )


if __name__ == "__main__":
    unittest.main()
