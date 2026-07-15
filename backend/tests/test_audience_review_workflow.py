"""Focused topology and pause behavior tests for the review graph."""

from collections.abc import Sequence
from datetime import UTC, date, datetime
from json import dumps
import unittest

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.agent.audience_finalization import (
    AudiencePreparation,
    PreparedAudienceCluster,
    prepare_audience_clusters,
)
from app.agent.audience_provider import (
    AudienceProviderResult,
    AudienceRevisionRequest,
    AudienceTokenUsage,
)
from app.agent.audience_review_workflow import (
    AudienceReviewWorkflowContext,
    _restore_preparation,
    build_audience_review_graph,
    build_review_initial_state,
    build_review_run_result,
    snapshot_preparation,
)
from app.agent.audience_trace import build_audience_decision_traces
from app.agent.audience_workflow import run_audience_workflow
from app.models import Article, TopicCluster
from app.models.audience_generation import (
    AudienceGenerationResponse,
    CompactArticleContext,
    CompactClusterContext,
    CreateAudienceDecision,
    SkipClusterDecision,
)
from app.models.audience_review import (
    ApproveReviewCommand,
    new_command_id,
    new_run_id,
    review_id_for,
    review_thread_id,
)


def make_article(title: str, views: int) -> Article:
    end = date(2026, 7, 12)
    return Article(
        title=title,
        normalized_title=title,
        url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
        weekly_views=views,
        daily_views={end: views},
        summary=f"{title} is useful evidence for this topic.",
        analysis_start_date=date(2026, 7, 6),
        analysis_end_date=end,
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
    *,
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
        reason="The topic does not support a sufficiently specific audience.",
    )


def response(*decisions: object) -> AudienceGenerationResponse:
    return AudienceGenerationResponse(decisions=list(decisions))


def make_boundary_preparation() -> tuple[AudiencePreparation, str]:
    """Build a valid preparation at source-model string boundaries."""
    cluster_id = "x" * 128
    start = date(2026, 7, 6)
    end = date(2026, 7, 12)
    articles = [
        Article(
            title=" " + ("A" * 2_000) + " ",
            normalized_title=" " + ("a" * 2_000) + " ",
            url=" https://example.test/" + ("u" * 8_000) + " ",
            weekly_views=200,
            daily_views={end: 100, start: 100},
            summary=" " + ("S" * 20_000) + " ",
            analysis_start_date=start,
            analysis_end_date=end,
        ),
        Article(
            title="B" * 2_000,
            normalized_title="b" * 2_000,
            url="https://example.test/" + ("v" * 8_000),
            weekly_views=100,
            daily_views={end: 50, start: 50},
            summary="T" * 20_000,
            analysis_start_date=start,
            analysis_end_date=end,
        ),
    ]
    cluster = TopicCluster(
        id=cluster_id,
        name="N" * 120,
        description="D" * 10_000,
        articles=articles,
        keywords=["keyword"],
        total_views=300,
        article_count=2,
        confidence_score=0.8,
    )
    references = ("ref-a", "ref-b")
    context = CompactClusterContext(
        cluster_id=cluster_id,
        name=cluster.name,
        keywords=["keyword"],
        total_views=300,
        article_count=2,
        topic_confidence=0.8,
        articles=[
            CompactArticleContext(
                reference_id=reference,
                title=f"Evidence {index}",
                weekly_views=article.weekly_views,
                summary="Compact evidence summary.",
            )
            for index, (reference, article) in enumerate(
                zip(references, articles, strict=True)
            )
        ],
    )
    prepared = PreparedAudienceCluster(
        cluster=cluster,
        context=context,
        cluster_id=cluster_id,
        cluster_pageviews=300,
        evidence_reference_ids=references,
        resolution_map=dict(zip(references, articles, strict=True)),
    )
    return (
        AudiencePreparation(
            clusters=(prepared,),
            total_analyzed_views=1_000,
            reference_cluster_ids={reference: cluster_id for reference in references},
        ),
        cluster_id,
    )


def make_boundary_create(cluster_id: str) -> CreateAudienceDecision:
    return CreateAudienceDecision(
        decision="create_audience",
        cluster_id=cluster_id,
        name="A" * 80,
        description="D" * 500,
        supporting_article_reference_ids=["ref-a", "ref-b"],
        buying_power="medium",
        buying_power_reason="B" * 300,
        brand_categories=["C" * 60],
        commercial_confidence=0.75,
        commercial_confidence_reason="R" * 300,
    )


def make_layered_preparation(
    *prepared_ids: str,
) -> AudiencePreparation:
    """Build an accepted preparation whose three identity layers differ."""
    base = prepare_audience_clusters(
        [make_cluster(cluster_id) for cluster_id in prepared_ids],
        total_analyzed_views=10_000,
    )
    layered = []
    for index, prepared in enumerate(base.clusters):
        source_cluster = prepared.cluster.model_copy(
            deep=True,
            update={
                "id": f"source-{prepared.cluster_id}",
                "name": f"Source {prepared.cluster_id}",
                "description": f"Source description {prepared.cluster_id}",
                "total_views": 700 + index,
                "confidence_score": 0.1 + (index * 0.05),
            },
        )
        context = prepared.context.model_copy(
            deep=True,
            update={
                "cluster_id": f"context-{prepared.cluster_id}",
                "name": f"Context {prepared.cluster_id}",
                "total_views": 600 + index,
                "topic_confidence": 0.9 - (index * 0.05),
            },
        )
        layered.append(
            PreparedAudienceCluster(
                cluster=source_cluster,
                context=context,
                cluster_id=prepared.cluster_id,
                cluster_pageviews=400 + index,
                evidence_reference_ids=prepared.evidence_reference_ids,
                resolution_map=prepared.resolution_map,
            )
        )
    return AudiencePreparation(
        clusters=tuple(layered),
        total_analyzed_views=base.total_analyzed_views,
        reference_cluster_ids=base.reference_cluster_ids,
    )


class FakeProvider:
    def __init__(
        self,
        initial: AudienceGenerationResponse | None,
        revision: AudienceGenerationResponse | None = None,
    ) -> None:
        self.initial = initial
        self.revision = revision
        self.generate_calls = 0
        self.revise_calls = 0

    async def generate(
        self,
        _contexts: Sequence[CompactClusterContext],
    ) -> AudienceProviderResult:
        self.generate_calls += 1
        if self.initial is None:
            raise AssertionError("generate was not expected")
        return AudienceProviderResult(
            response=self.initial,
            model="must-not-be-checkpointed",
            response_id="must-not-be-checkpointed",
            elapsed_seconds=0.1,
            usage=AudienceTokenUsage(10, 5, 15),
        )

    async def revise(
        self,
        _requests: Sequence[AudienceRevisionRequest],
    ) -> AudienceProviderResult:
        self.revise_calls += 1
        if self.revision is None:
            raise AssertionError("revision was not expected")
        return AudienceProviderResult(
            response=self.revision,
            model="must-not-be-checkpointed",
            response_id="must-not-be-checkpointed",
            elapsed_seconds=0.1,
            usage=AudienceTokenUsage(5, 3, 8),
        )


class AudienceReviewWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def _invoke(self, preparation, provider):
        run_id = new_run_id()
        thread_id = review_thread_id(run_id)
        graph = build_audience_review_graph(InMemorySaver())
        output = await graph.ainvoke(
            build_review_initial_state(
                preparation,
                run_id=run_id,
                expires_at=datetime(2026, 7, 15, tzinfo=UTC).isoformat(),
            ),
            config={"configurable": {"thread_id": thread_id}},
            context=AudienceReviewWorkflowContext(provider),
        )
        return graph, run_id, thread_id, output

    async def test_empty_preparation_completes_without_interrupt(self) -> None:
        preparation = prepare_audience_clusters([], total_analyzed_views=1_000)
        provider = FakeProvider(None)
        graph, _, thread_id, output = await self._invoke(preparation, provider)

        self.assertNotIn("__interrupt__", output)
        self.assertTrue(output["completed"])
        self.assertEqual(provider.generate_calls, 0)
        result = build_review_run_result(
            dict((await graph.aget_state({"configurable": {"thread_id": thread_id}})).values),
            thread_id=thread_id,
        )
        self.assertEqual(result.status, "completed")

    async def test_skips_and_validation_drops_do_not_interrupt(self) -> None:
        clusters = [make_cluster("skip"), make_cluster("drop")]
        preparation = prepare_audience_clusters(clusters, total_analyzed_views=1_000)
        invalid = make_create("drop", references=["drop:a0", "missing:a9"])
        provider = FakeProvider(
            response(make_skip("skip"), invalid),
            response(invalid),
        )
        _, _, _, output = await self._invoke(preparation, provider)

        self.assertNotIn("__interrupt__", output)
        self.assertTrue(output["completed"])
        self.assertEqual(len(output["provider_skips"]), 1)
        self.assertEqual(len(output["validation_drops"]), 1)

    async def test_candidates_interrupt_sequentially_in_source_order(self) -> None:
        preparation = prepare_audience_clusters(
            [make_cluster("first"), make_cluster("second")],
            total_analyzed_views=1_000,
        )
        graph, run_id, thread_id, output = await self._invoke(
            preparation,
            FakeProvider(response(make_create("second"), make_create("first"))),
        )
        interrupts = output["__interrupt__"]
        self.assertEqual(len(interrupts), 1)
        payload = interrupts[0].value
        self.assertEqual(payload["cluster_id"], "first")
        self.assertEqual(payload["review_id"], review_id_for(run_id, "first"))
        dumps(payload)
        self.assertEqual(
            [record["status"] for record in output["records"]],
            ["pending_review", "queued"],
        )
        self.assertEqual(
            [trace["final_outcome"] for trace in output["traces"]],
            ["pending_review", "queued"],
        )
        self.assertEqual(
            [
                sum(
                    event["code"] == "review_requested"
                    for event in trace["events"]
                )
                for trace in output["traces"]
            ],
            [1, 0],
        )

        command = ApproveReviewCommand(
            type="approve",
            run_id=run_id,
            review_id=payload["review_id"],
            cluster_id="first",
            expected_version=1,
            command_id=new_command_id(),
        )
        resumed = await graph.ainvoke(
            Command(
                resume={
                    **command.model_dump(mode="json"),
                    "command_digest": "0" * 64,
                }
            ),
            config={"configurable": {"thread_id": thread_id}},
            context=AudienceReviewWorkflowContext(FakeProvider(None)),
        )
        self.assertEqual(len(resumed["__interrupt__"]), 1)
        self.assertEqual(resumed["__interrupt__"][0].value["cluster_id"], "second")
        self.assertEqual(
            sum(record["status"] == "pending_review" for record in resumed["records"]),
            1,
        )
        self.assertEqual(
            [record["status"] for record in resumed["records"]],
            ["published", "pending_review"],
        )
        self.assertEqual(
            [
                sum(
                    event["code"] == "review_requested"
                    for event in trace["events"]
                )
                for trace in resumed["traces"]
            ],
            [1, 1],
        )

    async def test_wrong_thread_resume_cannot_cross_original_state(self) -> None:
        preparation = prepare_audience_clusters(
            [make_cluster("one")], total_analyzed_views=1_000
        )
        graph, run_id, _, output = await self._invoke(
            preparation, FakeProvider(response(make_create("one")))
        )
        payload = output["__interrupt__"][0].value
        command = ApproveReviewCommand(
            type="approve",
            run_id=run_id,
            review_id=payload["review_id"],
            cluster_id="one",
            expected_version=1,
            command_id=new_command_id(),
        )
        await graph.ainvoke(
            Command(resume=command.model_dump(mode="json")),
            config={"configurable": {"thread_id": "wrong-thread"}},
            context=AudienceReviewWorkflowContext(FakeProvider(None)),
        )
        original = await graph.aget_state(
            {"configurable": {"thread_id": review_thread_id(run_id)}}
        )
        self.assertEqual(original.values["records"][0]["status"], "pending_review")

    async def test_source_valid_long_values_and_128_character_id_project(self) -> None:
        preparation, cluster_id = make_boundary_preparation()
        graph, _, thread_id, output = await self._invoke(
            preparation,
            FakeProvider(response(make_boundary_create(cluster_id))),
        )
        self.assertEqual(output["__interrupt__"][0].value["cluster_id"], cluster_id)
        result = build_review_run_result(
            dict(
                (
                    await graph.aget_state(
                        {"configurable": {"thread_id": thread_id}}
                    )
                ).values
            ),
            thread_id=thread_id,
        )
        pending = result.pending_review
        self.assertEqual(len(pending.recommendation.audience_id), 137)
        self.assertEqual(len(pending.evidence[0].article.title), 2_002)
        self.assertEqual(len(pending.evidence[0].article.url), 8_023)
        self.assertEqual(len(pending.evidence[0].article.summary), 20_002)
        self.assertTrue(pending.evidence[0].article.title.startswith(" "))
        self.assertTrue(pending.evidence[0].article.url.endswith(" "))
        self.assertTrue(pending.evidence[0].article.summary.endswith(" "))
        self.assertEqual(
            [entry.day for entry in pending.evidence[0].article.daily_views],
            [date(2026, 7, 6), date(2026, 7, 12)],
        )
        self.assertEqual(len(pending.recommendation.name), 80)
        self.assertEqual(len(pending.recommendation.description), 500)
        self.assertEqual(len(pending.recommendation.buying_power_reason), 300)
        self.assertEqual(
            len(pending.recommendation.commercial_confidence_reason),
            300,
        )

    async def test_review_snapshots_are_deeply_immutable_and_ordered(self) -> None:
        preparation, _ = make_boundary_preparation()
        snapshot = snapshot_preparation(preparation)
        article = snapshot.clusters[0].source.articles[0]
        self.assertEqual(
            [entry.day for entry in article.daily_views],
            [date(2026, 7, 6), date(2026, 7, 12)],
        )
        with self.assertRaises(TypeError):
            article.daily_views[0] = article.daily_views[1]
        with self.assertRaises(Exception):
            article.daily_views[0].pageviews = 999
        restored = type(snapshot).model_validate_json(snapshot.model_dump_json())
        self.assertEqual(snapshot, restored)
        self.assertEqual(snapshot.model_dump_json(), restored.model_dump_json())

    async def test_source_prepared_and_context_layers_restore_without_drift(
        self,
    ) -> None:
        preparation = make_layered_preparation(
            "create",
            "skip",
            "revision-drop",
            "revised-valid",
        )
        initial = response(
            make_create(
                "revised-valid",
                references=["revised-valid:a0", "missing:a9"],
            ),
            make_skip("skip"),
            make_create("create"),
            make_create(
                "revision-drop",
                references=["revision-drop:a0", "missing:a9"],
            ),
            make_create(
                "unknown",
                references=["create:a0", "create:a1"],
            ),
        )
        revision = response(
            make_create(
                "revision-drop",
                references=["revision-drop:a0", "missing:a9"],
            ),
            make_create("revised-valid"),
        )
        snapshot = snapshot_preparation(preparation)
        restored = _restore_preparation(snapshot.model_dump(mode="json"))

        self.assertEqual(
            [item.cluster_id for item in restored.clusters],
            [item.cluster_id for item in preparation.clusters],
        )
        self.assertEqual(
            list(restored.reference_cluster_ids.items()),
            list(preparation.reference_cluster_ids.items()),
        )
        for original, copy, saved in zip(
            preparation.clusters,
            restored.clusters,
            snapshot.clusters,
            strict=True,
        ):
            self.assertEqual(copy.cluster, original.cluster)
            self.assertEqual(copy.context, original.context)
            self.assertEqual(copy.cluster_id, original.cluster_id)
            self.assertEqual(
                copy.cluster_pageviews,
                original.cluster_pageviews,
            )
            self.assertEqual(
                copy.evidence_reference_ids,
                original.evidence_reference_ids,
            )
            self.assertEqual(
                list(copy.resolution_map.items()),
                list(original.resolution_map.items()),
            )
            self.assertEqual(saved.source.id, original.cluster.id)
            self.assertEqual(
                saved.source.total_views,
                original.cluster.total_views,
            )
            self.assertEqual(
                saved.source.confidence_score,
                original.cluster.confidence_score,
            )
            self.assertEqual(saved.cluster_id, original.cluster_id)
            self.assertEqual(
                saved.cluster_pageviews,
                original.cluster_pageviews,
            )
            self.assertEqual(saved.context, snapshot_preparation(
                AudiencePreparation(
                    clusters=(original,),
                    total_analyzed_views=10_000,
                    reference_cluster_ids=preparation.reference_cluster_ids,
                )
            ).clusters[0].context)

        original_result = await run_audience_workflow(
            preparation,
            FakeProvider(initial, revision),
        )
        restored_result = await run_audience_workflow(
            restored,
            FakeProvider(initial, revision),
        )
        self.assertEqual(restored_result.segments, original_result.segments)
        self.assertEqual(
            restored_result.provider_skips,
            original_result.provider_skips,
        )
        self.assertEqual(
            restored_result.dropped_decisions,
            original_result.dropped_decisions,
        )
        self.assertEqual(
            restored_result.initial_validation_report,
            original_result.initial_validation_report,
        )
        self.assertEqual(
            restored_result.revision_validation_report,
            original_result.revision_validation_report,
        )
        self.assertEqual(restored_result.metrics, original_result.metrics)
        self.assertEqual(
            restored_result.is_publishable,
            original_result.is_publishable,
        )
        self.assertEqual(
            build_audience_decision_traces(restored, restored_result),
            build_audience_decision_traces(preparation, original_result),
        )
        self.assertEqual(
            [segment.topic_cluster_ids for segment in original_result.segments],
            [["create"], ["revised-valid"]],
        )
        self.assertEqual(
            [item.cluster.id for item in original_result.provider_skips],
            ["source-skip"],
        )
        self.assertEqual(
            [item.cluster_id for item in original_result.dropped_decisions],
            ["revision-drop", "unknown"],
        )
        self.assertIs(
            original_result.dropped_decisions[0].source_cluster.__class__,
            TopicCluster,
        )
        self.assertIsNone(
            original_result.dropped_decisions[1].source_cluster
        )

        _, _, _, output = await self._invoke(
            restored,
            FakeProvider(initial, revision),
        )
        self.assertEqual(
            [record["cluster_id"] for record in output["records"]],
            ["create", "revised-valid"],
        )
        self.assertEqual(
            [item["cluster_id"] for item in output["provider_skips"]],
            ["skip"],
        )
        self.assertEqual(
            [item["cluster_id"] for item in output["validation_drops"]],
            ["revision-drop", "unknown"],
        )
        revised_trace = next(
            trace
            for trace in output["traces"]
            if trace["cluster_id"] == "revised-valid"
        )
        self.assertIn(
            "revision_requested",
            [event["code"] for event in revised_trace["events"]],
        )

    async def test_noncanonical_prepared_ids_remain_terminal_and_exact(
        self,
    ) -> None:
        long_cluster_id = "x" * 129
        source = prepare_audience_clusters(
            [
                make_cluster("canonical"),
                make_cluster("alias-source"),
                make_cluster("long-source"),
                make_cluster("skip"),
            ],
            total_analyzed_views=2_000,
        )
        prepared_ids = (
            "canonical",
            "alias id",
            long_cluster_id,
            "skip",
        )
        preparation = AudiencePreparation(
            clusters=tuple(
                PreparedAudienceCluster(
                    cluster=prepared.cluster,
                    context=prepared.context,
                    cluster_id=prepared_id,
                    cluster_pageviews=prepared.cluster_pageviews,
                    evidence_reference_ids=(
                        prepared.evidence_reference_ids
                    ),
                    resolution_map=prepared.resolution_map,
                )
                for prepared, prepared_id in zip(
                    source.clusters,
                    prepared_ids,
                    strict=True,
                )
            ),
            total_analyzed_views=source.total_analyzed_views,
            reference_cluster_ids=source.reference_cluster_ids,
        )
        initial = response(
            make_create("canonical"),
            make_skip("alias-source"),
            make_skip("long-source"),
            make_skip("skip"),
        )
        revision = response(
            make_skip("alias-source"),
            make_skip("long-source"),
        )

        snapshot = snapshot_preparation(preparation)
        restored = _restore_preparation(snapshot.model_dump(mode="json"))
        self.assertEqual(
            tuple(item.cluster_id for item in snapshot.clusters),
            prepared_ids,
        )
        self.assertEqual(
            tuple(item.cluster_id for item in restored.clusters),
            prepared_ids,
        )

        standard_provider = FakeProvider(initial, revision)
        standard = await run_audience_workflow(
            preparation,
            standard_provider,
        )
        self.assertEqual(standard_provider.generate_calls, 1)
        self.assertEqual(standard_provider.revise_calls, 1)
        self.assertEqual(
            [segment.topic_cluster_ids for segment in standard.segments],
            [["canonical"]],
        )
        self.assertEqual(len(standard.provider_skips), 1)
        self.assertEqual(
            standard.provider_skips[0].cluster,
            source.clusters[3].cluster,
        )
        self.assertEqual(
            standard.provider_skips[0].reason,
            make_skip("skip").reason,
        )
        expected_drop_ids = [
            "alias id",
            long_cluster_id,
            "alias-source",
            "long-source",
            "alias-source",
            "long-source",
        ]
        self.assertEqual(
            [drop.cluster_id for drop in standard.dropped_decisions],
            expected_drop_ids,
        )

        _, _, _, output = await self._invoke(
            preparation,
            FakeProvider(initial, revision),
        )
        self.assertEqual(
            [record["cluster_id"] for record in output["records"]],
            ["canonical"],
        )
        self.assertEqual(
            output["__interrupt__"][0].value["cluster_id"],
            "canonical",
        )
        self.assertEqual(
            output["provider_skips"],
            [
                {
                    "cluster_id": "skip",
                    "cluster_name": standard.provider_skips[0].cluster.name,
                    "reason": standard.provider_skips[0].reason,
                }
            ],
        )
        self.assertEqual(
            output["validation_drops"],
            [
                {
                    "cluster_id": drop.cluster_id,
                    "cluster_name": (
                        drop.source_cluster.name
                        if drop.source_cluster is not None
                        else None
                    ),
                    "source_known": drop.source_cluster is not None,
                    "phase": drop.phase,
                    "drop_code": drop.drop_code,
                    "issue_codes": [
                        issue.code for issue in drop.issues
                    ],
                }
                for drop in standard.dropped_decisions
            ],
        )
        traces_by_cluster = {
            trace["cluster_id"]: trace for trace in output["traces"]
        }
        self.assertEqual(
            traces_by_cluster["alias id"]["final_outcome"],
            "validation_dropped",
        )
        self.assertEqual(
            traces_by_cluster[long_cluster_id]["cluster_id"],
            long_cluster_id,
        )
        self.assertEqual(
            traces_by_cluster[long_cluster_id]["final_outcome"],
            "validation_dropped",
        )
        self.assertNotIn(
            "alias id",
            [record["cluster_id"] for record in output["records"]],
        )
        self.assertNotIn(
            long_cluster_id,
            [record["cluster_id"] for record in output["records"]],
        )

    async def test_equal_distinct_resolution_articles_round_trip_exactly(
        self,
    ) -> None:
        source = prepare_audience_clusters(
            [make_cluster("copied")],
            total_analyzed_views=1_000,
        )
        prepared = source.clusters[0]
        copied_resolution = {
            reference_id: article.model_copy(deep=True)
            for reference_id, article in prepared.resolution_map.items()
        }
        self.assertTrue(
            all(
                copied_resolution[reference_id] == article
                and copied_resolution[reference_id] is not article
                for reference_id, article in prepared.resolution_map.items()
            )
        )
        distinct = AudiencePreparation(
            clusters=(
                PreparedAudienceCluster(
                    cluster=prepared.cluster,
                    context=prepared.context,
                    cluster_id=prepared.cluster_id,
                    cluster_pageviews=prepared.cluster_pageviews,
                    evidence_reference_ids=prepared.evidence_reference_ids,
                    resolution_map=copied_resolution,
                ),
            ),
            total_analyzed_views=source.total_analyzed_views,
            reference_cluster_ids=source.reference_cluster_ids,
        )

        snapshot = snapshot_preparation(distinct)
        restored = _restore_preparation(snapshot.model_dump(mode="json"))
        original_result = await run_audience_workflow(
            distinct,
            FakeProvider(response(make_create("copied"))),
        )
        restored_result = await run_audience_workflow(
            restored,
            FakeProvider(response(make_create("copied"))),
        )
        _, _, _, output = await self._invoke(
            distinct,
            FakeProvider(response(make_create("copied"))),
        )

        self.assertEqual(
            dict(restored.reference_cluster_ids),
            dict(distinct.reference_cluster_ids),
        )
        self.assertEqual(
            list(restored.clusters[0].resolution_map),
            list(distinct.clusters[0].resolution_map),
        )
        self.assertEqual(
            restored.clusters[0].evidence_reference_ids,
            distinct.clusters[0].evidence_reference_ids,
        )
        self.assertEqual(
            [segment.model_dump(mode="json") for segment in restored_result.segments],
            [segment.model_dump(mode="json") for segment in original_result.segments],
        )
        payload = output["__interrupt__"][0].value
        self.assertEqual(
            payload["recommendation"]["supporting_article_reference_ids"],
            list(prepared.evidence_reference_ids),
        )
        self.assertEqual(
            [item["article"]["title"] for item in payload["evidence"]],
            [
                copied_resolution[reference_id].title
                for reference_id in prepared.evidence_reference_ids
            ],
        )

    async def test_non_derived_authoritative_reference_owners_are_preserved(
        self,
    ) -> None:
        source = prepare_audience_clusters(
            [make_cluster("first"), make_cluster("second")],
            total_analyzed_views=1_000,
        )
        owners = dict(source.reference_cluster_ids)
        owners["first:a0"] = "external-owner"
        authoritative = AudiencePreparation(
            clusters=source.clusters,
            total_analyzed_views=source.total_analyzed_views,
            reference_cluster_ids=owners,
        )

        snapshot = snapshot_preparation(authoritative)
        restored = _restore_preparation(snapshot.model_dump(mode="json"))
        initial = response(make_create("first"), make_create("second"))
        revision = response(make_create("first"))
        original_result = await run_audience_workflow(
            authoritative,
            FakeProvider(initial, revision),
        )
        restored_result = await run_audience_workflow(
            restored,
            FakeProvider(initial, revision),
        )

        self.assertEqual(
            list(restored.reference_cluster_ids.items()),
            list(owners.items()),
        )
        self.assertEqual(
            snapshot.clusters[0].resolution[0].owning_cluster_id,
            "external-owner",
        )
        self.assertEqual(
            [segment.model_dump(mode="json") for segment in original_result.segments],
            [segment.model_dump(mode="json") for segment in restored_result.segments],
        )
        self.assertEqual(
            [
                (
                    dropped.cluster_id,
                    dropped.drop_code,
                    tuple(issue.code for issue in dropped.issues),
                )
                for dropped in original_result.dropped_decisions
            ],
            [
                (
                    dropped.cluster_id,
                    dropped.drop_code,
                    tuple(issue.code for issue in dropped.issues),
                )
                for dropped in restored_result.dropped_decisions
            ],
        )

    async def test_candidate_validated_by_revision_enters_review_queue(self) -> None:
        preparation = prepare_audience_clusters(
            [make_cluster("revised")],
            total_analyzed_views=1_000,
        )
        invalid = make_create(
            "revised",
            references=["revised:a0", "missing:a9"],
        )
        _, _, _, output = await self._invoke(
            preparation,
            FakeProvider(
                response(invalid),
                response(make_create("revised")),
            ),
        )
        self.assertEqual(output["records"][0]["status"], "pending_review")
        self.assertEqual(output["metrics"]["revision_count"], 1)
        event_codes = [event["code"] for event in output["traces"][0]["events"]]
        self.assertIn("revision_requested", event_codes)
        self.assertEqual(event_codes.count("review_requested"), 1)


if __name__ == "__main__":
    unittest.main()
