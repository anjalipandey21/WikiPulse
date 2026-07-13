"""Focused tests for deterministic audience preparation and finalization."""

from datetime import date
import unittest

from app.agent.audience_finalization import (
    CLUSTER_VIEWS_EXCEED_TOTAL,
    CROSS_CLUSTER_SUPPORTING_REFERENCE,
    DUPLICATE_CLUSTER_DECISION,
    DUPLICATE_SOURCE_ARTICLE,
    DUPLICATE_SOURCE_CLUSTER_ID,
    DUPLICATE_SUPPORTING_REFERENCE,
    INVALID_TOTAL_ANALYZED_VIEWS,
    MAX_CONTEXT_SUMMARY_CHARACTERS,
    MISSING_CLUSTER_DECISION,
    SOURCE_ARTICLE_COUNT_MISMATCH,
    SOURCE_CLUSTER_TOO_SMALL,
    SOURCE_PAGEVIEWS_MISMATCH,
    TOO_FEW_SUPPORTING_REFERENCES,
    TOO_MANY_SUPPORTING_REFERENCES,
    UNKNOWN_DECISION_CLUSTER,
    UNKNOWN_SUPPORTING_REFERENCE,
    AudienceSourceIntegrityError,
    finalize_audience_decisions,
    prepare_audience_clusters,
)
from app.models import Article, TopicCluster
from app.models.audience_generation import (
    AudienceGenerationResponse,
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


def make_cluster(
    cluster_id: str,
    article_specs: list[tuple[str, int]] | None = None,
) -> TopicCluster:
    articles = [
        make_article(title, views)
        for title, views in (
            article_specs
            or [(f"{cluster_id} Alpha", 200), (f"{cluster_id} Beta", 100)]
        )
    ]
    return TopicCluster(
        id=cluster_id,
        name=f"Topic {cluster_id}",
        articles=articles,
        keywords=[cluster_id, "example topic"],
        total_views=sum(article.weekly_views for article in articles),
        article_count=len(articles),
        confidence_score=0.8,
    )


def make_create_decision(
    cluster_id: str,
    references: list[str],
) -> CreateAudienceDecision:
    return CreateAudienceDecision(
        decision="create_audience",
        cluster_id=cluster_id,
        name=f"{cluster_id.title()} Followers",
        description=(
            "People following this coherent topic and its related developments."
        ),
        supporting_article_reference_ids=references,
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


class AudiencePreparationTests(unittest.TestCase):
    def test_normalizes_and_truncates_long_summary_context(self) -> None:
        cluster = make_cluster("long-summary")
        long_summary = "  Alpha\n\tbeta  " + "gamma delta  " * 150
        cluster.articles[0].summary = long_summary

        preparation = prepare_audience_clusters(
            [cluster],
            total_analyzed_views=1_000,
        )

        compact_summary = preparation.contexts[0].articles[0].summary
        normalized_summary = " ".join(long_summary.split())
        expected_summary = normalized_summary[
            :MAX_CONTEXT_SUMMARY_CHARACTERS
        ].rstrip()
        self.assertGreater(len(normalized_summary), 1_000)
        self.assertEqual(compact_summary, expected_summary)
        self.assertLessEqual(len(compact_summary or ""), 1_000)
        self.assertNotIn("\n", compact_summary or "")
        self.assertNotIn("\t", compact_summary or "")
        self.assertEqual(cluster.articles[0].summary, long_summary)

    def test_selects_top_five_and_builds_stable_immutable_mappings(self) -> None:
        cluster = make_cluster(
            "evidence",
            [
                ("Gamma", 300),
                ("Beta", 200),
                ("Alpha", 200),
                ("Epsilon", 100),
                ("Delta", 100),
                ("Zeta", 50),
            ],
        )

        preparation = prepare_audience_clusters(
            [cluster],
            total_analyzed_views=2_000,
        )
        prepared = preparation.clusters[0]

        self.assertEqual(
            [article.title for article in prepared.context.articles],
            ["Gamma", "Alpha", "Beta", "Delta", "Epsilon"],
        )
        self.assertEqual(
            list(prepared.resolution_map),
            [f"evidence:a{index}" for index in range(5)],
        )
        self.assertIs(prepared.resolution_map["evidence:a0"], cluster.articles[0])
        self.assertNotIn(cluster.articles[-1], prepared.resolution_map.values())
        self.assertEqual(preparation.contexts, (prepared.context,))

        with self.assertRaises(TypeError):
            prepared.resolution_map["evidence:a5"] = (  # type: ignore[index]
                cluster.articles[-1]
            )
        with self.assertRaises(TypeError):
            preparation.reference_cluster_ids[
                "other:a0"
            ] = "other"  # type: ignore[index]

    def test_rejects_fatal_source_integrity_errors_with_stable_codes(self) -> None:
        valid = make_cluster("valid")

        for denominator in (0, -1, True, 1.5):
            with self.subTest(denominator=denominator):
                with self.assertRaises(AudienceSourceIntegrityError) as raised:
                    prepare_audience_clusters(
                        [valid],
                        total_analyzed_views=denominator,  # type: ignore[arg-type]
                    )
                self.assertEqual(raised.exception.code, INVALID_TOTAL_ANALYZED_VIEWS)

        single = make_cluster("single", [("Only article", 50)])
        wrong_count = make_cluster("wrong-count")
        wrong_count.article_count = 3
        wrong_views = make_cluster("wrong-views")
        wrong_views.total_views += 1

        fatal_cases = [
            ([valid, valid], DUPLICATE_SOURCE_CLUSTER_ID, 1_000),
            ([single], SOURCE_CLUSTER_TOO_SMALL, 1_000),
            ([wrong_count], SOURCE_ARTICLE_COUNT_MISMATCH, 1_000),
            ([wrong_views], SOURCE_PAGEVIEWS_MISMATCH, 1_000),
            ([valid], CLUSTER_VIEWS_EXCEED_TOTAL, valid.total_views - 1),
        ]
        for clusters, expected_code, denominator in fatal_cases:
            with self.subTest(code=expected_code):
                with self.assertRaises(AudienceSourceIntegrityError) as raised:
                    prepare_audience_clusters(
                        clusters,
                        total_analyzed_views=denominator,
                    )
                self.assertEqual(raised.exception.code, expected_code)

        first = make_cluster("first")
        shared = first.articles[0]
        second_article = make_article("Unique second article", 50)
        second = TopicCluster(
            id="second",
            name="Second topic",
            articles=[shared, second_article],
            keywords=["second"],
            total_views=shared.weekly_views + second_article.weekly_views,
            article_count=2,
            confidence_score=0.8,
        )
        with self.assertRaises(AudienceSourceIntegrityError) as raised:
            prepare_audience_clusters(
                [first, second],
                total_analyzed_views=1_000,
            )
        self.assertEqual(raised.exception.code, DUPLICATE_SOURCE_ARTICLE)


class AudienceFinalizationTests(unittest.TestCase):
    def test_uses_prepared_snapshots_after_source_and_context_mutation(
        self,
    ) -> None:
        cluster = make_cluster(
            "stable",
            [("Lower", 100), ("Higher", 300), ("Middle", 200)],
        )
        lower, higher, _ = cluster.articles
        preparation = prepare_audience_clusters(
            [cluster],
            total_analyzed_views=1_200,
        )
        prepared = preparation.clusters[0]

        cluster.id = "mutated"
        cluster.total_views = 1_200
        cluster.articles.reverse()
        prepared.context.articles.reverse()
        response = AudienceGenerationResponse(
            decisions=[
                make_create_decision(
                    "stable",
                    ["stable:a2", "stable:a0"],
                )
            ]
        )

        report = finalize_audience_decisions(preparation, response)

        self.assertTrue(report.is_publishable)
        self.assertEqual(len(report.valid_segments), 1)
        segment = report.valid_segments[0]
        self.assertEqual(segment.id, "audience-stable")
        self.assertEqual(segment.topic_cluster_ids, ["stable"])
        self.assertAlmostEqual(segment.size_index, 50.0)
        self.assertEqual(report.metrics.represented_cluster_pageviews, 600)
        self.assertEqual(report.metrics.total_analyzed_pageviews, 1_200)
        self.assertEqual(segment.supporting_articles, [higher, lower])
        self.assertIs(segment.supporting_articles[0], higher)
        self.assertIs(segment.supporting_articles[1], lower)

    def test_builds_segments_in_routing_and_prepared_evidence_order(self) -> None:
        first = make_cluster(
            "first",
            [("First lower", 100), ("First higher", 300), ("First middle", 200)],
        )
        second = make_cluster("second")
        preparation = prepare_audience_clusters(
            [first, second],
            total_analyzed_views=1_200,
        )
        response = AudienceGenerationResponse(
            decisions=[
                make_create_decision("second", ["second:a1", "second:a0"]),
                make_create_decision("first", ["first:a2", "first:a0"]),
            ]
        )

        report = finalize_audience_decisions(preparation, response)

        self.assertTrue(report.is_publishable)
        self.assertEqual(
            [segment.id for segment in report.valid_segments],
            ["audience-first", "audience-second"],
        )
        first_segment = report.valid_segments[0]
        self.assertEqual(first_segment.topic_cluster_ids, ["first"])
        self.assertAlmostEqual(first_segment.size_index, 50.0)
        self.assertEqual(
            [article.title for article in first_segment.supporting_articles],
            ["First higher", "First lower"],
        )
        self.assertIs(first_segment.supporting_articles[0], first.articles[1])
        self.assertIs(first_segment.supporting_articles[1], first.articles[0])
        self.assertEqual(
            first_segment.commercial_confidence_reason,
            response.decisions[1].commercial_confidence_reason,
        )
        self.assertEqual(report.provider_skips, ())
        self.assertEqual(report.invalid_decisions, ())
        self.assertEqual(report.metrics.eligible_cluster_count, 2)
        self.assertEqual(report.metrics.received_decision_count, 2)
        self.assertEqual(report.metrics.valid_decision_count, 2)
        self.assertEqual(report.metrics.created_segment_count, 2)
        self.assertEqual(report.metrics.supporting_reference_count, 4)
        self.assertEqual(report.metrics.unique_supporting_article_count, 4)
        self.assertEqual(report.metrics.represented_cluster_pageviews, 900)
        self.assertEqual(report.metrics.total_analyzed_pageviews, 1_200)
        self.assertEqual(dict(report.metrics.issue_counts_by_code), {})

    def test_records_provider_skips_without_creating_segments(self) -> None:
        first = make_cluster("first")
        second = make_cluster("second")
        preparation = prepare_audience_clusters(
            [first, second],
            total_analyzed_views=1_000,
        )
        response = AudienceGenerationResponse(
            decisions=[
                make_skip_decision("second"),
                make_skip_decision("first"),
            ]
        )

        report = finalize_audience_decisions(preparation, response)

        self.assertTrue(report.is_publishable)
        self.assertEqual(report.valid_segments, ())
        self.assertEqual(
            [skipped.cluster.id for skipped in report.provider_skips],
            ["first", "second"],
        )
        self.assertEqual(report.metrics.valid_decision_count, 2)
        self.assertEqual(report.metrics.provider_skip_count, 2)
        self.assertEqual(report.metrics.represented_cluster_pageviews, 0)

    def test_retains_valid_results_and_reports_each_invalid_cluster(self) -> None:
        clusters = [
            make_cluster(cluster_id)
            for cluster_id in ("valid", "skip", "invalid", "missing", "duplicate")
        ]
        preparation = prepare_audience_clusters(
            clusters,
            total_analyzed_views=2_000,
        )
        response = AudienceGenerationResponse(
            decisions=[
                make_skip_decision("duplicate"),
                make_create_decision("valid", ["valid:a1", "valid:a0"]),
                make_create_decision(
                    "invalid",
                    ["skip:a0", "invalid:a99"],
                ),
                make_skip_decision("skip"),
                make_create_decision(
                    "duplicate",
                    ["duplicate:a0", "duplicate:a1"],
                ),
                make_skip_decision("unknown"),
            ]
        )

        report = finalize_audience_decisions(preparation, response)

        self.assertFalse(report.is_publishable)
        self.assertEqual(
            [segment.id for segment in report.valid_segments],
            ["audience-valid"],
        )
        self.assertEqual(
            [skipped.cluster.id for skipped in report.provider_skips],
            ["skip"],
        )
        self.assertEqual(
            [invalid.cluster_id for invalid in report.invalid_decisions],
            ["invalid", "missing", "duplicate", "unknown"],
        )
        issues_by_cluster = {
            invalid.cluster_id: [issue.code for issue in invalid.issues]
            for invalid in report.invalid_decisions
        }
        self.assertEqual(
            issues_by_cluster,
            {
                "invalid": [
                    CROSS_CLUSTER_SUPPORTING_REFERENCE,
                    UNKNOWN_SUPPORTING_REFERENCE,
                ],
                "missing": [MISSING_CLUSTER_DECISION],
                "duplicate": [DUPLICATE_CLUSTER_DECISION],
                "unknown": [UNKNOWN_DECISION_CLUSTER],
            },
        )
        self.assertIsNone(report.invalid_decisions[-1].source_cluster)
        self.assertEqual(len(report.invalid_decisions[2].decisions), 2)

        metrics = report.metrics
        self.assertEqual(metrics.eligible_cluster_count, 5)
        self.assertEqual(metrics.received_decision_count, 6)
        self.assertEqual(metrics.valid_decision_count, 2)
        self.assertEqual(metrics.invalid_decision_count, 4)
        self.assertEqual(metrics.created_segment_count, 1)
        self.assertEqual(metrics.provider_skip_count, 1)
        self.assertEqual(metrics.unresolved_source_cluster_count, 3)
        self.assertEqual(metrics.supporting_reference_count, 2)
        self.assertEqual(metrics.unique_supporting_article_count, 2)
        self.assertEqual(metrics.represented_cluster_pageviews, 300)
        self.assertEqual(metrics.issue_count, 5)
        self.assertEqual(
            dict(metrics.issue_counts_by_code),
            {
                CROSS_CLUSTER_SUPPORTING_REFERENCE: 1,
                DUPLICATE_CLUSTER_DECISION: 1,
                MISSING_CLUSTER_DECISION: 1,
                UNKNOWN_DECISION_CLUSTER: 1,
                UNKNOWN_SUPPORTING_REFERENCE: 1,
            },
        )
        with self.assertRaises(TypeError):
            metrics.issue_counts_by_code["new_issue"] = 1  # type: ignore[index]

    def test_revalidates_mutable_supporting_reference_lists(self) -> None:
        cluster = make_cluster(
            "mutable",
            [(f"Mutable {index}", 100 - index) for index in range(5)],
        )
        preparation = prepare_audience_clusters(
            [cluster],
            total_analyzed_views=1_000,
        )

        cases = [
            (
                ["mutable:a0", "mutable:a0"],
                [DUPLICATE_SUPPORTING_REFERENCE],
            ),
            (["mutable:a0"], [TOO_FEW_SUPPORTING_REFERENCES]),
            (
                [
                    "mutable:a0",
                    "mutable:a1",
                    "mutable:a2",
                    "mutable:a3",
                    "mutable:a4",
                    "mutable:a99",
                ],
                [TOO_MANY_SUPPORTING_REFERENCES, UNKNOWN_SUPPORTING_REFERENCE],
            ),
        ]
        for mutated_references, expected_codes in cases:
            with self.subTest(references=mutated_references):
                decision = make_create_decision(
                    "mutable",
                    ["mutable:a0", "mutable:a1"],
                )
                decision.supporting_article_reference_ids[:] = mutated_references
                response = AudienceGenerationResponse(decisions=[decision])

                report = finalize_audience_decisions(preparation, response)

                self.assertFalse(report.is_publishable)
                self.assertEqual(report.valid_segments, ())
                self.assertEqual(
                    [issue.code for issue in report.invalid_decisions[0].issues],
                    expected_codes,
                )


if __name__ == "__main__":
    unittest.main()
