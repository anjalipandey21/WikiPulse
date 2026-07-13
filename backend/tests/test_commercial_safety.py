"""Focused tests for deterministic commercial-safety topic routing."""

from datetime import date
import unittest

from app.filtering.commercial_safety import (
    INSUFFICIENT_SUMMARY_COVERAGE_REASON,
    LOW_COHESION_REASON,
    MAX_ELIGIBLE_CLUSTERS,
    MIN_SUMMARY_COVERAGE,
    MIN_TOPIC_COHESION,
    NO_PAGEVIEWS_REASON,
    SENSITIVE_EVENT_REASON,
    TOP_SIX_LIMIT_REASON,
    get_commercial_skip_reason,
    route_commercial_clusters,
)
from app.models import Article, TopicCluster


def make_article(title: str, summary: str | None) -> Article:
    observed_date = date(2026, 7, 12)
    return Article(
        title=title,
        normalized_title=title,
        url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
        weekly_views=100,
        daily_views={observed_date: 100},
        summary=summary,
        analysis_start_date=date(2026, 7, 6),
        analysis_end_date=observed_date,
    )


def make_cluster(
    cluster_id: str,
    name: str,
    *,
    total_views: int = 1_000,
    confidence: float | None = 0.8,
    titles_and_summaries: list[tuple[str, str | None]] | None = None,
    keywords: list[str] | None = None,
) -> TopicCluster:
    source_articles = titles_and_summaries or [
        (f"{name} A", f"{name} A is a notable subject."),
        (f"{name} B", f"{name} B is another notable subject."),
    ]
    articles = [
        make_article(title, summary)
        for title, summary in source_articles
    ]
    return TopicCluster(
        id=cluster_id,
        name=name,
        description=None,
        articles=articles,
        keywords=keywords or [name.casefold()],
        total_views=total_views,
        article_count=len(articles),
        confidence_score=confidence,
    )


class CommercialSafetyTests(unittest.TestCase):
    def test_accepts_safe_coherent_topics(self) -> None:
        clusters = [
            make_cluster("sports", "World Cup", total_views=5_000),
            make_cluster("science", "Mars exploration", total_views=4_000),
            make_cluster("technology", "Artificial intelligence", total_views=3_000),
        ]

        result = route_commercial_clusters(clusters)

        self.assertEqual(result.eligible_clusters, tuple(clusters))
        self.assertEqual(result.skipped_clusters, ())

    def test_rejects_explicit_sensitive_events(self) -> None:
        clusters = [
            make_cluster(
                "shooting",
                "Central City incident",
                titles_and_summaries=[
                    (
                        "2026 Central City mass shooting",
                        "The event was a mass shooting in Central City.",
                    ),
                    ("Central City", "Central City is a municipality."),
                ],
            ),
            make_cluster(
                "bombing",
                "Marathon incident",
                titles_and_summaries=[
                    (
                        "Boston Marathon bombing",
                        "The bombing occurred near the marathon finish line.",
                    ),
                    ("Boston Marathon", "The Boston Marathon is an annual race."),
                ],
            ),
            make_cluster(
                "summary-only",
                "Central Station incident",
                titles_and_summaries=[
                    (
                        "Central Station incident",
                        "The Central Station incident was a terrorist attack.",
                    ),
                    ("Central Station", "Central Station is a transit hub."),
                ],
            ),
        ]

        result = route_commercial_clusters(clusters)

        self.assertEqual(result.eligible_clusters, ())
        self.assertEqual(
            [skipped.reason for skipped in result.skipped_clusters],
            [SENSITIVE_EVENT_REASON] * len(clusters),
        )

    def test_protects_valid_topics_from_broad_keyword_false_positives(self) -> None:
        clusters = [
            make_cluster(
                "entertainment",
                "Horror cinema",
                titles_and_summaries=[
                    (
                        "The Texas Chain Saw Massacre",
                        "The Texas Chain Saw Massacre is a 1974 American horror film.",
                    ),
                    ("Horror film", "A horror film is a film genre."),
                ],
            ),
            make_cluster(
                "history",
                "History of aviation",
                titles_and_summaries=[
                    ("History of aviation", "Aviation history spans many eras."),
                    ("Early flying machines", "Early machines preceded aircraft."),
                ],
            ),
            make_cluster(
                "sports",
                "Shooting sports",
                titles_and_summaries=[
                    ("Shooting sports", "Shooting sports are competitive sports."),
                    ("Olympic shooting", "Olympic shooting is a sporting event."),
                ],
            ),
            make_cluster(
                "science",
                "Earthquake engineering",
                titles_and_summaries=[
                    (
                        "Earthquake engineering",
                        "Earthquake engineering is an engineering discipline.",
                    ),
                    (
                        "Seismology",
                        "Seismology is the scientific study of earthquakes.",
                    ),
                ],
            ),
            make_cluster(
                "technology",
                "Cybersecurity",
                titles_and_summaries=[
                    ("Cyberattack", "A cyberattack targets computer systems."),
                    ("Computer security", "Computer security protects systems."),
                ],
            ),
        ]

        result = route_commercial_clusters(clusters)

        self.assertEqual(
            {cluster.id for cluster in result.eligible_clusters},
            {cluster.id for cluster in clusters},
        )
        self.assertEqual(result.skipped_clusters, ())

    def test_applies_thresholds_with_stable_reason_precedence(self) -> None:
        self.assertEqual(MIN_TOPIC_COHESION, 0.60)
        self.assertEqual(MIN_SUMMARY_COVERAGE, 0.50)

        cohesion_boundary = make_cluster(
            "cohesion-boundary",
            "Boundary topic",
            confidence=MIN_TOPIC_COHESION,
        )
        summary_boundary = make_cluster(
            "summary-boundary",
            "Summary boundary",
            titles_and_summaries=[
                ("Summary present", "A useful summary."),
                ("Summary absent", None),
            ],
        )
        low_cohesion = make_cluster(
            "low-cohesion",
            "Low cohesion",
            confidence=MIN_TOPIC_COHESION - 0.001,
        )
        no_pageviews = make_cluster(
            "no-pageviews",
            "No pageviews",
            total_views=0,
        )
        low_summary_coverage = make_cluster(
            "low-summary",
            "Low summary coverage",
            titles_and_summaries=[
                ("Only summary", "A useful summary."),
                ("Missing one", None),
                ("Missing two", "   "),
            ],
        )
        multiple_failures = make_cluster(
            "precedence",
            "Mass shooting",
            total_views=0,
            confidence=0.1,
            titles_and_summaries=[("Mass shooting", None)],
        )

        self.assertIsNone(get_commercial_skip_reason(cohesion_boundary))
        self.assertIsNone(get_commercial_skip_reason(summary_boundary))
        self.assertEqual(
            get_commercial_skip_reason(low_cohesion),
            LOW_COHESION_REASON,
        )
        self.assertEqual(
            get_commercial_skip_reason(no_pageviews),
            NO_PAGEVIEWS_REASON,
        )
        self.assertEqual(
            get_commercial_skip_reason(low_summary_coverage),
            INSUFFICIENT_SUMMARY_COVERAGE_REASON,
        )
        self.assertEqual(
            get_commercial_skip_reason(multiple_failures),
            LOW_COHESION_REASON,
        )

    def test_ranks_by_pageviews_and_enforces_top_six_limit(self) -> None:
        clusters = [
            make_cluster("beta", "Beta", total_views=600),
            make_cluster("overflow", "Overflow", total_views=200),
            make_cluster("highest", "Highest", total_views=700),
            make_cluster("alpha", "Alpha", total_views=600),
            make_cluster("charlie", "Charlie", total_views=500),
            make_cluster("echo", "Echo", total_views=300),
            make_cluster("delta", "Delta", total_views=400),
        ]

        result = route_commercial_clusters(clusters)

        self.assertEqual(MAX_ELIGIBLE_CLUSTERS, 6)
        self.assertEqual(
            [cluster.id for cluster in result.eligible_clusters],
            ["highest", "alpha", "beta", "charlie", "delta", "echo"],
        )
        self.assertEqual(len(result.skipped_clusters), 1)
        self.assertIs(result.skipped_clusters[0].cluster, clusters[1])
        self.assertEqual(
            result.skipped_clusters[0].reason,
            TOP_SIX_LIMIT_REASON,
        )


if __name__ == "__main__":
    unittest.main()
