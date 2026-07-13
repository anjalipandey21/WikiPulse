"""Focused tests for deterministic topic-cluster finalization."""

from datetime import date
import unittest

from app.clustering.keyword_extraction import ArticleKeywords
from app.clustering.semantic_clustering import CandidateTopicCluster
from app.clustering.topic_finalization import finalize_topic_clusters
from app.models import Article, TopicCluster


def make_article(title: str, views: int) -> Article:
    observed_date = date(2026, 7, 12)
    return Article(
        title=title,
        normalized_title=title,
        url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
        weekly_views=views,
        daily_views={observed_date: views},
        summary=f"Summary for {title}",
        analysis_start_date=date(2026, 7, 6),
        analysis_end_date=observed_date,
    )


def make_candidate(
    candidate_id: str,
    articles: tuple[Article, ...],
    *,
    mean_similarity: float = 0.9,
    minimum_similarity: float = 0.8,
) -> CandidateTopicCluster:
    cohesion = 0.8 * mean_similarity + 0.2 * minimum_similarity
    return CandidateTopicCluster(
        id=candidate_id,
        articles=articles,
        mean_similarity=mean_similarity,
        minimum_similarity=minimum_similarity,
        cohesion_score=cohesion,
    )


class TopicFinalizationTests(unittest.TestCase):
    def test_aggregates_keywords_names_topic_and_calculates_totals(self) -> None:
        first = make_article("World Cup final", 200)
        second = make_article("Football championship", 300)
        candidate = make_candidate("candidate-football", (first, second))
        keyword_results = [
            ArticleKeywords(
                first.normalized_title,
                ("world cup", "football", "stadium"),
            ),
            ArticleKeywords(
                second.normalized_title,
                ("world cup", "tournament", "football"),
            ),
        ]

        topics = finalize_topic_clusters([candidate], keyword_results)

        self.assertEqual(len(topics), 1)
        topic = topics[0]
        self.assertIsInstance(topic, TopicCluster)
        self.assertEqual(topic.id, candidate.id)
        self.assertEqual(topic.name, "World Cup")
        self.assertEqual(topic.keywords[:2], ["world cup", "football"])
        self.assertEqual(topic.total_views, 500)
        self.assertEqual(topic.article_count, 2)
        self.assertEqual(topic.confidence_score, candidate.cohesion_score)
        self.assertIs(topic.articles[0], first)
        self.assertIs(topic.articles[1], second)

    def test_removes_redundant_aggregate_unigrams(self) -> None:
        first = make_article("Quantum computing", 100)
        second = make_article("Quantum algorithm", 90)
        candidate = make_candidate("candidate-quantum", (first, second))
        keyword_results = [
            ArticleKeywords(
                first.normalized_title,
                ("quantum computing", "quantum", "research"),
            ),
            ArticleKeywords(
                second.normalized_title,
                ("quantum computing", "computing", "algorithm"),
            ),
        ]

        topic = finalize_topic_clusters([candidate], keyword_results)[0]

        self.assertIn("quantum computing", topic.keywords)
        self.assertNotIn("quantum", topic.keywords)
        self.assertNotIn("computing", topic.keywords)

    def test_uses_highest_view_article_as_empty_keyword_fallback(self) -> None:
        lower = make_article("Alpha subject", 100)
        higher = make_article("Beta subject", 200)
        candidate = make_candidate("candidate-empty", (lower, higher))

        topic = finalize_topic_clusters(
            [candidate],
            [
                ArticleKeywords(lower.normalized_title, ()),
                ArticleKeywords(higher.normalized_title, ()),
            ],
        )[0]

        self.assertEqual(topic.name, higher.normalized_title)
        self.assertEqual(topic.keywords, [])

    def test_preserves_candidate_order_ids_identity_and_membership(self) -> None:
        first_members = (
            make_article("First A", 100),
            make_article("First B", 90),
        )
        second_members = (
            make_article("Second A", 80),
            make_article("Second B", 70),
        )
        candidates = [
            make_candidate("candidate-first", first_members),
            make_candidate("candidate-second", second_members),
        ]
        keyword_results = [
            ArticleKeywords(article.normalized_title, ("shared keyword",))
            for article in (*first_members, *second_members)
        ]
        keyword_results.append(ArticleKeywords("Unclustered article", ("shared",)))

        topics = finalize_topic_clusters(candidates, keyword_results)

        self.assertEqual(
            [topic.id for topic in topics],
            [candidate.id for candidate in candidates],
        )
        self.assertEqual(len(topics), 2)
        for topic, candidate in zip(topics, candidates, strict=True):
            self.assertEqual(len(topic.articles), len(candidate.articles))
            for topic_article, candidate_article in zip(
                topic.articles,
                candidate.articles,
                strict=True,
            ):
                self.assertIs(topic_article, candidate_article)

    def test_rejects_missing_and_duplicate_keyword_mappings(self) -> None:
        first = make_article("First", 100)
        second = make_article("Second", 90)
        candidate = make_candidate("candidate-example", (first, second))

        with self.assertRaisesRegex(
            ValueError,
            "missing keyword mapping for article Second",
        ):
            finalize_topic_clusters(
                [candidate],
                [ArticleKeywords(first.normalized_title, ("first",))],
            )

        duplicate_results = [
            ArticleKeywords(first.normalized_title, ("first",)),
            ArticleKeywords(first.normalized_title, ("duplicate",)),
            ArticleKeywords(second.normalized_title, ("second",)),
        ]
        with self.assertRaisesRegex(
            ValueError,
            "duplicate keyword mapping for article First",
        ):
            finalize_topic_clusters([candidate], duplicate_results)


if __name__ == "__main__":
    unittest.main()
