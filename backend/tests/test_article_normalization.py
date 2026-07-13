"""Focused tests for deterministic article normalization and aggregation."""

from datetime import date
import unittest

from app.filtering.article_normalization import (
    ArticleViewObservation,
    aggregate_articles,
    normalize_title,
)
from app.models import Article


class ArticleNormalizationTests(unittest.TestCase):
    def test_normalize_title_decodes_once_and_normalizes_text(self) -> None:
        self.assertEqual(normalize_title("  Cafe%CC%81__Society\t"), "Café Society")
        self.assertEqual(normalize_title("100%2525"), "100%25")

    def test_aggregate_articles_sums_observed_views_and_keeps_dates_absent(
        self,
    ) -> None:
        start_date = date(2026, 7, 1)
        observed_date = date(2026, 7, 3)
        end_date = date(2026, 7, 7)
        observations = [
            ArticleViewObservation(start_date, "Caf%C3%A9_Test", 10),
            ArticleViewObservation(start_date, "  Cafe%CC%81 Test ", 5),
            ArticleViewObservation(observed_date, "Café_Test", 7),
        ]

        articles = aggregate_articles(observations, start_date, end_date)

        self.assertEqual(len(articles), 1)
        article = articles[0]
        self.assertIsInstance(article, Article)
        self.assertEqual(article.title, "Café Test")
        self.assertEqual(article.normalized_title, "Café Test")
        self.assertEqual(
            article.url,
            "https://en.wikipedia.org/wiki/Caf%C3%A9_Test",
        )
        self.assertEqual(article.weekly_views, 22)
        self.assertEqual(
            article.daily_views,
            {start_date: 15, observed_date: 7},
        )
        self.assertNotIn(date(2026, 7, 2), article.daily_views)
        self.assertEqual(article.analysis_start_date, start_date)
        self.assertEqual(article.analysis_end_date, end_date)

    def test_aggregate_articles_sorts_by_views_then_normalized_title(self) -> None:
        observed_date = date(2026, 7, 1)
        observations = [
            ArticleViewObservation(observed_date, "Beta", 10),
            ArticleViewObservation(observed_date, "Gamma", 11),
            ArticleViewObservation(observed_date, "Alpha", 10),
        ]

        articles = aggregate_articles(observations, observed_date, observed_date)

        self.assertEqual(
            [article.normalized_title for article in articles],
            ["Gamma", "Alpha", "Beta"],
        )

    def test_aggregate_articles_rejects_negative_raw_observations(self) -> None:
        observed_date = date(2026, 7, 1)
        observations = [
            ArticleViewObservation(observed_date, "Example", 10),
            ArticleViewObservation(observed_date, "Example", -5),
        ]

        with self.assertRaisesRegex(ValueError, "must not be negative"):
            aggregate_articles(observations, observed_date, observed_date)

    def test_normalize_title_rejects_empty_normalized_titles(self) -> None:
        for title in ("", "  \t", "___", "%20_%20"):
            with self.subTest(title=title):
                with self.assertRaisesRegex(ValueError, "must not be empty"):
                    normalize_title(title)

    def test_aggregate_articles_rejects_observations_outside_date_range(
        self,
    ) -> None:
        start_date = date(2026, 7, 2)
        end_date = date(2026, 7, 8)

        for observed_date in (date(2026, 7, 1), date(2026, 7, 9)):
            with self.subTest(observed_date=observed_date):
                observations = [
                    ArticleViewObservation(observed_date, "Example", 10)
                ]
                with self.assertRaisesRegex(ValueError, "outside the analysis range"):
                    aggregate_articles(observations, start_date, end_date)

    def test_aggregate_articles_validates_date_range_for_empty_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be on or before"):
            aggregate_articles(
                [],
                analysis_start_date=date(2026, 7, 2),
                analysis_end_date=date(2026, 7, 1),
            )
