"""Focused validation tests for the public data models."""

from datetime import date
import unittest

from pydantic import ValidationError

from app.models import Article, AudienceSegment


def article_data() -> dict[str, object]:
    """Return valid Article input for targeted validation tests."""
    return {
        "title": "Example",
        "normalized_title": "Example",
        "url": "https://en.wikipedia.org/wiki/Example",
        "weekly_views": 10,
        "daily_views": {date(2026, 7, 6): 10},
        "analysis_start_date": date(2026, 7, 6),
        "analysis_end_date": date(2026, 7, 12),
    }


def audience_segment_data() -> dict[str, object]:
    """Return valid AudienceSegment input for targeted validation tests."""
    article = Article(**article_data())
    return {
        "id": "segment-1",
        "name": "Example audience",
        "description": "An example audience segment.",
        "topic_cluster_ids": ["topic-1"],
        "size_index": 50,
        "buying_power": "medium",
        "buying_power_reason": "Representative test data.",
        "brand_categories": ["Example"],
        "supporting_articles": [article],
        "commercial_confidence": 0.5,
        "commercial_confidence_reason": "Representative confidence rationale.",
    }


class ModelValidationTests(unittest.TestCase):
    def test_article_daily_views_reject_negative_values(self) -> None:
        data = article_data()
        data["daily_views"] = {date(2026, 7, 6): -1}

        with self.assertRaises(ValidationError):
            Article(**data)

        data["daily_views"] = {date(2026, 7, 6): 0}
        self.assertEqual(Article(**data).daily_views[date(2026, 7, 6)], 0)

    def test_article_rejects_start_date_later_than_end_date(self) -> None:
        data = article_data()
        data["analysis_start_date"] = date(2026, 7, 13)

        with self.assertRaises(ValidationError):
            Article(**data)

        data["analysis_start_date"] = data["analysis_end_date"]
        article = Article(**data)
        self.assertEqual(article.analysis_start_date, article.analysis_end_date)

    def test_audience_segment_size_index_is_between_zero_and_one_hundred(
        self,
    ) -> None:
        data = audience_segment_data()

        for valid_size_index in (0, 100):
            with self.subTest(size_index=valid_size_index):
                data["size_index"] = valid_size_index
                segment = AudienceSegment(**data)
                self.assertEqual(segment.size_index, valid_size_index)

        for invalid_size_index in (-0.1, 100.1):
            with self.subTest(size_index=invalid_size_index):
                data["size_index"] = invalid_size_index
                with self.assertRaises(ValidationError):
                    AudienceSegment(**data)
