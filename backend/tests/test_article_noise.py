"""Focused tests for conservative Wikipedia article noise filtering."""

from datetime import date
import unittest

from app.filtering.article_noise import (
    DISAMBIGUATION_REASON,
    MAIN_PAGE_REASON,
    UNKNOWN_TITLE_REASON,
    filter_noise_articles,
    get_noise_reason,
)
from app.models import Article


def make_article(title: str, weekly_views: int = 100) -> Article:
    observed_date = date(2026, 7, 11)
    return Article(
        title=title,
        normalized_title=title,
        url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
        weekly_views=weekly_views,
        daily_views={observed_date: weekly_views},
        summary=f"Summary for {title}",
        analysis_start_date=date(2026, 7, 5),
        analysis_end_date=observed_date,
    )


class ArticleNoiseTests(unittest.TestCase):
    def test_returns_reasons_for_exact_noise_titles(self) -> None:
        self.assertEqual(get_noise_reason("Main Page"), MAIN_PAGE_REASON)
        self.assertEqual(get_noise_reason("main page"), MAIN_PAGE_REASON)
        self.assertEqual(get_noise_reason("-"), UNKNOWN_TITLE_REASON)

    def test_rejects_explicit_administrative_namespaces(self) -> None:
        expected_reasons = {
            "Special:Search": "administrative_namespace:special",
            "Wikipedia:Featured articles": "administrative_namespace:wikipedia",
            "User talk:Example": "administrative_namespace:user talk",
            "Template:Infobox": "administrative_namespace:template",
            "Category:Physics": "administrative_namespace:category",
            "Portal:Current events": "administrative_namespace:portal",
            "MOS:Capitalization": "administrative_namespace:mos",
            "MOS talk:Capitalization": "administrative_namespace:mos talk",
            "Module:Citation": "administrative_namespace:module",
            "Event:Example": "administrative_namespace:event",
            "Event talk:Example": "administrative_namespace:event talk",
            "TM:Infobox": "administrative_namespace:tm",
            "WP:Manual of Style": "administrative_namespace:wp",
            "Image:Example.jpg": "administrative_namespace:image",
        }

        for title, expected_reason in expected_reasons.items():
            with self.subTest(title=title):
                self.assertEqual(get_noise_reason(title), expected_reason)

    def test_retains_stale_namespace_prefixes(self) -> None:
        stale_prefixes = [
            "Book",
            "Book talk",
            "Education Program",
            "Education Program talk",
            "Gadget",
            "Gadget talk",
            "Gadget definition",
            "Gadget definition talk",
            "Topic",
        ]

        for prefix in stale_prefixes:
            title = f"{prefix}:Example"
            with self.subTest(title=title):
                self.assertIsNone(get_noise_reason(title))

    def test_namespace_detection_uses_exact_text_before_first_colon(self) -> None:
        valid_titles = [
            "Star Trek: Discovery",
            "History: The Musical",
            "Portal (video game)",
            "Special effects: An introduction",
        ]

        for title in valid_titles:
            with self.subTest(title=title):
                self.assertIsNone(get_noise_reason(title))

    def test_rejects_titles_ending_in_disambiguation(self) -> None:
        self.assertEqual(
            get_noise_reason("Mercury (disambiguation)"),
            DISAMBIGUATION_REASON,
        )
        self.assertEqual(
            get_noise_reason("Example (DISAMBIGUATION)"),
            DISAMBIGUATION_REASON,
        )
        self.assertIsNone(get_noise_reason("Disambiguation (film)"))

    def test_keeps_valid_topic_articles(self) -> None:
        valid_titles = [
            "2026 FIFA World Cup",
            "Quantum entanglement",
            "Oppenheimer (film)",
            "Battle of Waterloo",
            "2026 United States elections",
            "Deaths in 2026",
            "List of solar eclipses in the 21st century",
        ]

        articles = [make_article(title) for title in valid_titles]

        self.assertEqual(filter_noise_articles(articles), articles)

    def test_filter_preserves_fields_identity_and_order(self) -> None:
        first = make_article("Quantum entanglement", weekly_views=300)
        noise = make_article("Main Page", weekly_views=200)
        second = make_article("Star Trek: Discovery", weekly_views=100)
        first_before = first.model_dump()
        second_before = second.model_dump()

        filtered = filter_noise_articles([first, noise, second])

        self.assertEqual(filtered, [first, second])
        self.assertIs(filtered[0], first)
        self.assertIs(filtered[1], second)
        self.assertEqual(first.model_dump(), first_before)
        self.assertEqual(second.model_dump(), second_before)


if __name__ == "__main__":
    unittest.main()
