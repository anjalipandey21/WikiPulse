"""Focused tests for deterministic article keyword extraction."""

from datetime import date
import unittest

from app.clustering.keyword_extraction import (
    TITLE_WEIGHT,
    ArticleKeywords,
    extract_article_keywords,
)
from app.models import Article


def make_article(title: str, summary: str | None) -> Article:
    observed_date = date(2026, 7, 11)
    return Article(
        title=title,
        normalized_title=title,
        url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
        weekly_views=100,
        daily_views={observed_date: 100},
        summary=summary,
        analysis_start_date=date(2026, 7, 5),
        analysis_end_date=observed_date,
    )


class KeywordExtractionTests(unittest.TestCase):
    def test_extracts_ranked_keywords_with_title_weight(self) -> None:
        self.assertEqual(TITLE_WEIGHT, 2)
        articles = [
            make_article(
                "Quantum computing",
                "Qubits enable quantum algorithms and new computing methods.",
            ),
            make_article(
                "FIFA World Cup",
                "The football tournament features national teams.",
            ),
        ]

        results = extract_article_keywords(articles, top_k=4)

        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], ArticleKeywords)
        self.assertIn("quantum", results[0].keywords)
        self.assertTrue(
            {"fifa", "world cup", "fifa world"} & set(results[1].keywords)
        )
        self.assertLessEqual(len(results[0].keywords), 4)
        self.assertLessEqual(len(results[1].keywords), 4)

    def test_removes_stop_words_and_duplicate_occurrences(self) -> None:
        article = make_article(
            "The History of Computing",
            "The history of computing is the history of machines and computing.",
        )

        result = extract_article_keywords([article], top_k=10)[0]

        self.assertNotIn("the", result.keywords)
        self.assertNotIn("of", result.keywords)
        self.assertEqual(result.keywords.count("computing"), 1)
        for keyword in result.keywords:
            self.assertFalse(
                keyword == "computing"
                and any(
                    "computing" in other.split()
                    for other in result.keywords
                    if other != keyword
                )
            )

    def test_uses_title_when_summary_is_missing_or_blank(self) -> None:
        articles = [
            make_article("Quantum computing", None),
            make_article("Marine biology", "   "),
        ]

        results = extract_article_keywords(articles)

        self.assertEqual(results[0].keywords, ("quantum computing",))
        self.assertEqual(results[1].keywords, ("marine biology",))

    def test_preserves_input_order_without_mutating_articles(self) -> None:
        articles = [
            make_article("Gamma rays", "High-energy electromagnetic radiation."),
            make_article("Alpha particle", "A helium nucleus."),
            make_article("Beta decay", "A radioactive decay process."),
        ]
        before = [article.model_dump() for article in articles]

        results = extract_article_keywords(articles)

        self.assertEqual(
            [result.normalized_title for result in results],
            [article.normalized_title for article in articles],
        )
        self.assertEqual([article.model_dump() for article in articles], before)

    def test_breaks_equal_score_ties_deterministically(self) -> None:
        article = make_article("Alpha Beta Gamma", None)

        result = extract_article_keywords([article], top_k=2)[0]

        self.assertEqual(result.keywords, ("alpha beta", "beta gamma"))

    def test_handles_empty_and_all_stop_word_inputs(self) -> None:
        self.assertEqual(extract_article_keywords([]), [])

        articles = [
            make_article("The And", "or but if"),
            make_article("Of The", None),
        ]
        results = extract_article_keywords(articles)

        self.assertEqual(
            results,
            [
                ArticleKeywords("The And", ()),
                ArticleKeywords("Of The", ()),
            ],
        )

    def test_rejects_invalid_top_k(self) -> None:
        article = make_article("Quantum computing", None)

        for top_k in (0, -1, True, 1.5, "5"):
            with self.subTest(top_k=top_k):
                with self.assertRaisesRegex(
                    ValueError,
                    "top_k must be a positive integer",
                ):
                    extract_article_keywords([article], top_k=top_k)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
