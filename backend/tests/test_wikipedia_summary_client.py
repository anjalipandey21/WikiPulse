"""Focused mocked tests for Wikipedia summary enrichment."""

from datetime import date
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from app.models import Article
from app.services.wikipedia_summary_client import (
    MAX_RETRIES,
    SUMMARY_BATCH_SIZE,
    WIKIPEDIA_USER_AGENT,
    WikipediaSummaryClient,
)


def make_article(index: int, *, normalized_title: str | None = None) -> Article:
    observed_date = date(2026, 7, 11)
    title = normalized_title or f"Article {index:02d}"
    return Article(
        title=title,
        normalized_title=title,
        url=f"https://en.wikipedia.org/wiki/Article_{index:02d}",
        weekly_views=1000 - index,
        daily_views={observed_date: 1000 - index},
        summary=None,
        analysis_start_date=date(2026, 7, 5),
        analysis_end_date=observed_date,
    )


class WikipediaSummaryClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_batches_requests_and_preserves_article_fields_and_order(
        self,
    ) -> None:
        articles = [make_article(index) for index in range(SUMMARY_BATCH_SIZE + 1)]
        requested_batches: list[list[str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.headers["User-Agent"], WIKIPEDIA_USER_AGENT)
            self.assertIn(
                "https://github.com/anjalipandey21/WikiPulse",
                request.headers["User-Agent"],
            )
            self.assertEqual(request.url.params["exsentences"], "2")
            self.assertEqual(request.url.params["explaintext"], "1")
            titles = request.url.params["titles"].split("|")
            requested_batches.append(titles)
            return httpx.Response(
                200,
                json={
                    "query": {
                        "pages": [
                            {
                                "title": title,
                                "extract": f"{title} first sentence. Second sentence.",
                            }
                            for title in titles
                        ]
                    }
                },
            )

        transport = httpx.MockTransport(handler)
        async with WikipediaSummaryClient(transport=transport) as client:
            enriched = await client.enrich_articles(articles)

        self.assertEqual(
            [len(batch) for batch in requested_batches],
            [SUMMARY_BATCH_SIZE, 1],
        )
        self.assertEqual(
            [article.normalized_title for article in enriched],
            [article.normalized_title for article in articles],
        )
        for original, result in zip(articles, enriched, strict=True):
            self.assertEqual(
                result.model_dump(exclude={"summary"}),
                original.model_dump(exclude={"summary"}),
            )
            self.assertEqual(
                result.summary,
                f"{original.normalized_title} first sentence. Second sentence.",
            )

    async def test_follows_normalization_and_redirect_mappings(self) -> None:
        article = make_article(1, normalized_title="example article")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "query": {
                        "normalized": [
                            {"from": "example article", "to": "Example article"}
                        ],
                        "redirects": [
                            {"from": "Example article", "to": "Target article"}
                        ],
                        "pages": [
                            {
                                "pageid": 123,
                                "title": "Target article",
                                "extract": "Target introduction. Another sentence.",
                            }
                        ],
                    }
                },
            )

        transport = httpx.MockTransport(handler)
        async with WikipediaSummaryClient(transport=transport) as client:
            enriched = await client.enrich_articles([article])

        self.assertEqual(
            enriched[0].model_dump(exclude={"summary"}),
            article.model_dump(exclude={"summary"}),
        )
        self.assertEqual(
            enriched[0].summary,
            "Target introduction. Another sentence.",
        )

    async def test_deep_copies_mutable_article_fields(self) -> None:
        article = make_article(1)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "query": {
                        "pages": [
                            {
                                "title": article.normalized_title,
                                "extract": "Introduction. Second sentence.",
                            }
                        ]
                    }
                },
            )

        transport = httpx.MockTransport(handler)
        async with WikipediaSummaryClient(transport=transport) as client:
            enriched = await client.enrich_articles([article])

        self.assertIsNot(enriched[0].daily_views, article.daily_views)
        enriched[0].daily_views[date(2026, 7, 11)] = 1
        self.assertEqual(article.daily_views[date(2026, 7, 11)], 999)

    async def test_preserves_existing_summary_when_extract_is_missing(self) -> None:
        article = make_article(1)
        article.summary = "Existing introduction. Existing second sentence."

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "query": {
                        "pages": [
                            {"title": article.normalized_title, "missing": True}
                        ]
                    }
                },
            )

        transport = httpx.MockTransport(handler)
        async with WikipediaSummaryClient(transport=transport) as client:
            enriched = await client.enrich_articles([article])

        self.assertEqual(enriched[0].summary, article.summary)

    async def test_preserves_existing_summary_when_batch_fails(self) -> None:
        article = make_article(1)
        article.summary = "Existing introduction. Existing second sentence."

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"unexpected": []})

        transport = httpx.MockTransport(handler)
        with self.assertLogs(
            "app.services.wikipedia_summary_client",
            level="WARNING",
        ):
            async with WikipediaSummaryClient(transport=transport) as client:
                enriched = await client.enrich_articles([article])

        self.assertEqual(enriched[0].summary, article.summary)

    async def test_skips_missing_and_malformed_page_entries(self) -> None:
        articles = [make_article(index) for index in range(4)]

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "query": {
                        "normalized": [None, {"from": 7, "to": "Ignored"}],
                        "redirects": "not a list",
                        "pages": [
                            {
                                "title": "Article 00",
                                "extract": "Valid introduction. Second sentence.",
                            },
                            {"title": "Article 01", "missing": True},
                            {"title": "Article 02", "extract": 42},
                            "not a page",
                        ],
                    }
                },
            )

        transport = httpx.MockTransport(handler)
        async with WikipediaSummaryClient(transport=transport) as client:
            enriched = await client.enrich_articles(articles)

        self.assertEqual(
            [article.summary for article in enriched],
            ["Valid introduction. Second sentence.", None, None, None],
        )

    async def test_malformed_batch_is_logged_and_later_batch_continues(self) -> None:
        articles = [make_article(index) for index in range(SUMMARY_BATCH_SIZE + 1)]
        request_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal request_count
            request_count += 1
            if request_count == 1:
                return httpx.Response(200, json={"unexpected": []})
            title = request.url.params["titles"]
            return httpx.Response(
                200,
                json={
                    "query": {
                        "pages": [
                            {
                                "title": title,
                                "extract": "Recovered introduction. Second sentence.",
                            }
                        ]
                    }
                },
            )

        transport = httpx.MockTransport(handler)
        with self.assertLogs(
            "app.services.wikipedia_summary_client",
            level="WARNING",
        ):
            async with WikipediaSummaryClient(transport=transport) as client:
                enriched = await client.enrich_articles(articles)

        self.assertEqual(request_count, 2)
        self.assertTrue(
            all(article.summary is None for article in enriched[:SUMMARY_BATCH_SIZE])
        )
        self.assertEqual(
            enriched[-1].summary,
            "Recovered introduction. Second sentence.",
        )

    async def test_exhausted_retries_are_logged_and_next_batch_continues(
        self,
    ) -> None:
        articles = [make_article(index) for index in range(SUMMARY_BATCH_SIZE + 1)]
        failed_batch_attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal failed_batch_attempts
            titles = request.url.params["titles"].split("|")
            if len(titles) == SUMMARY_BATCH_SIZE:
                failed_batch_attempts += 1
                return httpx.Response(429, headers={"Retry-After": "0"})
            return httpx.Response(
                200,
                json={
                    "query": {
                        "pages": [
                            {
                                "title": titles[0],
                                "extract": "Available introduction. Second sentence.",
                            }
                        ]
                    }
                },
            )

        transport = httpx.MockTransport(handler)
        with (
            patch(
                "app.services.wikipedia_summary_client.asyncio.sleep",
                new_callable=AsyncMock,
            ) as sleep,
            self.assertLogs(
                "app.services.wikipedia_summary_client",
                level="WARNING",
            ),
        ):
            async with WikipediaSummaryClient(transport=transport) as client:
                enriched = await client.enrich_articles(articles)

        self.assertEqual(failed_batch_attempts, MAX_RETRIES + 1)
        self.assertEqual(sleep.await_count, MAX_RETRIES)
        self.assertTrue(
            all(article.summary is None for article in enriched[:SUMMARY_BATCH_SIZE])
        )
        self.assertEqual(
            enriched[-1].summary,
            "Available introduction. Second sentence.",
        )


if __name__ == "__main__":
    unittest.main()
