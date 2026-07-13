"""Focused mocked tests for the Wikimedia Pageviews client."""

from datetime import date
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from app.services.wikimedia_client import (
    MAX_RETRIES,
    WIKIMEDIA_USER_AGENT,
    WikimediaPageviewsClient,
    WikimediaPageviewsError,
)


def pageviews_payload(title: str = "Example", views: int = 10) -> dict[str, object]:
    return {"items": [{"articles": [{"article": title, "views": views}]}]}


class WikimediaPageviewsClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_fetches_seven_complete_days_and_aggregates_articles(self) -> None:
        requested_paths: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requested_paths.append(request.url.path)
            self.assertEqual(request.headers["User-Agent"], WIKIMEDIA_USER_AGENT)
            self.assertIn(
                "https://github.com/anjalipandey21/WikiPulse",
                request.headers["User-Agent"],
            )
            return httpx.Response(200, json=pageviews_payload("Caf%C3%A9_Test", 10))

        transport = httpx.MockTransport(handler)
        async with WikimediaPageviewsClient(transport=transport) as client:
            articles = await client.fetch_latest_articles(today_utc=date(2026, 7, 12))

        self.assertEqual(len(requested_paths), 7)
        self.assertTrue(requested_paths[0].endswith("/2026/07/11"))
        self.assertTrue(requested_paths[-1].endswith("/2026/07/05"))
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0].normalized_title, "Café Test")
        self.assertEqual(articles[0].weekly_views, 70)
        self.assertEqual(len(articles[0].daily_views), 7)
        self.assertEqual(articles[0].analysis_start_date, date(2026, 7, 5))
        self.assertEqual(articles[0].analysis_end_date, date(2026, 7, 11))

    async def test_uses_latest_available_complete_day(self) -> None:
        requested_paths: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requested_paths.append(request.url.path)
            if request.url.path.endswith("/2026/07/11"):
                return httpx.Response(404)
            return httpx.Response(200, json=pageviews_payload())

        transport = httpx.MockTransport(handler)
        async with WikimediaPageviewsClient(transport=transport) as client:
            articles = await client.fetch_latest_articles(today_utc=date(2026, 7, 12))

        self.assertEqual(len(requested_paths), 8)
        self.assertEqual(articles[0].analysis_start_date, date(2026, 7, 4))
        self.assertEqual(articles[0].analysis_end_date, date(2026, 7, 10))

    async def test_retries_rate_limits_and_transient_server_errors(self) -> None:
        latest_day_attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal latest_day_attempts
            if request.url.path.endswith("/2026/07/11"):
                latest_day_attempts += 1
                if latest_day_attempts == 1:
                    return httpx.Response(429, headers={"Retry-After": "0"})
                if latest_day_attempts == 2:
                    return httpx.Response(503)
            return httpx.Response(200, json=pageviews_payload())

        transport = httpx.MockTransport(handler)
        with patch(
            "app.services.wikimedia_client.asyncio.sleep",
            new_callable=AsyncMock,
        ) as sleep:
            async with WikimediaPageviewsClient(transport=transport) as client:
                await client.fetch_latest_articles(today_utc=date(2026, 7, 12))

        self.assertEqual(latest_day_attempts, 3)
        self.assertEqual(sleep.await_count, 2)

    async def test_stops_after_bounded_retries(self) -> None:
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            return httpx.Response(503)

        transport = httpx.MockTransport(handler)
        with patch(
            "app.services.wikimedia_client.asyncio.sleep",
            new_callable=AsyncMock,
        ):
            async with WikimediaPageviewsClient(transport=transport) as client:
                with self.assertRaisesRegex(
                    WikimediaPageviewsError,
                    "exhausted retries",
                ):
                    await client.fetch_latest_articles(today_utc=date(2026, 7, 12))

        self.assertEqual(attempts, MAX_RETRIES + 1)

    async def test_missing_middle_day_fails_complete_window(self) -> None:
        requested_paths: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requested_paths.append(request.url.path)
            if request.url.path.endswith("/2026/07/08"):
                return httpx.Response(404)
            return httpx.Response(200, json=pageviews_payload())

        transport = httpx.MockTransport(handler)
        async with WikimediaPageviewsClient(transport=transport) as client:
            with self.assertRaisesRegex(
                WikimediaPageviewsError,
                "unavailable for 2026-07-08",
            ):
                await client.fetch_latest_articles(today_utc=date(2026, 7, 12))

        self.assertTrue(requested_paths[-1].endswith("/2026/07/08"))
        self.assertFalse(any(path.endswith("/2026/07/07") for path in requested_paths))

    async def test_wraps_normalization_failures_with_window_context(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=pageviews_payload("___"))

        transport = httpx.MockTransport(handler)
        async with WikimediaPageviewsClient(transport=transport) as client:
            with self.assertRaisesRegex(
                WikimediaPageviewsError,
                "2026-07-05 through 2026-07-11.*must not be empty",
            ) as raised:
                await client.fetch_latest_articles(today_utc=date(2026, 7, 12))

        self.assertIsInstance(raised.exception.__cause__, ValueError)

    async def test_retries_transport_error(self) -> None:
        latest_day_attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal latest_day_attempts
            if request.url.path.endswith("/2026/07/11"):
                latest_day_attempts += 1
                if latest_day_attempts == 1:
                    raise httpx.ConnectError("connection failed", request=request)
            return httpx.Response(200, json=pageviews_payload())

        transport = httpx.MockTransport(handler)
        with patch(
            "app.services.wikimedia_client.asyncio.sleep",
            new_callable=AsyncMock,
        ) as sleep:
            async with WikimediaPageviewsClient(transport=transport) as client:
                await client.fetch_latest_articles(today_utc=date(2026, 7, 12))

        self.assertEqual(latest_day_attempts, 2)
        sleep.assert_awaited_once_with(0.25)

    async def test_rejects_malformed_payload(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"unexpected": []})

        transport = httpx.MockTransport(handler)
        async with WikimediaPageviewsClient(transport=transport) as client:
            with self.assertRaisesRegex(
                WikimediaPageviewsError,
                "payload has no items for 2026-07-11",
            ):
                await client.fetch_latest_articles(today_utc=date(2026, 7, 12))
