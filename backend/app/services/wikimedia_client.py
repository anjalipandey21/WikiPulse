"""Async client for Wikimedia's daily top Pageviews endpoint."""

import asyncio
from datetime import date, datetime, timedelta, timezone
from types import TracebackType
from typing import Any, Self

import httpx

from ..filtering.article_normalization import (
    ArticleViewObservation,
    aggregate_articles,
)
from ..models import Article


WIKIMEDIA_BASE_URL = "https://wikimedia.org"
WIKIMEDIA_USER_AGENT = (
    "WikiPulse/0.1 (AI Builder audience-trend project; Wikimedia Pageviews client)"
)
REQUEST_TIMEOUT = httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0)
MAX_RETRIES = 2
MAX_LATEST_DAY_LOOKBACK = 7
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class WikimediaPageviewsError(RuntimeError):
    """Raised when a complete Wikimedia Pageviews window cannot be produced."""


class WikimediaPageviewsClient:
    """Fetch and aggregate the latest seven complete daily Pageviews lists."""

    def __init__(
        self,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=WIKIMEDIA_BASE_URL,
            headers={
                "Accept": "application/json",
                "User-Agent": WIKIMEDIA_USER_AGENT,
            },
            timeout=REQUEST_TIMEOUT,
            transport=transport,
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the underlying HTTP connection pool."""
        await self._client.aclose()

    async def fetch_latest_articles(
        self,
        *,
        today_utc: date | None = None,
    ) -> list[Article]:
        """Return articles aggregated over the latest seven available UTC days."""
        current_utc_date = today_utc or datetime.now(timezone.utc).date()
        latest_candidate = current_utc_date - timedelta(days=1)
        latest_date, latest_payload = await self._find_latest_available_day(
            latest_candidate
        )
        analysis_start_date = latest_date - timedelta(days=6)

        observations = self._parse_observations(latest_payload, latest_date)
        for days_before_latest in range(1, 7):
            observed_date = latest_date - timedelta(days=days_before_latest)
            payload = await self._fetch_day(observed_date)
            if payload is None:
                raise WikimediaPageviewsError(
                    f"Wikimedia Pageviews data is unavailable for {observed_date}"
                )
            observations.extend(self._parse_observations(payload, observed_date))

        try:
            return aggregate_articles(
                observations,
                analysis_start_date=analysis_start_date,
                analysis_end_date=latest_date,
            )
        except ValueError as exc:
            raise WikimediaPageviewsError(
                "Failed to normalize or aggregate Wikimedia Pageviews data for "
                f"{analysis_start_date} through {latest_date}: {exc}"
            ) from exc

    async def _find_latest_available_day(
        self,
        latest_candidate: date,
    ) -> tuple[date, dict[str, Any]]:
        for days_back in range(MAX_LATEST_DAY_LOOKBACK + 1):
            candidate = latest_candidate - timedelta(days=days_back)
            payload = await self._fetch_day(candidate)
            if payload is not None:
                return candidate, payload

        raise WikimediaPageviewsError(
            "No complete Wikimedia Pageviews day was available within "
            f"{MAX_LATEST_DAY_LOOKBACK + 1} checked dates"
        )

    async def _fetch_day(self, observed_date: date) -> dict[str, Any] | None:
        path = (
            "/api/rest_v1/metrics/pageviews/top/"
            "en.wikipedia.org/all-access/"
            f"{observed_date:%Y/%m/%d}"
        )
        response = await self._request_with_retries(path, observed_date)

        if response.status_code == 404:
            return None
        if response.is_error:
            raise WikimediaPageviewsError(
                "Wikimedia Pageviews request failed for "
                f"{observed_date} with HTTP {response.status_code}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise WikimediaPageviewsError(
                f"Wikimedia Pageviews returned invalid JSON for {observed_date}"
            ) from exc

        if not isinstance(payload, dict):
            raise WikimediaPageviewsError(
                f"Wikimedia Pageviews returned an invalid payload for {observed_date}"
            )
        return payload

    async def _request_with_retries(
        self,
        path: str,
        observed_date: date,
    ) -> httpx.Response:
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await self._client.get(path)
            except httpx.TransportError as exc:
                if attempt == MAX_RETRIES:
                    raise WikimediaPageviewsError(
                        f"Wikimedia Pageviews request failed for {observed_date}"
                    ) from exc
                await asyncio.sleep(self._retry_delay_seconds(attempt, None))
                continue

            if response.status_code not in RETRYABLE_STATUS_CODES:
                return response
            if attempt == MAX_RETRIES:
                raise WikimediaPageviewsError(
                    "Wikimedia Pageviews request exhausted retries for "
                    f"{observed_date} with HTTP {response.status_code}"
                )

            await asyncio.sleep(self._retry_delay_seconds(attempt, response))

        raise AssertionError("retry loop exited unexpectedly")

    @staticmethod
    def _retry_delay_seconds(
        attempt: int,
        response: httpx.Response | None,
    ) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    return min(max(float(retry_after), 0.0), 5.0)
                except ValueError:
                    pass
        return 0.25 * (2**attempt)

    @staticmethod
    def _parse_observations(
        payload: dict[str, Any],
        observed_date: date,
    ) -> list[ArticleViewObservation]:
        items = payload.get("items")
        if not isinstance(items, list) or not items or not isinstance(items[0], dict):
            raise WikimediaPageviewsError(
                f"Wikimedia Pageviews payload has no items for {observed_date}"
            )

        articles = items[0].get("articles")
        if not isinstance(articles, list):
            raise WikimediaPageviewsError(
                f"Wikimedia Pageviews payload has no articles for {observed_date}"
            )

        observations: list[ArticleViewObservation] = []
        for article in articles:
            if not isinstance(article, dict):
                raise WikimediaPageviewsError(
                    f"Wikimedia Pageviews returned an invalid article for {observed_date}"
                )
            title = article.get("article")
            views = article.get("views")
            if (
                not isinstance(title, str)
                or not isinstance(views, int)
                or isinstance(views, bool)
            ):
                raise WikimediaPageviewsError(
                    f"Wikimedia Pageviews returned invalid article fields for {observed_date}"
                )
            observations.append(ArticleViewObservation(observed_date, title, views))

        return observations
