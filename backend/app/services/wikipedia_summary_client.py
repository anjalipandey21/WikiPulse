"""Async client for deterministic Wikipedia article summary enrichment."""

import asyncio
from collections.abc import Sequence
import logging
from types import TracebackType
from typing import Any, Self

import httpx

from ..models import Article


WIKIPEDIA_BASE_URL = "https://en.wikipedia.org"
WIKIPEDIA_USER_AGENT = (
    "WikiPulse/0.1 (AI Builder audience-trend project; Wikipedia summary client; "
    "https://github.com/anjalipandey21/WikiPulse)"
)
REQUEST_TIMEOUT = httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0)
SUMMARY_BATCH_SIZE = 20
MAX_RETRIES = 2
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

logger = logging.getLogger(__name__)


class WikipediaSummaryError(RuntimeError):
    """Raised when a batch of Wikipedia summaries cannot be fetched."""


class WikipediaSummaryClient:
    """Fetch short introductions and attach them to existing articles."""

    def __init__(
        self,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=WIKIPEDIA_BASE_URL,
            headers={
                "Accept": "application/json",
                "User-Agent": WIKIPEDIA_USER_AGENT,
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

    async def enrich_articles(self, articles: Sequence[Article]) -> list[Article]:
        """Return articles in input order with summaries where available."""
        enriched_articles: list[Article] = []

        for start in range(0, len(articles), SUMMARY_BATCH_SIZE):
            batch = articles[start : start + SUMMARY_BATCH_SIZE]
            titles = [article.normalized_title for article in batch]

            try:
                summaries = await self._fetch_batch(titles)
            except WikipediaSummaryError as exc:
                logger.warning(
                    "Wikipedia summary batch failed for %s: %s",
                    ", ".join(titles),
                    exc,
                )
                summaries = {}

            for article in batch:
                summary = summaries.get(article.normalized_title)
                if summary is None and article.summary and article.summary.strip():
                    summary = article.summary
                enriched_articles.append(
                    article.model_copy(
                        update={"summary": summary},
                        deep=True,
                    )
                )

        return enriched_articles

    async def _fetch_batch(self, titles: list[str]) -> dict[str, str]:
        response = await self._request_with_retries(titles)

        if response.is_error:
            raise WikipediaSummaryError(
                f"request failed with HTTP {response.status_code}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise WikipediaSummaryError("response contained invalid JSON") from exc

        return self._parse_summaries(payload, titles)

    async def _request_with_retries(self, titles: list[str]) -> httpx.Response:
        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "prop": "extracts",
            "exintro": "1",
            "explaintext": "1",
            "exsentences": "2",
            "exlimit": "max",
            "redirects": "1",
            "titles": "|".join(titles),
        }

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await self._client.get("/w/api.php", params=params)
            except httpx.TransportError as exc:
                if attempt == MAX_RETRIES:
                    raise WikipediaSummaryError(
                        "request exhausted retries after a transport error"
                    ) from exc
                await asyncio.sleep(self._retry_delay_seconds(attempt, None))
                continue

            if response.status_code not in RETRYABLE_STATUS_CODES:
                return response
            if attempt == MAX_RETRIES:
                raise WikipediaSummaryError(
                    "request exhausted retries with "
                    f"HTTP {response.status_code}"
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
    def _parse_summaries(
        payload: Any,
        requested_titles: list[str],
    ) -> dict[str, str]:
        if not isinstance(payload, dict):
            raise WikipediaSummaryError("response payload was not an object")

        api_error = payload.get("error")
        if api_error is not None:
            raise WikipediaSummaryError("response contained an API error")

        query = payload.get("query")
        if not isinstance(query, dict):
            raise WikipediaSummaryError("response payload had no query object")

        pages = query.get("pages")
        if not isinstance(pages, list):
            raise WikipediaSummaryError("response payload had no pages list")

        title_mappings: dict[str, str] = {}
        for mapping_name in ("normalized", "redirects"):
            mappings = query.get(mapping_name, [])
            if not isinstance(mappings, list):
                continue
            for mapping in mappings:
                if not isinstance(mapping, dict):
                    continue
                source = mapping.get("from")
                target = mapping.get("to")
                if isinstance(source, str) and isinstance(target, str):
                    title_mappings[source] = target

        summaries_by_page_title: dict[str, str] = {}
        for page in pages:
            if not isinstance(page, dict) or "missing" in page:
                continue
            page_title = page.get("title")
            extract = page.get("extract")
            if not isinstance(page_title, str) or not isinstance(extract, str):
                continue
            summary = extract.strip()
            if summary:
                summaries_by_page_title[page_title] = summary

        summaries: dict[str, str] = {}
        for requested_title in requested_titles:
            resolved_title = WikipediaSummaryClient._resolve_title(
                requested_title,
                title_mappings,
            )
            summary = summaries_by_page_title.get(resolved_title)
            if summary is not None:
                summaries[requested_title] = summary

        return summaries

    @staticmethod
    def _resolve_title(title: str, mappings: dict[str, str]) -> str:
        resolved_title = title
        seen_titles: set[str] = set()

        while resolved_title in mappings and resolved_title not in seen_titles:
            seen_titles.add(resolved_title)
            resolved_title = mappings[resolved_title]

        return resolved_title
