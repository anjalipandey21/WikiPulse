"""Focused mocked tests for deterministic topic-analysis orchestration."""

from collections.abc import Sequence
import asyncio
from datetime import date
import threading
import unittest
from unittest.mock import patch

from app.clustering.semantic_clustering import SemanticClusteringResult
from app.models import Article
from app.progress import AnalysisProgressStage
from app.services.wikimedia_client import WikimediaPageviewsError
from app.services.wikipedia_summary_client import WikipediaSummaryError
from app.topic_analysis import (
    MAX_TOP_N,
    TopicAnalysisInvariantError,
    TopicAnalysisMetrics,
    analyze_topics,
)


def make_article(
    title: str,
    views: int,
    *,
    summary: str | None = None,
) -> Article:
    observed_date = date(2026, 7, 12)
    return Article(
        title=title,
        normalized_title=title,
        url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
        weekly_views=views,
        daily_views={observed_date: views},
        summary=summary,
        analysis_start_date=date(2026, 7, 6),
        analysis_end_date=observed_date,
    )


class FakePageviewClient:
    def __init__(
        self,
        articles: Sequence[Article] = (),
        *,
        error: WikimediaPageviewsError | None = None,
    ) -> None:
        self.articles = list(articles)
        self.error = error
        self.call_count = 0
        self.today_utc: date | None = None
        self.closed = False

    async def fetch_latest_articles(
        self,
        *,
        today_utc: date | None = None,
    ) -> list[Article]:
        self.call_count += 1
        self.today_utc = today_utc
        if self.error is not None:
            raise self.error
        return list(self.articles)

    async def aclose(self) -> None:
        self.closed = True


class FakeSummaryClient:
    def __init__(
        self,
        summaries: dict[str, str | None] | None = None,
        *,
        error: WikipediaSummaryError | None = None,
    ) -> None:
        self.summaries = summaries or {}
        self.error = error
        self.call_count = 0
        self.received_articles: list[Article] | None = None
        self.enriched_articles: list[Article] = []
        self.closed = False

    async def enrich_articles(
        self,
        articles: Sequence[Article],
    ) -> list[Article]:
        self.call_count += 1
        self.received_articles = list(articles)
        if self.error is not None:
            raise self.error

        self.enriched_articles = [
            article.model_copy(
                update={
                    "summary": (
                        self.summaries[article.normalized_title]
                        if article.normalized_title in self.summaries
                        else article.summary
                    )
                },
                deep=True,
            )
            for article in articles
        ]
        return self.enriched_articles

    async def aclose(self) -> None:
        self.closed = True


class FakeEncoder:
    def __init__(self, embeddings: object) -> None:
        self.embeddings = embeddings
        self.encoded_texts: list[str] | None = None

    def encode(self, texts: Sequence[str]) -> object:
        self.encoded_texts = list(texts)
        return self.embeddings


class TopicAnalysisTests(unittest.IsolatedAsyncioTestCase):
    async def test_connects_pipeline_and_returns_traceable_outcomes(self) -> None:
        main_page = make_article("Main Page", 1_000)
        football_final = make_article("Football final", 900)
        football_championship = make_article("Football championship", 800)
        quantum_science = make_article("Quantum science", 700)
        ancient_history = make_article("Ancient history", 600)
        pageview_client = FakePageviewClient(
            [
                main_page,
                football_final,
                football_championship,
                quantum_science,
                ancient_history,
            ]
        )
        summary_client = FakeSummaryClient(
            {
                "Football final": "An association football championship match.",
                "Football championship": "An association football tournament.",
                "Quantum science": "Quantum theory studies physical systems.",
            }
        )
        encoder = FakeEncoder(
            [
                [1.0, 0.0, 0.0],
                [0.99, 0.1, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        analysis_date = date(2026, 7, 13)
        progress: list[AnalysisProgressStage] = []

        async def report(stage: AnalysisProgressStage) -> None:
            progress.append(stage)

        result = await analyze_topics(
            pageview_client,
            summary_client,
            encoder,
            today_utc=analysis_date,
            top_n=3,
            similarity_threshold=0.8,
            progress_reporter=report,
        )

        self.assertEqual(pageview_client.today_utc, analysis_date)
        self.assertEqual(
            [
                article.normalized_title
                for article in summary_client.received_articles or []
            ],
            [
                "Football final",
                "Football championship",
                "Quantum science",
            ],
        )
        self.assertEqual(
            encoder.encoded_texts,
            [
                "Football final\nAn association football championship match.",
                "Football championship\nAn association football tournament.",
                "Quantum science\nQuantum theory studies physical systems.",
            ],
        )
        self.assertEqual(len(result.topics), 1)
        self.assertIs(
            result.topics[0].articles[0],
            summary_client.enriched_articles[0],
        )
        self.assertIs(
            result.topics[0].articles[1],
            summary_client.enriched_articles[1],
        )
        self.assertEqual(
            result.unclustered_articles,
            (summary_client.enriched_articles[2],),
        )
        self.assertEqual(len(result.rejected_articles), 1)
        self.assertIs(result.rejected_articles[0].article, main_page)
        self.assertEqual(result.rejected_articles[0].reason, "main_page")
        self.assertEqual(
            result.metrics,
            TopicAnalysisMetrics(
                fetched_article_count=5,
                rejected_article_count=1,
                eligible_article_count=4,
                top_n_omitted_article_count=1,
                selected_article_count=3,
                summary_available_article_count=3,
                summary_missing_article_count=0,
                topic_cluster_count=1,
                clustered_article_count=2,
                unclustered_article_count=1,
                selected_pageviews=2_400,
            ),
        )
        self.assertFalse(pageview_client.closed)
        self.assertFalse(summary_client.closed)
        self.assertEqual(
            progress,
            [
                "fetching_pageviews",
                "selecting_articles",
                "enriching_summaries",
                "modeling_topics",
            ],
        )

    async def test_summary_failure_logs_and_continues_with_existing_data(self) -> None:
        first = make_article(
            "Existing summary topic",
            200,
            summary="An existing summary.",
        )
        second = make_article("Related topic", 100)
        pageview_client = FakePageviewClient([first, second])
        summary_client = FakeSummaryClient(
            error=WikipediaSummaryError("summary service unavailable")
        )
        encoder = FakeEncoder([[1.0, 0.0], [0.99, 0.1]])
        before = [first.model_dump(), second.model_dump()]

        with self.assertLogs("app.topic_analysis", level="WARNING") as logs:
            result = await analyze_topics(
                pageview_client,
                summary_client,
                encoder,
                similarity_threshold=0.8,
            )

        self.assertIn("continuing without new summaries", logs.output[0])
        self.assertEqual(
            encoder.encoded_texts,
            [
                "Existing summary topic\nAn existing summary.",
                "Related topic",
            ],
        )
        self.assertEqual(len(result.topics), 1)
        self.assertIsNot(result.topics[0].articles[0], first)
        self.assertIsNot(result.topics[0].articles[1], second)
        self.assertEqual([first.model_dump(), second.model_dump()], before)
        self.assertEqual(result.metrics.summary_available_article_count, 1)
        self.assertEqual(result.metrics.summary_missing_article_count, 1)

    async def test_pageview_failure_remains_fatal(self) -> None:
        failure = WikimediaPageviewsError("seven-day window unavailable")
        pageview_client = FakePageviewClient(error=failure)
        summary_client = FakeSummaryClient()
        encoder = FakeEncoder([])

        with self.assertRaises(WikimediaPageviewsError) as raised:
            await analyze_topics(pageview_client, summary_client, encoder)

        self.assertIs(raised.exception, failure)
        self.assertEqual(summary_client.call_count, 0)
        self.assertIsNone(encoder.encoded_texts)

    async def test_rejects_invalid_top_n_before_calling_dependencies(self) -> None:
        pageview_client = FakePageviewClient()
        summary_client = FakeSummaryClient()
        encoder = FakeEncoder([])

        for top_n in (0, MAX_TOP_N + 1, True, 1.5):
            with self.subTest(top_n=top_n):
                with self.assertRaisesRegex(ValueError, "top_n"):
                    await analyze_topics(
                        pageview_client,
                        summary_client,
                        encoder,
                        top_n=top_n,  # type: ignore[arg-type]
                    )

        self.assertEqual(pageview_client.call_count, 0)
        self.assertEqual(summary_client.call_count, 0)
        self.assertIsNone(encoder.encoded_texts)

    async def test_validates_selected_article_partition(self) -> None:
        articles = [make_article("First topic", 200), make_article("Second topic", 100)]
        pageview_client = FakePageviewClient(articles)
        summary_client = FakeSummaryClient()
        encoder = FakeEncoder([])

        def omit_selected_article(
            enriched_articles: Sequence[Article],
            **_: object,
        ) -> SemanticClusteringResult:
            return SemanticClusteringResult((), (enriched_articles[0],))

        with patch(
            "app.topic_analysis.group_candidate_topics",
            side_effect=omit_selected_article,
        ):
            with self.assertRaisesRegex(
                TopicAnalysisInvariantError,
                "partition selected articles exactly once",
            ):
                await analyze_topics(
                    pageview_client,
                    summary_client,
                    encoder,
                )

    async def test_empty_source_avoids_summary_and_encoder_work(self) -> None:
        pageview_client = FakePageviewClient()
        summary_client = FakeSummaryClient()
        encoder = FakeEncoder([])

        result = await analyze_topics(
            pageview_client,
            summary_client,
            encoder,
        )

        self.assertEqual(result.topics, ())
        self.assertEqual(result.unclustered_articles, ())
        self.assertEqual(result.rejected_articles, ())
        self.assertEqual(summary_client.call_count, 0)
        self.assertIsNone(encoder.encoded_texts)
        self.assertEqual(result.metrics.selected_article_count, 0)
        self.assertEqual(result.metrics.selected_pageviews, 0)

    async def test_cancellation_waits_for_topic_worker_to_release_resources(
        self,
    ) -> None:
        article = make_article("Worker topic", 100)
        pageview_client = FakePageviewClient([article])
        summary_client = FakeSummaryClient()
        encoder = FakeEncoder([])
        worker_started = threading.Event()
        release_worker = threading.Event()

        def blocking_analysis(
            enriched_articles: Sequence[Article],
            **_: object,
        ) -> tuple[tuple[object, ...], tuple[Article, ...]]:
            worker_started.set()
            release_worker.wait(timeout=5)
            return (), tuple(enriched_articles)

        with patch(
            "app.topic_analysis._analyze_enriched_articles",
            side_effect=blocking_analysis,
        ):
            task = asyncio.create_task(
                analyze_topics(pageview_client, summary_client, encoder)
            )
            started = await asyncio.to_thread(worker_started.wait, 2)
            self.assertTrue(started)
            task.cancel()
            await asyncio.sleep(0.05)
            self.assertFalse(task.done())
            release_worker.set()
            with self.assertRaises(asyncio.CancelledError):
                await task


if __name__ == "__main__":
    unittest.main()
