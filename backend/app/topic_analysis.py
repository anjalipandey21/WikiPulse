"""Deterministic orchestration for the WikiPulse topic-analysis pipeline."""

import asyncio
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
import logging
from typing import Protocol

from .clustering.keyword_extraction import DEFAULT_TOP_K, extract_article_keywords
from .clustering.semantic_clustering import (
    DEFAULT_SIMILARITY_THRESHOLD,
    MIN_CLUSTER_SIZE,
    ArticleEncoder,
    group_candidate_topics,
)
from .clustering.topic_finalization import finalize_topic_clusters
from .filtering.article_noise import get_noise_reason
from .models import Article, TopicCluster
from .progress import AnalysisProgressReporter, report_progress
from .services.wikipedia_summary_client import WikipediaSummaryError


DEFAULT_TOP_N = 100
MAX_TOP_N = 100

logger = logging.getLogger(__name__)


class PageviewClient(Protocol):
    """Injected source for already normalized seven-day Pageviews articles."""

    async def fetch_latest_articles(
        self,
        *,
        today_utc: date | None = None,
    ) -> list[Article]:
        """Return articles in descending aggregated Pageviews order."""
        ...


class SummaryClient(Protocol):
    """Injected best-effort Wikipedia summary enricher."""

    async def enrich_articles(self, articles: Sequence[Article]) -> list[Article]:
        """Return corresponding enriched articles in input order."""
        ...


class TopicAnalysisInvariantError(RuntimeError):
    """Raised when a pipeline component changes the selected article partition."""


@dataclass(frozen=True, slots=True)
class RejectedArticle:
    """An article excluded by the deterministic noise filter and its reason."""

    article: Article
    reason: str


@dataclass(frozen=True, slots=True)
class TopicAnalysisMetrics:
    """Deterministic volume and coverage metrics for one analysis run.

    Counts cover fetched, rejected, eligible, top-N-omitted, selected, summary
    availability, final topic membership, and unclustered membership. The
    selected Pageviews total is the sum of the selected seven-day aggregates.
    """

    fetched_article_count: int
    rejected_article_count: int
    eligible_article_count: int
    top_n_omitted_article_count: int
    selected_article_count: int
    summary_available_article_count: int
    summary_missing_article_count: int
    topic_cluster_count: int
    clustered_article_count: int
    unclustered_article_count: int
    selected_pageviews: int


@dataclass(frozen=True, slots=True)
class TopicAnalysisResult:
    """Final topics plus traceable non-topic outcomes and run metrics."""

    topics: tuple[TopicCluster, ...]
    unclustered_articles: tuple[Article, ...]
    rejected_articles: tuple[RejectedArticle, ...]
    metrics: TopicAnalysisMetrics


async def analyze_topics(
    pageview_client: PageviewClient,
    summary_client: SummaryClient,
    encoder: ArticleEncoder,
    *,
    today_utc: date | None = None,
    top_n: int = DEFAULT_TOP_N,
    keyword_top_k: int = DEFAULT_TOP_K,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    progress_reporter: AnalysisProgressReporter | None = None,
) -> TopicAnalysisResult:
    """Run the deterministic topic pipeline using caller-owned dependencies."""
    _validate_top_n(top_n)

    await report_progress(progress_reporter, "fetching_pageviews")
    fetched_articles = await pageview_client.fetch_latest_articles(
        today_utc=today_utc
    )
    await report_progress(progress_reporter, "selecting_articles")
    eligible_articles, rejected_articles = _partition_noise(fetched_articles)
    selected_articles = eligible_articles[:top_n]
    if selected_articles:
        await report_progress(progress_reporter, "enriching_summaries")
    enriched_articles = await _enrich_best_effort(
        summary_client,
        selected_articles,
    )

    await report_progress(progress_reporter, "modeling_topics")
    topics, unclustered_articles = await _analyze_in_worker(
        enriched_articles,
        encoder=encoder,
        keyword_top_k=keyword_top_k,
        similarity_threshold=similarity_threshold,
        min_cluster_size=min_cluster_size,
    )
    _validate_output_partition(
        enriched_articles,
        topics,
        unclustered_articles,
    )

    summary_available_count = sum(
        1
        for article in enriched_articles
        if article.summary is not None and article.summary.strip()
    )
    clustered_article_count = sum(len(topic.articles) for topic in topics)
    metrics = TopicAnalysisMetrics(
        fetched_article_count=len(fetched_articles),
        rejected_article_count=len(rejected_articles),
        eligible_article_count=len(eligible_articles),
        top_n_omitted_article_count=(
            len(eligible_articles) - len(selected_articles)
        ),
        selected_article_count=len(selected_articles),
        summary_available_article_count=summary_available_count,
        summary_missing_article_count=(
            len(enriched_articles) - summary_available_count
        ),
        topic_cluster_count=len(topics),
        clustered_article_count=clustered_article_count,
        unclustered_article_count=len(unclustered_articles),
        selected_pageviews=sum(
            article.weekly_views for article in selected_articles
        ),
    )
    return TopicAnalysisResult(
        topics=topics,
        unclustered_articles=unclustered_articles,
        rejected_articles=tuple(rejected_articles),
        metrics=metrics,
    )


async def _analyze_in_worker(
    enriched_articles: Sequence[Article],
    *,
    encoder: ArticleEncoder,
    keyword_top_k: int,
    similarity_threshold: float,
    min_cluster_size: int,
) -> tuple[tuple[TopicCluster, ...], tuple[Article, ...]]:
    """Keep shared encoder work alive until its thread finishes on cancellation."""
    worker = asyncio.create_task(
        asyncio.to_thread(
            _analyze_enriched_articles,
            enriched_articles,
            encoder=encoder,
            keyword_top_k=keyword_top_k,
            similarity_threshold=similarity_threshold,
            min_cluster_size=min_cluster_size,
        )
    )
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError as cancelled:
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
        if not worker.cancelled():
            try:
                worker.result()
            except Exception:
                pass
        raise cancelled


def _validate_top_n(top_n: int) -> None:
    if (
        isinstance(top_n, bool)
        or not isinstance(top_n, int)
        or not 1 <= top_n <= MAX_TOP_N
    ):
        raise ValueError(f"top_n must be an integer between 1 and {MAX_TOP_N}")


def _partition_noise(
    articles: Sequence[Article],
) -> tuple[list[Article], list[RejectedArticle]]:
    eligible_articles: list[Article] = []
    rejected_articles: list[RejectedArticle] = []

    for article in articles:
        reason = get_noise_reason(article.normalized_title)
        if reason is None:
            eligible_articles.append(article)
        else:
            rejected_articles.append(RejectedArticle(article, reason))

    return eligible_articles, rejected_articles


async def _enrich_best_effort(
    summary_client: SummaryClient,
    selected_articles: Sequence[Article],
) -> list[Article]:
    if not selected_articles:
        return []

    try:
        enriched_articles = await summary_client.enrich_articles(selected_articles)
    except WikipediaSummaryError as exc:
        logger.warning(
            "Wikipedia summary enrichment failed; continuing without new summaries: %s",
            exc,
        )
        return [article.model_copy(deep=True) for article in selected_articles]

    _validate_enrichment_output(selected_articles, enriched_articles)
    return enriched_articles


def _validate_enrichment_output(
    selected_articles: Sequence[Article],
    enriched_articles: Sequence[Article],
) -> None:
    if len(enriched_articles) != len(selected_articles):
        raise TopicAnalysisInvariantError(
            "summary enrichment did not return one article per selected article"
        )

    for selected, enriched in zip(
        selected_articles,
        enriched_articles,
        strict=True,
    ):
        if (
            not isinstance(enriched, Article)
            or enriched.normalized_title != selected.normalized_title
        ):
            raise TopicAnalysisInvariantError(
                "summary enrichment changed selected article ordering or identity"
            )


def _analyze_enriched_articles(
    enriched_articles: Sequence[Article],
    *,
    encoder: ArticleEncoder,
    keyword_top_k: int,
    similarity_threshold: float,
    min_cluster_size: int,
) -> tuple[tuple[TopicCluster, ...], tuple[Article, ...]]:
    article_keywords = extract_article_keywords(
        enriched_articles,
        top_k=keyword_top_k,
    )
    semantic_result = group_candidate_topics(
        enriched_articles,
        encoder=encoder,
        similarity_threshold=similarity_threshold,
        min_cluster_size=min_cluster_size,
    )
    topics = finalize_topic_clusters(
        semantic_result.clusters,
        article_keywords,
    )
    return tuple(topics), semantic_result.unclustered_articles


def _validate_output_partition(
    selected_articles: Sequence[Article],
    topics: Sequence[TopicCluster],
    unclustered_articles: Sequence[Article],
) -> None:
    selected_id_counts = Counter(id(article) for article in selected_articles)
    output_id_counts = Counter(
        id(article)
        for topic in topics
        for article in topic.articles
    )
    output_id_counts.update(id(article) for article in unclustered_articles)

    if output_id_counts != selected_id_counts:
        raise TopicAnalysisInvariantError(
            "clustered and unclustered articles must partition selected articles "
            "exactly once"
        )
