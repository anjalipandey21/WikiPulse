"""Deterministic Wikipedia article title normalization and aggregation."""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
import unicodedata
from urllib.parse import quote, unquote

from ..models import Article


WIKIPEDIA_ARTICLE_BASE_URL = "https://en.wikipedia.org/wiki/"


@dataclass(frozen=True, slots=True)
class ArticleViewObservation:
    """An observed article pageview count for one date."""

    observed_date: date
    title: str
    views: int


def normalize_title(title: str) -> str:
    """Normalize a Wikimedia article title without changing its case."""
    decoded_title = unquote(title)
    spaced_title = decoded_title.replace("_", " ")
    unicode_normalized_title = unicodedata.normalize("NFC", spaced_title)
    normalized_title = " ".join(unicode_normalized_title.split())
    if not normalized_title:
        raise ValueError("article title must not be empty after normalization")
    return normalized_title


def build_article_url(normalized_title: str) -> str:
    """Build an English Wikipedia URL from a normalized article title."""
    url_title = normalized_title.replace(" ", "_")
    return f"{WIKIPEDIA_ARTICLE_BASE_URL}{quote(url_title, safe='')}"


def aggregate_articles(
    observations: Iterable[ArticleViewObservation],
    analysis_start_date: date,
    analysis_end_date: date,
) -> list[Article]:
    """Aggregate dated observations into deterministically sorted articles."""
    if analysis_start_date > analysis_end_date:
        raise ValueError(
            "analysis_start_date must be on or before analysis_end_date"
        )

    views_by_article: dict[str, dict[date, int]] = {}

    for observation in observations:
        if observation.views < 0:
            raise ValueError("article view observations must not be negative")
        if not analysis_start_date <= observation.observed_date <= analysis_end_date:
            raise ValueError("article view observation date is outside the analysis range")

        normalized_title = normalize_title(observation.title)
        daily_views = views_by_article.setdefault(normalized_title, {})
        daily_views[observation.observed_date] = (
            daily_views.get(observation.observed_date, 0) + observation.views
        )

    articles = [
        Article(
            title=normalized_title,
            normalized_title=normalized_title,
            url=build_article_url(normalized_title),
            weekly_views=sum(daily_views.values()),
            daily_views=dict(sorted(daily_views.items())),
            summary=None,
            analysis_start_date=analysis_start_date,
            analysis_end_date=analysis_end_date,
        )
        for normalized_title, daily_views in views_by_article.items()
    ]

    return sorted(
        articles,
        key=lambda article: (-article.weekly_views, article.normalized_title),
    )
