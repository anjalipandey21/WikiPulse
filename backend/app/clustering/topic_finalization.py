"""Deterministic conversion of semantic candidates into topic clusters."""

from collections.abc import Sequence

from ..models import TopicCluster
from .keyword_extraction import ArticleKeywords
from .semantic_clustering import CandidateTopicCluster


MAX_CLUSTER_KEYWORDS = 5


def finalize_topic_clusters(
    candidates: Sequence[CandidateTopicCluster],
    article_keywords: Sequence[ArticleKeywords],
) -> list[TopicCluster]:
    """Build final topic models without changing semantic membership."""
    keywords_by_title = _build_keyword_lookup(article_keywords)
    topic_clusters: list[TopicCluster] = []

    for candidate in candidates:
        if not candidate.articles:
            raise ValueError(f"candidate {candidate.id} has no articles")

        member_keywords: list[ArticleKeywords] = []
        for article in candidate.articles:
            keywords = keywords_by_title.get(article.normalized_title)
            if keywords is None:
                raise ValueError(
                    "missing keyword mapping for article "
                    f"{article.normalized_title}"
                )
            member_keywords.append(keywords)

        aggregated_keywords = _aggregate_keywords(member_keywords)
        topic_clusters.append(
            TopicCluster(
                id=candidate.id,
                name=_build_topic_name(candidate, aggregated_keywords),
                description=None,
                articles=list(candidate.articles),
                keywords=aggregated_keywords,
                total_views=sum(
                    article.weekly_views for article in candidate.articles
                ),
                article_count=len(candidate.articles),
                confidence_score=candidate.cohesion_score,
            )
        )

    return topic_clusters


def _build_keyword_lookup(
    article_keywords: Sequence[ArticleKeywords],
) -> dict[str, ArticleKeywords]:
    keywords_by_title: dict[str, ArticleKeywords] = {}
    for keyword_result in article_keywords:
        if keyword_result.normalized_title in keywords_by_title:
            raise ValueError(
                "duplicate keyword mapping for article "
                f"{keyword_result.normalized_title}"
            )
        keywords_by_title[keyword_result.normalized_title] = keyword_result
    return keywords_by_title


def _aggregate_keywords(
    member_keywords: Sequence[ArticleKeywords],
) -> list[str]:
    support_by_keyword: dict[str, int] = {}
    rank_score_by_keyword: dict[str, float] = {}

    for keyword_result in member_keywords:
        seen_in_article: set[str] = set()
        for rank, raw_keyword in enumerate(keyword_result.keywords):
            keyword = " ".join(raw_keyword.split()).casefold()
            if not keyword or keyword in seen_in_article:
                continue
            seen_in_article.add(keyword)
            support_by_keyword[keyword] = support_by_keyword.get(keyword, 0) + 1
            rank_score_by_keyword[keyword] = (
                rank_score_by_keyword.get(keyword, 0.0) + 1.0 / (rank + 1)
            )

    ranked_keywords = sorted(
        support_by_keyword,
        key=lambda keyword: (
            -support_by_keyword[keyword],
            -rank_score_by_keyword[keyword],
            -len(keyword.split()),
            keyword,
        ),
    )

    selected: list[str] = []
    for keyword in ranked_keywords:
        if _is_redundant(keyword, selected):
            continue
        selected.append(keyword)
        if len(selected) == MAX_CLUSTER_KEYWORDS:
            break

    return selected


def _build_topic_name(
    candidate: CandidateTopicCluster,
    aggregated_keywords: Sequence[str],
) -> str:
    if aggregated_keywords:
        return aggregated_keywords[0].title()

    fallback_article = min(
        candidate.articles,
        key=lambda article: (
            -article.weekly_views,
            article.normalized_title.casefold(),
            article.normalized_title,
        ),
    )
    return fallback_article.normalized_title


def _is_redundant(candidate: str, selected: Sequence[str]) -> bool:
    candidate_tokens = candidate.split()
    for selected_keyword in selected:
        selected_tokens = selected_keyword.split()
        if len(candidate_tokens) == 1 and len(selected_tokens) == 2:
            if candidate_tokens[0] in selected_tokens:
                return True
        elif len(candidate_tokens) == 2 and len(selected_tokens) == 1:
            if selected_tokens[0] in candidate_tokens:
                return True
    return False
