"""Deterministic TF-IDF keyword extraction for enriched articles."""

from collections.abc import Sequence
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer

from ..models import Article


TITLE_WEIGHT = 2
DEFAULT_TOP_K = 5
FIELD_BOUNDARY_TOKEN = "wikipulsefieldboundarytoken"


@dataclass(frozen=True, slots=True)
class ArticleKeywords:
    """Ranked keywords associated with one input article."""

    normalized_title: str
    keywords: tuple[str, ...]


def extract_article_keywords(
    articles: Sequence[Article],
    *,
    top_k: int = DEFAULT_TOP_K,
) -> list[ArticleKeywords]:
    """Return deterministic keywords for each article in input order."""
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if not articles:
        return []

    documents = [_build_document(article) for article in articles]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    analyzer = vectorizer.build_analyzer()

    if not any(analyzer(document) for document in documents):
        return [
            ArticleKeywords(article.normalized_title, ()) for article in articles
        ]

    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    results: list[ArticleKeywords] = []
    for index, article in enumerate(articles):
        row = tfidf_matrix.getrow(index)
        title_features = set(analyzer(article.normalized_title))
        candidates = [
            (
                feature_names[feature_index],
                float(score)
                * (
                    TITLE_WEIGHT
                    if feature_names[feature_index] in title_features
                    else 1
                ),
            )
            for feature_index, score in zip(row.indices, row.data, strict=True)
            if FIELD_BOUNDARY_TOKEN not in feature_names[feature_index].split()
        ]
        candidates.sort(
            key=lambda candidate: (
                -candidate[1],
                -len(candidate[0].split()),
                candidate[0],
            )
        )
        keywords = _select_keywords(candidates, top_k)
        results.append(ArticleKeywords(article.normalized_title, keywords))

    return results


def _build_document(article: Article) -> str:
    if article.summary and article.summary.strip():
        return (
            f"{article.normalized_title} {FIELD_BOUNDARY_TOKEN} "
            f"{article.summary.strip()}"
        )
    return article.normalized_title


def _select_keywords(
    candidates: list[tuple[str, float]],
    top_k: int,
) -> tuple[str, ...]:
    selected: list[str] = []

    for term, _ in candidates:
        if _is_redundant(term, selected):
            continue
        selected.append(term)
        if len(selected) == top_k:
            break

    return tuple(selected)


def _is_redundant(candidate: str, selected: list[str]) -> bool:
    candidate_tokens = candidate.split()
    if len(candidate_tokens) not in (1, 2):
        return False

    for selected_term in selected:
        selected_tokens = selected_term.split()
        if len(candidate_tokens) == 1 and len(selected_tokens) == 2:
            if candidate_tokens[0] in selected_tokens:
                return True
        elif len(candidate_tokens) == 2 and len(selected_tokens) == 1:
            if selected_tokens[0] in candidate_tokens:
                return True

    return False
