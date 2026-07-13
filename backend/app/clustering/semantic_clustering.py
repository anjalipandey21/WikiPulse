"""Local semantic embeddings and deterministic candidate topic grouping."""

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
import math
from typing import Protocol

from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.utils.validation import check_array

from ..models import Article


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SIMILARITY_THRESHOLD = 0.55
MIN_CLUSTER_SIZE = 2
EMBEDDING_BATCH_SIZE = 32


class ArticleEncoder(Protocol):
    """Interface for interchangeable local article encoders."""

    def encode(self, texts: Sequence[str]) -> object:
        """Return one embedding row per input text."""
        ...


class MiniLMArticleEncoder:
    """CPU wrapper around the default local Sentence Transformer model."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self._model = SentenceTransformer(model_name, device="cpu")

    def encode(self, texts: Sequence[str]) -> object:
        """Encode and normalize article text in bounded local batches."""
        return self._model.encode(
            list(texts),
            batch_size=EMBEDDING_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )


@dataclass(frozen=True, slots=True)
class CandidateTopicCluster:
    """A stable candidate group of semantically related articles."""

    id: str
    articles: tuple[Article, ...]


@dataclass(frozen=True, slots=True)
class SemanticClusteringResult:
    """Candidate clusters and articles that did not meet the size threshold."""

    clusters: tuple[CandidateTopicCluster, ...]
    unclustered_articles: tuple[Article, ...]


def group_candidate_topics(
    articles: Sequence[Article],
    *,
    encoder: ArticleEncoder | None = None,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
) -> SemanticClusteringResult:
    """Embed articles and form cosine-distance candidate topic groups."""
    _validate_options(similarity_threshold, min_cluster_size)

    if len(articles) < min_cluster_size:
        return SemanticClusteringResult((), tuple(articles))

    article_texts = [_build_article_text(article) for article in articles]
    active_encoder = encoder or MiniLMArticleEncoder()
    embeddings = _validate_embeddings(
        active_encoder.encode(article_texts),
        expected_count=len(articles),
    )

    labels = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=1.0 - similarity_threshold,
        compute_full_tree=True,
    ).fit_predict(embeddings)

    indices_by_label: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        indices_by_label.setdefault(int(label), []).append(index)

    grouped_indices = sorted(
        indices_by_label.values(),
        key=lambda indices: indices[0],
    )
    candidate_clusters: list[CandidateTopicCluster] = []
    unclustered_indices: list[int] = []

    for indices in grouped_indices:
        if len(indices) < min_cluster_size:
            unclustered_indices.extend(indices)
            continue

        cluster_articles = tuple(articles[index] for index in indices)
        candidate_clusters.append(
            CandidateTopicCluster(
                id=_build_candidate_id(cluster_articles),
                articles=cluster_articles,
            )
        )

    unclustered_articles = tuple(
        articles[index] for index in sorted(unclustered_indices)
    )
    return SemanticClusteringResult(
        clusters=tuple(candidate_clusters),
        unclustered_articles=unclustered_articles,
    )


def _build_article_text(article: Article) -> str:
    if article.summary and article.summary.strip():
        return f"{article.normalized_title}\n{article.summary.strip()}"
    return article.normalized_title


def _validate_embeddings(embeddings: object, *, expected_count: int) -> object:
    try:
        validated = check_array(
            embeddings,
            accept_sparse=False,
            dtype=float,
            ensure_2d=True,
            ensure_all_finite=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "encoder must return a finite two-dimensional embedding matrix"
        ) from exc

    if validated.shape[0] != expected_count:
        raise ValueError(
            "encoder returned "
            f"{validated.shape[0]} embeddings for {expected_count} articles"
        )
    if validated.shape[1] == 0:
        raise ValueError("encoder returned embeddings with no dimensions")

    return validated


def _validate_options(
    similarity_threshold: float,
    min_cluster_size: int,
) -> None:
    if (
        isinstance(similarity_threshold, bool)
        or not isinstance(similarity_threshold, (int, float))
        or not math.isfinite(similarity_threshold)
        or not 0 <= similarity_threshold < 1
    ):
        raise ValueError("similarity_threshold must be between 0 and 1")
    if (
        isinstance(min_cluster_size, bool)
        or not isinstance(min_cluster_size, int)
        or min_cluster_size < MIN_CLUSTER_SIZE
    ):
        raise ValueError("min_cluster_size must be an integer of at least 2")


def _build_candidate_id(articles: Sequence[Article]) -> str:
    stable_titles = "\0".join(
        sorted(article.normalized_title for article in articles)
    )
    digest = sha256(stable_titles.encode("utf-8")).hexdigest()[:12]
    return f"candidate-{digest}"
