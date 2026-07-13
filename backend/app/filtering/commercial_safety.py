"""Deterministic commercial-safety routing for finalized topic clusters."""

from collections.abc import Sequence
from dataclasses import dataclass
import re

from ..models import Article, TopicCluster


MIN_TOPIC_COHESION = 0.60
MIN_SUMMARY_COVERAGE = 0.50
MAX_ELIGIBLE_CLUSTERS = 6

LOW_COHESION_REASON = "low_topic_cohesion"
NO_PAGEVIEWS_REASON = "no_pageviews"
INSUFFICIENT_SUMMARY_COVERAGE_REASON = "insufficient_summary_coverage"
SENSITIVE_EVENT_REASON = "sensitive_event"
TOP_SIX_LIMIT_REASON = "top_six_limit"

_CREATIVE_WORK_SUFFIX = re.compile(
    r"\((?:film|television series|tv series|song|album|novel|play|"
    r"video game|band|comics?|musical)\)$",
    re.IGNORECASE,
)
_CREATIVE_WORK_SUMMARY = re.compile(
    r"^[^.]{0,160}\b(?:is|was)\s+(?:an?|the)\s+"
    r"(?:\d{4}\s+)?(?:[\w-]+\s+){0,6}"
    r"(?:film|television series|tv series|song|album|novel|play|"
    r"video game|band|comic|musical)\b",
    re.IGNORECASE,
)
_SENSITIVE_CLUSTER_PHRASE = re.compile(
    r"\b(?:mass shooting|terrorist attacks?|natural disaster|"
    r"mass-casualty event)\b",
    re.IGNORECASE,
)
_SENSITIVE_TITLE_PATTERNS = (
    re.compile(r"^deaths in \d{4}(?:\b|$)", re.IGNORECASE),
    re.compile(r"^list of (?:notable )?deaths\b", re.IGNORECASE),
    re.compile(
        r"^(?:murder|killing|assassination) of\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:mass|school|university|church|mosque|synagogue|"
        r"nightclub) shooting\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bshooting (?:at|in|of)\b", re.IGNORECASE),
    re.compile(r"\bbombing(?: of| at| in)?$", re.IGNORECASE),
    re.compile(r"\bterrorist attacks?\b", re.IGNORECASE),
    re.compile(r"\bmassacre (?:of|at|in)\b", re.IGNORECASE),
    re.compile(
        r"^(?:\d{4}\s+)?[\w'’.-]+(?:\s+[\w'’.-]+){0,4}\s+massacre$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:aircraft|airplane|plane|helicopter|train|bus) "
        r"(?:crash|accident)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:shipwreck|ferry disaster)\b", re.IGNORECASE),
    re.compile(r"\bdisaster$", re.IGNORECASE),
    re.compile(
        r"^\d{4}\b.*\b(?:attacks?|earthquake|wildfires?|floods?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:wildfires|floods)$", re.IGNORECASE),
    re.compile(r"^hurricane [\w'-]+(?: \(\d{4}\))?$", re.IGNORECASE),
    re.compile(r"\b(?:genocide|war crimes?)\b", re.IGNORECASE),
)
_SENSITIVE_SUMMARY_PATTERNS = (
    re.compile(
        r"\b(?:is|was|were)\s+(?:an?|the)\s+"
        r"(?:(?:deadly|fatal|major|devastating|mass-casualty)\s+){0,2}"
        r"(?:mass shooting|terrorist attack|bombing|massacre|"
        r"natural disaster|aviation disaster|aircraft crash|"
        r"train crash|genocide)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:killed|killing)\s+(?:at least\s+)?\d[\d,]*\s+"
        r"(?:people|persons|civilians|passengers)\b",
        re.IGNORECASE,
    ),
)


@dataclass(frozen=True, slots=True)
class SkippedTopicCluster:
    """A topic excluded from commercial interpretation and its stable reason."""

    cluster: TopicCluster
    reason: str


@dataclass(frozen=True, slots=True)
class CommercialSafetyResult:
    """Pageview-ranked eligible topics and reason-coded skipped topics."""

    eligible_clusters: tuple[TopicCluster, ...]
    skipped_clusters: tuple[SkippedTopicCluster, ...]


def get_commercial_skip_reason(cluster: TopicCluster) -> str | None:
    """Return the first intrinsic commercial-safety rejection reason."""
    if (
        cluster.confidence_score is None
        or cluster.confidence_score < MIN_TOPIC_COHESION
    ):
        return LOW_COHESION_REASON

    if cluster.total_views <= 0:
        return NO_PAGEVIEWS_REASON

    if _summary_coverage(cluster.articles) < MIN_SUMMARY_COVERAGE:
        return INSUFFICIENT_SUMMARY_COVERAGE_REASON

    if _contains_sensitive_event(cluster):
        return SENSITIVE_EVENT_REASON

    return None


def route_commercial_clusters(
    clusters: Sequence[TopicCluster],
) -> CommercialSafetyResult:
    """Route at most six safe clusters by pageviews with stable tie-breaking."""
    reasons_by_index: dict[int, str] = {}
    intrinsically_eligible: list[tuple[int, TopicCluster]] = []

    for index, cluster in enumerate(clusters):
        reason = get_commercial_skip_reason(cluster)
        if reason is None:
            intrinsically_eligible.append((index, cluster))
        else:
            reasons_by_index[index] = reason

    ranked_eligible = sorted(
        intrinsically_eligible,
        key=lambda indexed_cluster: (
            -indexed_cluster[1].total_views,
            indexed_cluster[1].id.casefold(),
            indexed_cluster[1].id,
            indexed_cluster[0],
        ),
    )
    selected = ranked_eligible[:MAX_ELIGIBLE_CLUSTERS]
    for index, _ in ranked_eligible[MAX_ELIGIBLE_CLUSTERS:]:
        reasons_by_index[index] = TOP_SIX_LIMIT_REASON

    skipped_clusters = tuple(
        SkippedTopicCluster(cluster, reasons_by_index[index])
        for index, cluster in enumerate(clusters)
        if index in reasons_by_index
    )
    return CommercialSafetyResult(
        eligible_clusters=tuple(cluster for _, cluster in selected),
        skipped_clusters=skipped_clusters,
    )


def _summary_coverage(articles: Sequence[Article]) -> float:
    if not articles:
        return 0.0
    summary_count = sum(
        1
        for article in articles
        if article.summary is not None and article.summary.strip()
    )
    return summary_count / len(articles)


def _contains_sensitive_event(cluster: TopicCluster) -> bool:
    if _SENSITIVE_CLUSTER_PHRASE.search(cluster.name):
        return True
    if any(_SENSITIVE_CLUSTER_PHRASE.search(keyword) for keyword in cluster.keywords):
        return True

    for article in cluster.articles:
        if _is_clearly_creative_work(article):
            continue
        if any(
            pattern.search(article.normalized_title)
            for pattern in _SENSITIVE_TITLE_PATTERNS
        ):
            return True
        if article.summary and any(
            pattern.search(article.summary)
            for pattern in _SENSITIVE_SUMMARY_PATTERNS
        ):
            return True

    return False


def _is_clearly_creative_work(article: Article) -> bool:
    if _CREATIVE_WORK_SUFFIX.search(article.normalized_title):
        return True
    return bool(
        article.summary and _CREATIVE_WORK_SUMMARY.search(article.summary.strip())
    )
