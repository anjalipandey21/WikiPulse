"""Deterministic preparation and validation of audience decisions."""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from pydantic import ValidationError

from ..models import Article, AudienceSegment, TopicCluster
from ..models.audience_generation import (
    AudienceDecision,
    AudienceGenerationResponse,
    CompactArticleContext,
    CompactClusterContext,
    CreateAudienceDecision,
    SkipClusterDecision,
)


MAX_EVIDENCE_ARTICLES = 5
MAX_CONTEXT_SUMMARY_CHARACTERS = 1_000

INVALID_TOTAL_ANALYZED_VIEWS = "invalid_total_analyzed_views"
DUPLICATE_SOURCE_CLUSTER_ID = "duplicate_source_cluster_id"
SOURCE_CLUSTER_TOO_SMALL = "source_cluster_too_small"
SOURCE_ARTICLE_COUNT_MISMATCH = "source_article_count_mismatch"
SOURCE_PAGEVIEWS_MISMATCH = "source_pageviews_mismatch"
DUPLICATE_SOURCE_ARTICLE = "duplicate_source_article"
CLUSTER_VIEWS_EXCEED_TOTAL = "cluster_views_exceed_total"
INVALID_SOURCE_CLUSTER_CONTEXT = "invalid_source_cluster_context"

UNKNOWN_DECISION_CLUSTER = "unknown_decision_cluster"
MISSING_CLUSTER_DECISION = "missing_cluster_decision"
DUPLICATE_CLUSTER_DECISION = "duplicate_cluster_decision"
DUPLICATE_SUPPORTING_REFERENCE = "duplicate_supporting_reference"
TOO_FEW_SUPPORTING_REFERENCES = "too_few_supporting_references"
TOO_MANY_SUPPORTING_REFERENCES = "too_many_supporting_references"
CROSS_CLUSTER_SUPPORTING_REFERENCE = "cross_cluster_supporting_reference"
UNKNOWN_SUPPORTING_REFERENCE = "unknown_supporting_reference"


class AudienceSourceIntegrityError(ValueError):
    """Fatal source-data failure that cannot be repaired by the provider."""

    def __init__(self, code: str, cluster_id: str | None = None) -> None:
        self.code = code
        self.cluster_id = cluster_id
        location = f" for cluster {cluster_id!r}" if cluster_id else ""
        super().__init__(f"{code}{location}")


@dataclass(frozen=True, slots=True)
class PreparedAudienceCluster:
    """Provider context and immutable references for one source cluster."""

    cluster: TopicCluster
    context: CompactClusterContext
    cluster_id: str
    cluster_pageviews: int
    evidence_reference_ids: tuple[str, ...]
    resolution_map: Mapping[str, Article]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evidence_reference_ids",
            tuple(self.evidence_reference_ids),
        )
        object.__setattr__(
            self,
            "resolution_map",
            MappingProxyType(dict(self.resolution_map)),
        )


@dataclass(frozen=True, slots=True)
class AudiencePreparation:
    """Prepared clusters plus immutable global reference ownership."""

    clusters: tuple[PreparedAudienceCluster, ...]
    total_analyzed_views: int
    reference_cluster_ids: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "clusters", tuple(self.clusters))
        object.__setattr__(
            self,
            "reference_cluster_ids",
            MappingProxyType(dict(self.reference_cluster_ids)),
        )

    @property
    def contexts(self) -> tuple[CompactClusterContext, ...]:
        """Return provider contexts in commercial-routing order."""
        return tuple(prepared.context for prepared in self.clusters)


@dataclass(frozen=True, slots=True)
class ProviderSkippedCluster:
    """A valid provider decision not to create an audience."""

    cluster: TopicCluster
    reason: str


@dataclass(frozen=True, slots=True)
class AudienceDecisionIssue:
    """One stable, machine-readable problem with a provider decision."""

    code: str
    reference_id: str | None = None


@dataclass(frozen=True, slots=True)
class InvalidAudienceDecision:
    """An unresolved per-cluster result for a later bounded revision."""

    cluster_id: str
    source_cluster: TopicCluster | None
    decisions: tuple[AudienceDecision, ...]
    issues: tuple[AudienceDecisionIssue, ...]


@dataclass(frozen=True, slots=True)
class AudienceValidationMetrics:
    """Deterministic audience-decision validation and coverage metrics.

    Valid decisions are accepted creates plus provider skips. Invalid decisions
    are report records, so they include synthetic missing-decision records and
    unknown cluster IDs and need not sum with valid decisions to the number
    received. Unresolved source clusters exclude unknown IDs. Supporting and
    represented Pageviews metrics cover accepted created segments only.
    """

    eligible_cluster_count: int
    received_decision_count: int
    valid_decision_count: int
    invalid_decision_count: int
    created_segment_count: int
    provider_skip_count: int
    unresolved_source_cluster_count: int
    supporting_reference_count: int
    unique_supporting_article_count: int
    represented_cluster_pageviews: int
    total_analyzed_pageviews: int
    issue_count: int
    issue_counts_by_code: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "issue_counts_by_code",
            MappingProxyType(dict(self.issue_counts_by_code)),
        )


@dataclass(frozen=True, slots=True)
class AudienceValidationReport:
    """Retained valid outcomes and unresolved per-cluster decisions."""

    valid_segments: tuple[AudienceSegment, ...]
    provider_skips: tuple[ProviderSkippedCluster, ...]
    invalid_decisions: tuple[InvalidAudienceDecision, ...]
    metrics: AudienceValidationMetrics
    is_publishable: bool


def prepare_audience_clusters(
    eligible_clusters: Sequence[TopicCluster],
    *,
    total_analyzed_views: int,
) -> AudiencePreparation:
    """Select evidence and create stable references for eligible clusters."""
    _validate_total_analyzed_views(
        total_analyzed_views,
        allow_zero=not eligible_clusters,
    )

    seen_cluster_ids: set[str] = set()
    seen_article_ids: set[int] = set()
    seen_normalized_titles: set[str] = set()
    reference_cluster_ids: dict[str, str] = {}
    prepared_clusters: list[PreparedAudienceCluster] = []
    cumulative_cluster_views = 0

    for cluster in eligible_clusters:
        if not isinstance(cluster, TopicCluster):
            raise AudienceSourceIntegrityError(INVALID_SOURCE_CLUSTER_CONTEXT)
        if cluster.id in seen_cluster_ids:
            raise AudienceSourceIntegrityError(
                DUPLICATE_SOURCE_CLUSTER_ID,
                cluster.id,
            )
        seen_cluster_ids.add(cluster.id)

        if len(cluster.articles) < 2:
            raise AudienceSourceIntegrityError(
                SOURCE_CLUSTER_TOO_SMALL,
                cluster.id,
            )
        if cluster.article_count != len(cluster.articles):
            raise AudienceSourceIntegrityError(
                SOURCE_ARTICLE_COUNT_MISMATCH,
                cluster.id,
            )

        for article in cluster.articles:
            if not isinstance(article, Article):
                raise AudienceSourceIntegrityError(
                    INVALID_SOURCE_CLUSTER_CONTEXT,
                    cluster.id,
                )
            if (
                id(article) in seen_article_ids
                or article.normalized_title in seen_normalized_titles
            ):
                raise AudienceSourceIntegrityError(
                    DUPLICATE_SOURCE_ARTICLE,
                    cluster.id,
                )
            seen_article_ids.add(id(article))
            seen_normalized_titles.add(article.normalized_title)

        calculated_views = sum(
            article.weekly_views for article in cluster.articles
        )
        if cluster.total_views != calculated_views:
            raise AudienceSourceIntegrityError(
                SOURCE_PAGEVIEWS_MISMATCH,
                cluster.id,
            )
        cumulative_cluster_views += cluster.total_views
        if (
            cluster.total_views > total_analyzed_views
            or cumulative_cluster_views > total_analyzed_views
        ):
            raise AudienceSourceIntegrityError(
                CLUSTER_VIEWS_EXCEED_TOTAL,
                cluster.id,
            )

        evidence_articles = sorted(
            cluster.articles,
            key=lambda article: (
                -article.weekly_views,
                article.normalized_title,
            ),
        )[:MAX_EVIDENCE_ARTICLES]
        resolution_map = {
            f"{cluster.id}:a{index}": article
            for index, article in enumerate(evidence_articles)
        }
        reference_cluster_ids.update(
            (reference_id, cluster.id) for reference_id in resolution_map
        )

        try:
            context = CompactClusterContext(
                cluster_id=cluster.id,
                name=cluster.name,
                keywords=cluster.keywords[:5],
                total_views=cluster.total_views,
                article_count=cluster.article_count,
                topic_confidence=cluster.confidence_score,
                articles=[
                    CompactArticleContext(
                        reference_id=reference_id,
                        title=article.title,
                        weekly_views=article.weekly_views,
                        summary=_compact_summary(article.summary),
                    )
                    for reference_id, article in resolution_map.items()
                ],
            )
        except ValidationError as exc:
            raise AudienceSourceIntegrityError(
                INVALID_SOURCE_CLUSTER_CONTEXT,
                cluster.id,
            ) from exc

        prepared_clusters.append(
            PreparedAudienceCluster(
                cluster=cluster,
                context=context,
                cluster_id=cluster.id,
                cluster_pageviews=cluster.total_views,
                evidence_reference_ids=tuple(resolution_map),
                resolution_map=resolution_map,
            )
        )

    return AudiencePreparation(
        clusters=tuple(prepared_clusters),
        total_analyzed_views=total_analyzed_views,
        reference_cluster_ids=reference_cluster_ids,
    )


def finalize_audience_decisions(
    preparation: AudiencePreparation,
    response: AudienceGenerationResponse,
) -> AudienceValidationReport:
    """Validate each provider decision without discarding valid outcomes."""
    prepared_by_id = {
        prepared.cluster_id: prepared for prepared in preparation.clusters
    }
    decisions_by_cluster: dict[str, list[AudienceDecision]] = {
        cluster_id: [] for cluster_id in prepared_by_id
    }
    unknown_decisions: list[InvalidAudienceDecision] = []

    for decision in response.decisions:
        if decision.cluster_id not in prepared_by_id:
            unknown_decisions.append(
                InvalidAudienceDecision(
                    cluster_id=decision.cluster_id,
                    source_cluster=None,
                    decisions=(decision,),
                    issues=(AudienceDecisionIssue(UNKNOWN_DECISION_CLUSTER),),
                )
            )
            continue
        decisions_by_cluster[decision.cluster_id].append(decision)

    valid_segments: list[AudienceSegment] = []
    provider_skips: list[ProviderSkippedCluster] = []
    invalid_decisions: list[InvalidAudienceDecision] = []
    unresolved_source_cluster_count = 0

    for prepared in preparation.clusters:
        cluster = prepared.cluster
        cluster_decisions = decisions_by_cluster[prepared.cluster_id]
        if not cluster_decisions:
            unresolved_source_cluster_count += 1
            invalid_decisions.append(
                InvalidAudienceDecision(
                    cluster_id=prepared.cluster_id,
                    source_cluster=cluster,
                    decisions=(),
                    issues=(AudienceDecisionIssue(MISSING_CLUSTER_DECISION),),
                )
            )
            continue
        if len(cluster_decisions) > 1:
            unresolved_source_cluster_count += 1
            invalid_decisions.append(
                InvalidAudienceDecision(
                    cluster_id=prepared.cluster_id,
                    source_cluster=cluster,
                    decisions=tuple(cluster_decisions),
                    issues=(AudienceDecisionIssue(DUPLICATE_CLUSTER_DECISION),),
                )
            )
            continue

        decision = cluster_decisions[0]
        if isinstance(decision, SkipClusterDecision):
            provider_skips.append(
                ProviderSkippedCluster(cluster=cluster, reason=decision.reason)
            )
            continue

        issues = _get_create_decision_issues(
            prepared,
            decision,
            preparation.reference_cluster_ids,
        )
        if issues:
            unresolved_source_cluster_count += 1
            invalid_decisions.append(
                InvalidAudienceDecision(
                    cluster_id=prepared.cluster_id,
                    source_cluster=cluster,
                    decisions=(decision,),
                    issues=issues,
                )
            )
            continue

        valid_segments.append(
            _build_audience_segment(
                prepared,
                decision,
                preparation.total_analyzed_views,
            )
        )

    invalid_decisions.extend(unknown_decisions)
    issue_counts = Counter(
        issue.code
        for invalid in invalid_decisions
        for issue in invalid.issues
    )
    supporting_articles = [
        article
        for segment in valid_segments
        for article in segment.supporting_articles
    ]
    metrics = AudienceValidationMetrics(
        eligible_cluster_count=len(preparation.clusters),
        received_decision_count=len(response.decisions),
        valid_decision_count=len(valid_segments) + len(provider_skips),
        invalid_decision_count=len(invalid_decisions),
        created_segment_count=len(valid_segments),
        provider_skip_count=len(provider_skips),
        unresolved_source_cluster_count=unresolved_source_cluster_count,
        supporting_reference_count=len(supporting_articles),
        unique_supporting_article_count=len(
            {id(article) for article in supporting_articles}
        ),
        represented_cluster_pageviews=sum(
            prepared_by_id[
                segment.topic_cluster_ids[0]
            ].cluster_pageviews
            for segment in valid_segments
        ),
        total_analyzed_pageviews=preparation.total_analyzed_views,
        issue_count=sum(issue_counts.values()),
        issue_counts_by_code={
            code: issue_counts[code] for code in sorted(issue_counts)
        },
    )
    return AudienceValidationReport(
        valid_segments=tuple(valid_segments),
        provider_skips=tuple(provider_skips),
        invalid_decisions=tuple(invalid_decisions),
        metrics=metrics,
        is_publishable=not invalid_decisions,
    )


def _validate_total_analyzed_views(
    total_analyzed_views: int,
    *,
    allow_zero: bool,
) -> None:
    if (
        isinstance(total_analyzed_views, bool)
        or not isinstance(total_analyzed_views, int)
        or total_analyzed_views < 0
        or (total_analyzed_views == 0 and not allow_zero)
    ):
        raise AudienceSourceIntegrityError(INVALID_TOTAL_ANALYZED_VIEWS)


def _compact_summary(summary: str | None) -> str | None:
    if summary is None:
        return None
    normalized_summary = " ".join(summary.split())
    if not normalized_summary:
        return None
    return normalized_summary[:MAX_CONTEXT_SUMMARY_CHARACTERS].rstrip()


def _get_create_decision_issues(
    prepared: PreparedAudienceCluster,
    decision: CreateAudienceDecision,
    reference_cluster_ids: Mapping[str, str],
) -> tuple[AudienceDecisionIssue, ...]:
    references = decision.supporting_article_reference_ids
    issues: list[AudienceDecisionIssue] = []

    seen: set[str] = set()
    duplicate_references: list[str] = []
    for reference_id in references:
        if reference_id in seen and reference_id not in duplicate_references:
            duplicate_references.append(reference_id)
        seen.add(reference_id)
    issues.extend(
        AudienceDecisionIssue(DUPLICATE_SUPPORTING_REFERENCE, reference_id)
        for reference_id in duplicate_references
    )

    if len(references) < 2:
        issues.append(AudienceDecisionIssue(TOO_FEW_SUPPORTING_REFERENCES))
    if len(references) > MAX_EVIDENCE_ARTICLES:
        issues.append(AudienceDecisionIssue(TOO_MANY_SUPPORTING_REFERENCES))

    cross_cluster_references: list[str] = []
    unknown_references: list[str] = []
    for reference_id in dict.fromkeys(references):
        owner_cluster_id = reference_cluster_ids.get(reference_id)
        if owner_cluster_id is None:
            unknown_references.append(reference_id)
        elif owner_cluster_id != prepared.cluster_id:
            cross_cluster_references.append(reference_id)

    issues.extend(
        AudienceDecisionIssue(CROSS_CLUSTER_SUPPORTING_REFERENCE, reference_id)
        for reference_id in cross_cluster_references
    )
    issues.extend(
        AudienceDecisionIssue(UNKNOWN_SUPPORTING_REFERENCE, reference_id)
        for reference_id in unknown_references
    )
    return tuple(issues)


def _build_audience_segment(
    prepared: PreparedAudienceCluster,
    decision: CreateAudienceDecision,
    total_analyzed_views: int,
) -> AudienceSegment:
    selected_references = set(decision.supporting_article_reference_ids)
    supporting_articles = [
        prepared.resolution_map[reference_id]
        for reference_id in prepared.evidence_reference_ids
        if reference_id in selected_references
    ]
    return AudienceSegment(
        id=f"audience-{prepared.cluster_id}",
        name=decision.name,
        description=decision.description,
        topic_cluster_ids=[prepared.cluster_id],
        size_index=(
            prepared.cluster_pageviews / total_analyzed_views * 100
        ),
        buying_power=decision.buying_power,
        buying_power_reason=decision.buying_power_reason,
        brand_categories=list(decision.brand_categories),
        supporting_articles=supporting_articles,
        commercial_confidence=decision.commercial_confidence,
        commercial_confidence_reason=decision.commercial_confidence_reason,
    )
