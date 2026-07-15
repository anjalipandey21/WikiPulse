"""Bounded, public-safe grounding contracts for Ask WikiPulse."""

from __future__ import annotations

from collections.abc import Iterable
import json
import re
from typing import Literal, Protocol
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field

from ..models.audience_review import ReviewRunResult


MAX_GROUNDED_AUDIENCES = 5
MAX_GROUNDED_EVIDENCE = 15
MAX_GROUNDED_CONTEXT_CHARS = 18_000
INSUFFICIENT_EVIDENCE_ANSWER = (
    "The current analysis does not contain enough evidence to answer that."
)


class _GroundedContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        hide_input_in_errors=True,
    )


class GroundedEvidenceItem(_GroundedContract):
    evidence_id: str = Field(min_length=1, max_length=300)
    article_title: str = Field(min_length=1, max_length=500)
    article_url: str | None = Field(default=None, max_length=2_000)
    audience_label: str = Field(min_length=1, max_length=300)
    summary: str | None = Field(default=None, max_length=600)
    weekly_views: int = Field(ge=0)


class GroundedAudienceItem(_GroundedContract):
    context_rank: int = Field(ge=1, le=MAX_GROUNDED_AUDIENCES)
    audience_label: str = Field(min_length=1, max_length=300)
    description: str = Field(min_length=1, max_length=1_000)
    publication_source: Literal["original", "analyst_edit"]
    cluster_name: str = Field(min_length=1, max_length=500)
    cluster_pageviews: int = Field(ge=0)
    article_count: int = Field(ge=0)
    size_index: float = Field(ge=0, le=100)
    topic_confidence: float = Field(ge=0, le=1)
    buying_power: Literal["high", "medium", "low"]
    buying_power_reason: str = Field(min_length=1, max_length=1_000)
    brand_categories: tuple[str, ...]
    commercial_confidence: float = Field(ge=0, le=1)
    commercial_confidence_reason: str = Field(min_length=1, max_length=1_000)
    evidence: tuple[GroundedEvidenceItem, ...]


class GroundedAssistantContext(_GroundedContract):
    audience_count: int = Field(ge=0)
    audiences: tuple[GroundedAudienceItem, ...]


class GroundedAssistantModelResponse(_GroundedContract):
    evidence_status: Literal["grounded", "insufficient_evidence"]
    answer: str = Field(min_length=1, max_length=2_000)
    citation_ids: tuple[str, ...] = Field(default=(), max_length=8)


class GroundedAssistantProvider(Protocol):
    async def answer_grounded(
        self,
        question: str,
        context: GroundedAssistantContext,
    ) -> GroundedAssistantModelResponse:
        """Answer one question using only the supplied public-safe context."""


def build_grounded_context(result: ReviewRunResult) -> GroundedAssistantContext:
    """Build a deterministic bounded context from published review outcomes."""

    published = [
        candidate
        for candidate in result.review_candidates
        if candidate.status == "published"
    ]
    published.sort(
        key=lambda candidate: (
            -(
                candidate.edited_recommendation
                or candidate.recommendation
            ).commercial_confidence,
            -candidate.cluster_pageviews,
            candidate.ordinal,
        )
    )
    audiences: list[GroundedAudienceItem] = []
    evidence_count = 0
    for context_rank, candidate in enumerate(
        published[:MAX_GROUNDED_AUDIENCES],
        start=1,
    ):
        recommendation = (
            candidate.edited_recommendation or candidate.recommendation
        )
        evidence_by_id = {
            evidence.reference_id: evidence for evidence in candidate.evidence
        }
        evidence_items: list[GroundedEvidenceItem] = []
        for reference_id in recommendation.supporting_article_reference_ids:
            if evidence_count >= MAX_GROUNDED_EVIDENCE:
                break
            source = evidence_by_id.get(reference_id)
            if source is None:
                continue
            article = source.article
            evidence_items.append(
                GroundedEvidenceItem(
                    evidence_id=f"{candidate.review_id}:{reference_id}",
                    article_title=article.title,
                    article_url=_public_article_url(article.url),
                    audience_label=recommendation.name,
                    summary=(article.summary[:600] if article.summary else None),
                    weekly_views=article.weekly_views,
                )
            )
            evidence_count += 1
        audiences.append(
            GroundedAudienceItem(
                context_rank=context_rank,
                audience_label=recommendation.name,
                description=recommendation.description[:1_000],
                publication_source=(
                    "analyst_edit"
                    if candidate.edited_recommendation is not None
                    else "original"
                ),
                cluster_name=candidate.cluster_name,
                cluster_pageviews=candidate.cluster_pageviews,
                article_count=candidate.article_count,
                size_index=recommendation.size_index,
                topic_confidence=candidate.topic_confidence,
                buying_power=recommendation.buying_power,
                buying_power_reason=recommendation.buying_power_reason[:1_000],
                brand_categories=recommendation.brand_categories,
                commercial_confidence=recommendation.commercial_confidence,
                commercial_confidence_reason=(
                    recommendation.commercial_confidence_reason[:1_000]
                ),
                evidence=tuple(evidence_items),
            )
        )

    while audiences and len(_canonical_context_json(audiences)) > (
        MAX_GROUNDED_CONTEXT_CHARS
    ):
        audiences.pop()
    return GroundedAssistantContext(
        audience_count=len(audiences),
        audiences=tuple(audiences),
    )


def context_has_publishable_evidence(context: GroundedAssistantContext) -> bool:
    return any(audience.evidence for audience in context.audiences)


def question_requests_private_data(question: str) -> bool:
    normalized = " ".join(question.casefold().split())
    forbidden_phrases = (
        "api key",
        "environment variable",
        "private feedback",
        "analyst feedback",
        "private note",
        "system prompt",
        "hidden prompt",
        "chain of thought",
        "model reasoning",
        "checkpoint",
        "thread id",
        "command digest",
        "start digest",
        "database path",
        "raw model output",
        "response id",
    )
    return any(phrase in normalized for phrase in forbidden_phrases)


def validate_grounded_response(
    response: GroundedAssistantModelResponse,
    context: GroundedAssistantContext,
) -> GroundedAssistantModelResponse | None:
    """Require allowlisted citations and context-backed numeric literals."""

    if response.evidence_status == "insufficient_evidence":
        return GroundedAssistantModelResponse(
            evidence_status="insufficient_evidence",
            answer=INSUFFICIENT_EVIDENCE_ANSWER,
            citation_ids=(),
        )
    allowlisted_ids = {
        evidence.evidence_id
        for audience in context.audiences
        for evidence in audience.evidence
    }
    if not response.citation_ids or any(
        citation_id not in allowlisted_ids
        for citation_id in response.citation_ids
    ):
        return None
    number_pattern = r"(?<!\w)\d+(?:\.\d+)?%?"
    allowed_numbers = set(
        re.findall(number_pattern, _canonical_context_json(context.audiences))
    )
    for audience in context.audiences:
        allowed_numbers.update(
            {
                f"{audience.topic_confidence * 100:.0f}%",
                f"{audience.commercial_confidence * 100:.0f}%",
                f"{audience.topic_confidence * 100:.0f}",
                f"{audience.commercial_confidence * 100:.0f}",
            }
        )
    answer_numbers = set(re.findall(number_pattern, response.answer))
    if not answer_numbers.issubset(allowed_numbers):
        return None
    return response


def deterministic_suggestions(
    context: GroundedAssistantContext,
) -> tuple[str, ...]:
    if not context.audiences:
        return ()
    suggestions = [
        "Which published audience appears strongest for a premium consumer brand?",
        "What evidence supports the published audience recommendation?",
    ]
    if len(context.audiences) > 1:
        suggestions.append(
            "How do the published audiences differ in reach and commercial confidence?"
        )
    return tuple(suggestions)


def evidence_by_id(
    context: GroundedAssistantContext,
) -> dict[str, GroundedEvidenceItem]:
    return {
        evidence.evidence_id: evidence
        for audience in context.audiences
        for evidence in audience.evidence
    }


def _canonical_context_json(
    audiences: Iterable[GroundedAudienceItem],
) -> str:
    return json.dumps(
        [audience.model_dump(mode="json") for audience in audiences],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _public_article_url(value: str) -> str | None:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return value
