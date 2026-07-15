"""FastAPI integration and explicit public projection for analyst review."""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import json
import logging
from typing import Mapping

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse
from pydantic import TypeAdapter, ValidationError

from ..agent.audience_assistant import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    GroundedAssistantProvider,
    build_grounded_context,
    context_has_publishable_evidence,
    deterministic_suggestions,
    evidence_by_id,
    question_requests_private_data,
    validate_grounded_response,
)
from ..agent.audience_provider import AudienceProviderError
from ..agent.audience_review_runtime import (
    AudienceReviewRuntime,
    AudienceReviewRuntimeError,
    NormalizedReviewTTL,
)
from ..agent.audience_review_workflow import AudienceReviewConflictError
from ..audience_analysis import prepare_audience_analysis
from ..models.audience_api import ApiErrorDetailResponse, ApiErrorResponse
from ..models.audience_assistant_api import (
    AssistantCitationResponse,
    AudienceQuestionRequest,
    AudienceQuestionResponse,
)
from ..models.audience_review import (
    ReviewArticleSnapshot,
    ReviewCandidateSnapshot,
    ReviewCommandReceipt,
    ReviewConflictCode,
    ReviewDecisionTrace,
    ReviewRecommendationSnapshot,
    ReviewRunResult,
    parse_review_command,
)
from ..models.audience_review_api import (
    AudienceReviewApiCommand,
    AudienceReviewCommandResponse,
    AudienceReviewProgressResponse,
    AudienceReviewRunResponse,
    AudienceReviewStartRequest,
    EditValidationDropResponse,
    EditingReviewResponse,
    ExpiredReviewResponse,
    PendingReviewResponse,
    ProviderSkipReviewResponse,
    PublicReviewCommandReceipt,
    PublishedAudienceResponse,
    RejectedReviewResponse,
    ReviewArticleResponse,
    ReviewDailyViewResponse,
    ReviewEvidenceResponse,
    ReviewJourneyEventResponse,
    ReviewJourneyIssueResponse,
    ReviewJourneyResponse,
    ReviewMetricCodeCountResponse,
    ReviewRecommendationResponse,
    ReviewWorkflowMetricsResponse,
    ValidationDropReviewResponse,
)
from .analysis_errors import classify_analysis_exception
from .audience_analysis import AudienceAnalysisResources


logger = logging.getLogger(__name__)

REVIEW_RESPONSE_HEADERS = {
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
}
MAX_REVIEW_COMMAND_BODY_BYTES = 16_384
STANDARD_ANALYSIS_START_CONTRACT = "standard-analysis-defaults-v1"


def _build_command_request_schema() -> dict[str, object]:
    raw = TypeAdapter(AudienceReviewApiCommand).json_schema()
    definitions = raw.pop("$defs", {})

    def inline(value: object) -> object:
        if isinstance(value, dict):
            reference = value.get("$ref")
            if isinstance(reference, str) and reference.startswith("#/$defs/"):
                name = reference.removeprefix("#/$defs/")
                return inline(definitions[name])
            return {key: inline(item) for key, item in value.items()}
        if isinstance(value, list):
            return [inline(item) for item in value]
        return value

    schema = inline(raw)
    if not isinstance(schema, dict):
        raise RuntimeError("review command schema must be an object")
    schema["discriminator"] = {"propertyName": "type"}
    return schema


REVIEW_COMMAND_REQUEST_SCHEMA = _build_command_request_schema()


class ReviewRequestParseError(ValueError):
    """Safe request-envelope failure with no submitted value attached."""

    def __init__(self, code: str = "invalid_request") -> None:
        self.code = code
        super().__init__(code)


class ReviewRequestValidationError(ReviewRequestParseError):
    """A redacted authoritative command-parser failure."""


@dataclass(frozen=True, slots=True)
class AudienceReviewResources:
    """Lifespan-owned review runtime sharing standard analysis resources."""

    analysis: AudienceAnalysisResources
    runtime: AudienceReviewRuntime
    assistant_provider: GroundedAssistantProvider | None = None


router = APIRouter(prefix="/api", tags=["audience-reviews"])


def get_audience_review_resources(request: Request) -> AudienceReviewResources:
    resources = getattr(request.app.state, "audience_review_resources", None)
    if resources is None:
        raise AudienceReviewRuntimeError("review_resources_unavailable")
    return resources


@router.post(
    "/audience-reviews",
    response_model=AudienceReviewRunResponse,
    status_code=201,
    responses={
        200: {"model": AudienceReviewRunResponse},
        409: {"model": ApiErrorResponse},
        422: {"model": ApiErrorResponse},
        500: {"model": ApiErrorResponse},
        502: {"model": ApiErrorResponse},
    },
)
async def start_audience_review(
    body: AudienceReviewStartRequest,
    response: Response,
    resources: AudienceReviewResources = Depends(get_audience_review_resources),
) -> AudienceReviewRunResponse | JSONResponse:
    """Start once with a client UUID or replay the authoritative run."""
    response.headers.update(REVIEW_RESPONSE_HEADERS)
    response.headers["Location"] = f"/api/audience-reviews/{body.run_id}"
    request_digest = ""
    owns_start = False
    try:
        try:
            normalized_ttl = resources.runtime.normalize_start_ttl(
                ttl_seconds=body.ttl_seconds
            )
        except ValueError:
            raise ReviewRequestParseError() from None
        except AudienceReviewRuntimeError as exc:
            if exc.code == "invalid_review_ttl":
                raise ReviewRequestParseError() from None
            raise
        request_digest = _start_request_digest(normalized_ttl)
        owns_start = await resources.runtime.claim_start_request(
            body.run_id,
            request_digest,
        )
        if not owns_start:
            response.status_code = 200
            response.headers["X-Idempotent-Replay"] = "true"
            return map_audience_review_run(
                await resources.runtime.get_run(body.run_id)
            )

        async with resources.analysis.analysis_lock:
            prepared = await prepare_audience_analysis(
                resources.analysis.pageview_client,
                resources.analysis.summary_client,
                resources.analysis.encoder,
            )
            result = await resources.runtime.start(
                prepared.preparation,
                resources.analysis.audience_provider,
                normalized_ttl=normalized_ttl,
                run_id=body.run_id,
                start_request_digest=request_digest,
            )
        response.headers["X-Idempotent-Replay"] = "false"
        return map_audience_review_run(result)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return _review_error_response(exc, operation="start")
    finally:
        if owns_start:
            await _release_start_request_safely(
                resources.runtime,
                body.run_id,
                request_digest,
            )


@router.get(
    "/audience-reviews/{run_id}",
    response_model=AudienceReviewRunResponse,
    responses={
        404: {"model": ApiErrorResponse},
        500: {"model": ApiErrorResponse},
    },
)
async def get_audience_review(
    run_id: str,
    response: Response,
    resources: AudienceReviewResources = Depends(get_audience_review_resources),
) -> AudienceReviewRunResponse | JSONResponse:
    response.headers.update(REVIEW_RESPONSE_HEADERS)
    try:
        return map_audience_review_run(await resources.runtime.get_run(run_id))
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return _review_error_response(exc, operation="get")


@router.post(
    "/audience-reviews/{run_id}/questions",
    response_model=AudienceQuestionResponse,
    responses={
        404: {"model": ApiErrorResponse},
        409: {"model": ApiErrorResponse},
        422: {"model": ApiErrorResponse},
        500: {"model": ApiErrorResponse},
        502: {"model": ApiErrorResponse},
    },
)
async def ask_audience_review_question(
    run_id: str,
    body: AudienceQuestionRequest,
    response: Response,
    resources: AudienceReviewResources = Depends(get_audience_review_resources),
) -> AudienceQuestionResponse | JSONResponse:
    """Answer once from a read-only, public-safe review projection."""
    response.headers.update(REVIEW_RESPONSE_HEADERS)
    try:
        result = await resources.runtime.peek_run(run_id)
        if result.status == "failed":
            return _assistant_error_response(
                409,
                "assistant_run_unavailable",
                "This review run is not available for questions.",
            )
        context = build_grounded_context(result)
        suggestions = deterministic_suggestions(context)
        if (
            not context_has_publishable_evidence(context)
            or question_requests_private_data(body.question)
        ):
            return AudienceQuestionResponse(
                answer=INSUFFICIENT_EVIDENCE_ANSWER,
                citations=(),
                evidence_status="insufficient_evidence",
                suggested_follow_up_questions=suggestions,
            )
        provider = resources.assistant_provider
        if provider is None:
            return _assistant_error_response(
                500,
                "assistant_unavailable",
                "Ask WikiPulse is temporarily unavailable.",
            )
        provider_response = await provider.answer_grounded(
            body.question,
            context,
        )
        validated = validate_grounded_response(provider_response, context)
        if validated is None:
            return AudienceQuestionResponse(
                answer=INSUFFICIENT_EVIDENCE_ANSWER,
                citations=(),
                evidence_status="insufficient_evidence",
                suggested_follow_up_questions=suggestions,
            )
        evidence = evidence_by_id(context)
        return AudienceQuestionResponse(
            answer=validated.answer,
            citations=tuple(
                AssistantCitationResponse(
                    article_title=evidence[citation_id].article_title,
                    article_url=evidence[citation_id].article_url,
                    audience_label=evidence[citation_id].audience_label,
                    relevance=(
                        "Supporting evidence for "
                        f"{evidence[citation_id].audience_label}."
                    ),
                )
                for citation_id in dict.fromkeys(validated.citation_ids)
            ),
            evidence_status=validated.evidence_status,
            suggested_follow_up_questions=suggestions,
        )
    except asyncio.CancelledError:
        raise
    except AudienceReviewConflictError as exc:
        return _review_error_response(exc, operation="get")
    except AudienceProviderError:
        return _assistant_error_response(
            502,
            "assistant_provider_failed",
            "Ask WikiPulse could not answer safely.",
        )
    except Exception as exc:
        logger.error(
            "Ask WikiPulse failed safely: context=%s",
            type(exc).__name__,
        )
        return _assistant_error_response(
            500,
            "internal_error",
            "An unexpected internal error occurred.",
        )


@router.post(
    "/audience-reviews/{run_id}/commands",
    response_model=AudienceReviewCommandResponse,
    responses={
        404: {"model": ApiErrorResponse},
        409: {"model": ApiErrorResponse},
        410: {"model": ApiErrorResponse},
        415: {"model": ApiErrorResponse},
        422: {"model": ApiErrorResponse},
        500: {"model": ApiErrorResponse},
    },
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": REVIEW_COMMAND_REQUEST_SCHEMA,
                }
            },
        }
    },
)
async def submit_audience_review_command(
    run_id: str,
    request: Request,
    response: Response,
    resources: AudienceReviewResources = Depends(get_audience_review_resources),
) -> AudienceReviewCommandResponse | JSONResponse:
    """Parse private command input safely and return the committed state."""
    response.headers.update(REVIEW_RESPONSE_HEADERS)
    try:
        body = await _read_command_body(request)
        if "run_id" in body:
            raise ReviewRequestParseError()
        internal_body = dict(body)
        internal_body["run_id"] = run_id
        try:
            validated_command = parse_review_command(internal_body)
        except ValidationError:
            raise ReviewRequestValidationError() from None
        receipt = await resources.runtime.submit_command(validated_command)
        result = await resources.runtime.get_run(run_id)
        return AudienceReviewCommandResponse(
            receipt=_map_receipt(receipt),
            run=map_audience_review_run(result),
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return _review_error_response(exc, operation="command")


async def _read_command_body(request: Request) -> Mapping[str, object]:
    media_type = request.headers.get("content-type", "").split(";", 1)[0]
    media_type = media_type.strip().lower()
    top_level, separator, subtype = media_type.partition("/")
    if not (
        media_type == "application/json"
        or (
            separator == "/"
            and top_level == "application"
            and len(subtype) > len("+json")
            and media_type.endswith("+json")
        )
    ):
        raise ReviewRequestParseError("unsupported_media_type")
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            parsed_length = int(content_length)
        except ValueError:
            raise ReviewRequestParseError() from None
        if parsed_length < 0 or parsed_length > MAX_REVIEW_COMMAND_BODY_BYTES:
            raise ReviewRequestParseError()
    chunks = bytearray()
    async for chunk in request.stream():
        chunks.extend(chunk)
        if len(chunks) > MAX_REVIEW_COMMAND_BODY_BYTES:
            raise ReviewRequestParseError()
    raw = bytes(chunks)
    try:
        value = json.loads(raw, object_pairs_hook=_strict_json_object)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ReviewRequestParseError() from None
    if not isinstance(value, dict):
        raise ReviewRequestParseError()
    return value


def _strict_json_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ReviewRequestParseError()
        result[key] = value
    return result


def _start_request_digest(ttl: NormalizedReviewTTL) -> str:
    payload = {
        "analysis_contract": STANDARD_ANALYSIS_START_CONTRACT,
        "ttl_microseconds": ttl.microseconds,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


async def _release_start_request_safely(
    runtime: AudienceReviewRuntime,
    run_id: str,
    request_digest: str,
) -> None:
    release = asyncio.create_task(
        runtime.release_start_request(run_id, request_digest)
    )
    try:
        await asyncio.shield(release)
    except asyncio.CancelledError:
        await release
        raise


def map_audience_review_run(result: ReviewRunResult) -> AudienceReviewRunResponse:
    """Project one reconciled internal result without dumping internal state."""
    if result.created_at is None or result.expires_at is None:
        raise AudienceReviewRuntimeError("review_state_unavailable")
    current_candidates = (
        tuple(
            candidate
            for candidate in result.review_candidates
            if candidate.status in {"pending_review", "editing"}
        )
        if result.status in {"pending_review", "editing"}
        else ()
    )
    if len(current_candidates) > 1:
        raise AudienceReviewRuntimeError("review_state_unavailable")
    current = current_candidates[0] if current_candidates else None
    current_review = _map_current_review(current, len(result.review_candidates))
    completed_statuses = {
        "published",
        "rejected",
        "edit_validation_dropped",
        "expired",
    }
    traces_by_review_id = {
        trace.review_id: trace
        for trace in result.traces
        if trace.review_id is not None
    }
    terminal_traces = {
        (trace.cluster_id, trace.final_outcome): trace
        for trace in result.traces
        if trace.review_id is None
    }
    published = tuple(
        PublishedAudienceResponse(
            review_id=candidate.review_id,
            cluster_id=candidate.cluster_id,
            trace_id=traces_by_review_id[candidate.review_id].trace_id,
            publication_source=(
                "analyst_edit"
                if candidate.edited_recommendation is not None
                else "original"
            ),
            audience=_map_recommendation(
                candidate.edited_recommendation or candidate.recommendation
            ),
        )
        for candidate in result.review_candidates
        if candidate.status == "published"
    )
    rejected = tuple(
        RejectedReviewResponse(
            review_id=candidate.review_id,
            cluster_id=candidate.cluster_id,
            cluster_name=candidate.cluster_name,
            reason_code=candidate.reject_reason_code,
        )
        for candidate in result.review_candidates
        if candidate.status == "rejected"
        and candidate.reject_reason_code is not None
    )
    edit_drops = tuple(
        EditValidationDropResponse(
            review_id=candidate.review_id,
            cluster_id=candidate.cluster_id,
            cluster_name=candidate.cluster_name,
            drop_code=candidate.edit_drop_code,
        )
        for candidate in result.review_candidates
        if candidate.status == "edit_validation_dropped"
        and candidate.edit_drop_code is not None
    )
    expired = tuple(
        ExpiredReviewResponse(
            review_id=candidate.review_id,
            cluster_id=candidate.cluster_id,
            cluster_name=candidate.cluster_name,
        )
        for candidate in result.review_candidates
        if candidate.status == "expired"
    )
    provider_skips = tuple(
        ProviderSkipReviewResponse(
            trace_id=terminal_traces[
                (item.cluster_id, "provider_skipped")
            ].trace_id,
            cluster_id=item.cluster_id,
            cluster_name=item.cluster_name,
            reason=item.reason,
        )
        for item in result.provider_skips
    )
    validation_drops = tuple(
        ValidationDropReviewResponse(
            trace_id=terminal_traces[
                (item.cluster_id, "validation_dropped")
            ].trace_id,
            cluster_id=item.cluster_id,
            cluster_name=item.cluster_name,
            source_known=item.source_known,
            phase=item.phase,
            drop_code=item.drop_code,
            issue_codes=item.issue_codes,
        )
        for item in result.validation_drops
    )
    return AudienceReviewRunResponse(
        run_id=result.run_id,
        status=result.status,
        is_complete=result.is_complete,
        created_at=datetime.fromisoformat(result.created_at),
        expires_at=datetime.fromisoformat(result.expires_at),
        progress=AudienceReviewProgressResponse(
            total_reviews=len(result.review_candidates),
            completed_reviews=sum(
                candidate.status in completed_statuses
                for candidate in result.review_candidates
            ),
            queued_reviews=sum(
                candidate.status == "queued"
                for candidate in result.review_candidates
            ),
            current_position=(current.ordinal + 1 if current is not None else None),
        ),
        current_review=current_review,
        published_audiences=published,
        rejected_reviews=rejected,
        edit_validation_drops=edit_drops,
        expired_reviews=expired,
        provider_skips=provider_skips,
        validation_drops=validation_drops,
        journey=tuple(_map_journey(trace) for trace in result.traces),
        automatic_workflow_metrics=_map_metrics(result),
        failure_code=result.failure_code,
    )


def _map_current_review(
    candidate: ReviewCandidateSnapshot | None,
    total_reviews: int,
) -> PendingReviewResponse | EditingReviewResponse | None:
    if candidate is None:
        return None
    common = {
        "review_id": candidate.review_id,
        "cluster_id": candidate.cluster_id,
        "position": candidate.ordinal + 1,
        "total_reviews": total_reviews,
        "cluster_name": candidate.cluster_name,
        "cluster_pageviews": candidate.cluster_pageviews,
        "article_count": candidate.article_count,
        "size_index": candidate.recommendation.size_index,
        "topic_confidence": candidate.topic_confidence,
        "original_recommendation": _map_recommendation(
            candidate.recommendation
        ),
        "evidence": tuple(
            ReviewEvidenceResponse(
                reference_id=evidence.reference_id,
                article=_map_article(evidence.article),
            )
            for evidence in candidate.evidence
        ),
    }
    if candidate.status == "pending_review":
        return PendingReviewResponse(
            expected_version=candidate.version,
            edit_available=not candidate.edit_attempted,
            **common,
        )
    if candidate.status == "editing":
        return EditingReviewResponse(**common)
    raise AudienceReviewRuntimeError("review_state_unavailable")


def _map_article(article: ReviewArticleSnapshot) -> ReviewArticleResponse:
    return ReviewArticleResponse(
        title=article.title,
        normalized_title=article.normalized_title,
        url=article.url,
        weekly_views=article.weekly_views,
        daily_views=tuple(
            ReviewDailyViewResponse(day=item.day, pageviews=item.pageviews)
            for item in article.daily_views
        ),
        summary=article.summary,
        analysis_start_date=article.analysis_start_date,
        analysis_end_date=article.analysis_end_date,
    )


def _map_recommendation(
    recommendation: ReviewRecommendationSnapshot,
) -> ReviewRecommendationResponse:
    return ReviewRecommendationResponse(
        audience_id=recommendation.audience_id,
        name=recommendation.name,
        description=recommendation.description,
        topic_cluster_ids=recommendation.topic_cluster_ids,
        size_index=recommendation.size_index,
        buying_power=recommendation.buying_power,
        buying_power_reason=recommendation.buying_power_reason,
        brand_categories=recommendation.brand_categories,
        supporting_article_reference_ids=(
            recommendation.supporting_article_reference_ids
        ),
        supporting_articles=tuple(
            _map_article(article)
            for article in recommendation.supporting_articles
        ),
        commercial_confidence=recommendation.commercial_confidence,
        commercial_confidence_reason=(
            recommendation.commercial_confidence_reason
        ),
    )


def _map_journey(trace: ReviewDecisionTrace) -> ReviewJourneyResponse:
    return ReviewJourneyResponse(
        trace_id=trace.trace_id,
        cluster_id=trace.cluster_id,
        cluster_name=trace.cluster_name,
        source_known=trace.source_known,
        final_outcome=trace.final_outcome,
        review_id=trace.review_id,
        events=tuple(
            ReviewJourneyEventResponse(
                sequence=event.sequence,
                phase=event.phase,
                code=event.code,
                outcome_code=event.outcome_code,
                issues=tuple(
                    ReviewJourneyIssueResponse(
                        code=issue.code,
                        reference_id=issue.reference_id,
                    )
                    for issue in event.issues
                ),
            )
            for event in trace.events
        ),
    )


def _map_metrics(result: ReviewRunResult) -> ReviewWorkflowMetricsResponse:
    metrics = result.metrics
    return ReviewWorkflowMetricsResponse(
        initial_decision_count=metrics.initial_decision_count,
        initial_valid_decision_count=metrics.initial_valid_decision_count,
        initial_invalid_report_count=metrics.initial_invalid_report_count,
        revision_count=metrics.revision_count,
        revision_requested_cluster_count=(
            metrics.revision_requested_cluster_count
        ),
        revision_decision_count=metrics.revision_decision_count,
        revision_valid_decision_count=metrics.revision_valid_decision_count,
        final_valid_decision_count=metrics.final_valid_decision_count,
        final_segment_count=metrics.final_segment_count,
        final_provider_skip_count=metrics.final_provider_skip_count,
        dropped_source_cluster_count=metrics.dropped_source_cluster_count,
        dropped_unmatched_decision_count=(
            metrics.dropped_unmatched_decision_count
        ),
        provider_call_count=metrics.provider_call_count,
        provider_input_tokens=metrics.provider_input_tokens,
        provider_output_tokens=metrics.provider_output_tokens,
        provider_total_tokens=metrics.provider_total_tokens,
        provider_elapsed_seconds=metrics.provider_elapsed_seconds,
        validation_issue_count=metrics.validation_issue_count,
        validation_issue_counts_by_code=tuple(
            ReviewMetricCodeCountResponse(code=item.code, count=item.count)
            for item in metrics.validation_issue_counts_by_code
        ),
        drop_counts_by_code=tuple(
            ReviewMetricCodeCountResponse(code=item.code, count=item.count)
            for item in metrics.drop_counts_by_code
        ),
    )


def _map_receipt(receipt: ReviewCommandReceipt) -> PublicReviewCommandReceipt:
    return PublicReviewCommandReceipt(
        command_id=receipt.command_id,
        type=receipt.command_type,
        review_id=receipt.review_id,
        cluster_id=receipt.cluster_id,
        accepted=receipt.accepted,
        idempotent_replay=receipt.idempotent_replay,
        resulting_status=receipt.resulting_status,
        run_status=receipt.run_status,
    )


def _review_error_response(
    exc: Exception,
    *,
    operation: str,
) -> JSONResponse:
    status_code, code, message = _classify_review_error(exc, operation=operation)
    log_level = logging.INFO if status_code < 500 else logging.ERROR
    safe_context = exc.code.value if isinstance(
        exc, AudienceReviewConflictError
    ) else type(exc).__name__
    logger.log(
        log_level,
        "Audience review API operation failed: operation=%s context=%s",
        operation,
        safe_context,
    )
    payload = ApiErrorResponse(
        error=ApiErrorDetailResponse(code=code, message=message)
    )
    return JSONResponse(
        status_code=status_code,
        content=payload.model_dump(mode="json"),
        headers=REVIEW_RESPONSE_HEADERS,
    )


def _assistant_error_response(
    status_code: int,
    code: str,
    message: str,
) -> JSONResponse:
    payload = ApiErrorResponse(
        error=ApiErrorDetailResponse(code=code, message=message)
    )
    return JSONResponse(
        status_code=status_code,
        content=payload.model_dump(mode="json"),
        headers=REVIEW_RESPONSE_HEADERS,
    )


def _classify_review_error(
    exc: Exception,
    *,
    operation: str,
) -> tuple[int, str, str]:
    if isinstance(exc, ReviewRequestParseError):
        if exc.code == "unsupported_media_type":
            return (
                415,
                "unsupported_media_type",
                "The review command must use a JSON media type.",
            )
        code = "invalid_review_start" if operation == "start" else (
            "invalid_review_command"
        )
        noun = "start request" if operation == "start" else "review command"
        return 422, code, f"The {noun} is invalid."
    if isinstance(exc, AudienceReviewConflictError):
        mappings = {
            ReviewConflictCode.RUN_NOT_FOUND: (
                404,
                "review_run_not_found",
                "The review run is unavailable.",
            ),
            ReviewConflictCode.RUN_EXPIRED: (
                410,
                "review_run_expired",
                "The review run has expired.",
            ),
            ReviewConflictCode.STALE_VERSION: (
                409,
                "review_version_conflict",
                "The review changed before this command was accepted.",
            ),
            ReviewConflictCode.COMMAND_ID_REUSED: (
                409,
                "review_command_id_reused",
                "The command identifier was already used differently.",
            ),
            ReviewConflictCode.REVIEW_CURRENTLY_EDITING: (
                409,
                "review_currently_editing",
                "An analyst edit is already in progress.",
            ),
            ReviewConflictCode.EDIT_ALREADY_ATTEMPTED: (
                409,
                "review_edit_already_attempted",
                "The analyst edit allowance has already been used.",
            ),
            ReviewConflictCode.REVIEW_NOT_PENDING: (
                409,
                "review_not_pending",
                "The requested review is not pending.",
            ),
            ReviewConflictCode.RUN_TERMINAL: (
                409,
                "review_not_pending",
                "The requested review is not pending.",
            ),
            ReviewConflictCode.RUN_ID_MISMATCH: (
                409,
                "review_identity_conflict",
                "The review command identity does not match.",
            ),
            ReviewConflictCode.REVIEW_ID_MISMATCH: (
                409,
                "review_identity_conflict",
                "The review command identity does not match.",
            ),
            ReviewConflictCode.CLUSTER_ID_MISMATCH: (
                409,
                "review_identity_conflict",
                "The review command identity does not match.",
            ),
            ReviewConflictCode.WRONG_THREAD: (
                409,
                "review_identity_conflict",
                "The review command identity does not match.",
            ),
            ReviewConflictCode.CHECKPOINT_NOT_FOUND: (
                500,
                "review_state_unavailable",
                "The review state is temporarily unavailable.",
            ),
        }
        return mappings[exc.code]
    if isinstance(exc, AudienceReviewRuntimeError):
        if exc.code == "review_start_request_conflict":
            return (
                409,
                "review_start_request_conflict",
                "The review run was started with different configuration.",
            )
        if operation == "start" and exc.code == "duplicate_run_id":
            return (
                409,
                "review_identity_conflict",
                "The review run identity is already in use.",
            )
        if operation == "start":
            return 500, "review_start_failed", "The review run could not start."
        return (
            500,
            "review_state_unavailable",
            "The review state is temporarily unavailable.",
        )
    if operation == "start":
        analysis_error = classify_analysis_exception(exc)
        if analysis_error.code != "internal_error":
            return (
                analysis_error.status_code,
                analysis_error.code,
                analysis_error.message,
            )
    return 500, "internal_error", "An unexpected internal error occurred."
