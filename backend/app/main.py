"""Import-safe FastAPI application factory for WikiPulse."""

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
import logging
import os
from pathlib import Path
from typing import AsyncIterator
from dotenv import load_dotenv

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

load_dotenv(
    Path(__file__).resolve().parents[1] / ".env",
    override=False,
)

from .agent.audience_finalization import AudienceSourceIntegrityError
from .agent.audience_provider import AudienceProviderError
from .agent.audience_workflow import AudienceWorkflowInvariantError
from .agent.openai_audience_provider import OpenAIAudienceProvider
from .agent.audience_review_runtime import (
    AudienceReviewRuntime,
    AudienceReviewRuntimeError,
)
from .api.audience_analysis import (
    AudienceAnalysisResources,
    router as audience_analysis_router,
)
from .api.audience_reviews import (
    AudienceReviewResources,
    REVIEW_RESPONSE_HEADERS,
    router as audience_review_router,
)
from .api.analysis_errors import classify_analysis_exception
from .audience_analysis import AudienceAnalysisInvariantError
from .clustering.semantic_clustering import MiniLMArticleEncoder
from .models.audience_api import ApiErrorDetailResponse, ApiErrorResponse
from .services.wikimedia_client import (
    WikimediaPageviewsClient,
    WikimediaPageviewsError,
)
from .services.wikipedia_summary_client import (
    WikipediaSummaryClient,
    WikipediaSummaryError,
)
from .topic_analysis import TopicAnalysisInvariantError


logger = logging.getLogger(__name__)


@asynccontextmanager
async def _production_lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create and reliably close one reusable production resource set."""
    provider = OpenAIAudienceProvider.from_environment()
    provider_owned_by_runtime = False
    try:
        async with AsyncExitStack() as stack:
            pageview_client = await stack.enter_async_context(
                WikimediaPageviewsClient()
            )
            summary_client = await stack.enter_async_context(
                WikipediaSummaryClient()
            )
            encoder = await asyncio.to_thread(MiniLMArticleEncoder)
            analysis_lock = asyncio.Lock()
            analysis_resources = AudienceAnalysisResources(
                pageview_client=pageview_client,
                summary_client=summary_client,
                encoder=encoder,
                audience_provider=provider,
                analysis_lock=analysis_lock,
            )
            review_runtime = AudienceReviewRuntime(
                provider_call_lock=analysis_lock,
                provider_cleanup=provider.aclose,
                durable_path=os.environ.get(
                    "WIKIPULSE_REVIEW_DB_PATH",
                    str(
                        Path(__file__).resolve().parents[1]
                        / "data"
                        / "wikipulse_review.db"
                    ),
                ),
            )
            provider_owned_by_runtime = True
            stack.push_async_callback(review_runtime.aclose)
            hydrate = getattr(review_runtime, "hydrate", None)
            if hydrate is not None:
                await hydrate(provider)
            app.state.audience_analysis_resources = analysis_resources
            app.state.audience_review_resources = AudienceReviewResources(
                analysis=analysis_resources,
                runtime=review_runtime,
                assistant_provider=provider,
            )
            try:
                yield
            finally:
                app.state.audience_review_resources = None
                app.state.audience_analysis_resources = None
    finally:
        if not provider_owned_by_runtime:
            await provider.aclose()


def _injected_lifespan(
    resources: AudienceAnalysisResources,
    review_resources: AudienceReviewResources | None,
):
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.audience_analysis_resources = resources
        app.state.audience_review_resources = review_resources
        try:
            yield
        finally:
            if review_resources is not None:
                await review_resources.runtime.aclose()
            app.state.audience_review_resources = None
            app.state.audience_analysis_resources = None

    return lifespan


def create_app(
    *,
    resources: AudienceAnalysisResources | None = None,
    review_resources: AudienceReviewResources | None = None,
) -> FastAPI:
    """Create the API without initializing expensive resources at import time."""
    if resources is None and review_resources is None:
        lifespan = _production_lifespan
    else:
        if resources is not None:
            injected_analysis = resources
        else:
            assert review_resources is not None
            injected_analysis = review_resources.analysis
        if (
            review_resources is not None
            and review_resources.analysis is not injected_analysis
        ):
            raise ValueError("review resources must share analysis resources")
        lifespan = _injected_lifespan(
            injected_analysis,
            review_resources,
        )
    application = FastAPI(
        title="WikiPulse API",
        debug=False,
        lifespan=lifespan,
    )
    application.include_router(audience_analysis_router)
    application.include_router(audience_review_router)
    _register_exception_handlers(application)
    return application


def _register_exception_handlers(application: FastAPI) -> None:
    application.add_exception_handler(
        WikimediaPageviewsError,
        _wikimedia_pageviews_error_handler,
    )
    application.add_exception_handler(
        WikipediaSummaryError,
        _wikipedia_summary_error_handler,
    )
    application.add_exception_handler(
        AudienceSourceIntegrityError,
        _source_integrity_error_handler,
    )
    application.add_exception_handler(
        AudienceProviderError,
        _provider_error_handler,
    )
    application.add_exception_handler(
        AudienceReviewRuntimeError,
        _review_runtime_error_handler,
    )
    for invariant_error in (
        TopicAnalysisInvariantError,
        AudienceWorkflowInvariantError,
        AudienceAnalysisInvariantError,
    ):
        application.add_exception_handler(
            invariant_error,
            _invariant_error_handler,
        )
    application.add_exception_handler(
        RequestValidationError,
        _request_validation_error_handler,
    )
    application.add_exception_handler(Exception, _unexpected_error_handler)


async def _wikimedia_pageviews_error_handler(
    request: Request,
    exc: WikimediaPageviewsError,
) -> JSONResponse:
    logger.warning(
        "Audience analysis Pageviews source failed: %s",
        type(exc).__name__,
    )
    error = classify_analysis_exception(exc)
    return _error_response(error.status_code, error.code, error.message)


async def _wikipedia_summary_error_handler(
    request: Request,
    exc: WikipediaSummaryError,
) -> JSONResponse:
    logger.warning(
        "Audience analysis summary source failed unexpectedly: %s",
        type(exc).__name__,
    )
    error = classify_analysis_exception(exc)
    return _error_response(error.status_code, error.code, error.message)


async def _source_integrity_error_handler(
    request: Request,
    exc: AudienceSourceIntegrityError,
) -> JSONResponse:
    logger.error("Audience source integrity failed: code=%s", exc.code)
    error = classify_analysis_exception(exc)
    return _error_response(error.status_code, error.code, error.message)


async def _provider_error_handler(
    request: Request,
    exc: AudienceProviderError,
) -> JSONResponse:
    logger.warning(
        "Initial audience provider request failed: %s",
        type(exc).__name__,
    )
    error = classify_analysis_exception(exc)
    return _error_response(error.status_code, error.code, error.message)


async def _review_runtime_error_handler(
    request: Request,
    exc: AudienceReviewRuntimeError,
) -> JSONResponse:
    logger.error(
        "Audience review resources are unavailable: %s",
        type(exc).__name__,
    )
    return _error_response(
        500,
        "review_state_unavailable",
        "The review state is temporarily unavailable.",
        headers=REVIEW_RESPONSE_HEADERS,
    )


async def _invariant_error_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    logger.error("Audience analysis invariant failed: %s", type(exc).__name__)
    error = classify_analysis_exception(exc)
    return _error_response(error.status_code, error.code, error.message)


async def _request_validation_error_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    logger.info("Audience API request validation failed")
    if request.url.path == "/api/audience-reviews":
        return _error_response(
            422,
            "invalid_review_start",
            "The review start request is invalid.",
            headers=REVIEW_RESPONSE_HEADERS,
        )
    if request.url.path.endswith("/questions"):
        return _error_response(
            422,
            "invalid_assistant_question",
            "The assistant question is invalid.",
            headers=REVIEW_RESPONSE_HEADERS,
        )
    error = classify_analysis_exception(exc)
    return _error_response(error.status_code, error.code, error.message)


async def _unexpected_error_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    logger.error("Unexpected audience API failure: %s", type(exc).__name__)
    error = classify_analysis_exception(exc)
    return _error_response(error.status_code, error.code, error.message)


def _error_response(
    status_code: int,
    code: str,
    message: str,
    *,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    payload = ApiErrorResponse(
        error=ApiErrorDetailResponse(code=code, message=message)
    )
    return JSONResponse(
        status_code=status_code,
        content=payload.model_dump(mode="json"),
        headers=headers,
    )


app = create_app()
