"""Import-safe FastAPI application factory for WikiPulse."""

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
import logging
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .agent.audience_finalization import AudienceSourceIntegrityError
from .agent.audience_provider import AudienceProviderError
from .agent.audience_workflow import AudienceWorkflowInvariantError
from .agent.openai_audience_provider import OpenAIAudienceProvider
from .api.audience_analysis import AudienceAnalysisResources, router
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
    async with AsyncExitStack() as stack:
        provider = OpenAIAudienceProvider.from_environment()
        stack.push_async_callback(provider.aclose)
        pageview_client = await stack.enter_async_context(
            WikimediaPageviewsClient()
        )
        summary_client = await stack.enter_async_context(
            WikipediaSummaryClient()
        )
        encoder = await asyncio.to_thread(MiniLMArticleEncoder)
        app.state.audience_analysis_resources = AudienceAnalysisResources(
            pageview_client=pageview_client,
            summary_client=summary_client,
            encoder=encoder,
            audience_provider=provider,
            analysis_lock=asyncio.Lock(),
        )
        try:
            yield
        finally:
            app.state.audience_analysis_resources = None


def _injected_lifespan(
    resources: AudienceAnalysisResources,
):
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.audience_analysis_resources = resources
        try:
            yield
        finally:
            app.state.audience_analysis_resources = None

    return lifespan


def create_app(
    *,
    resources: AudienceAnalysisResources | None = None,
) -> FastAPI:
    """Create the API without initializing expensive resources at import time."""
    lifespan = (
        _production_lifespan
        if resources is None
        else _injected_lifespan(resources)
    )
    application = FastAPI(
        title="WikiPulse API",
        debug=False,
        lifespan=lifespan,
    )
    application.include_router(router)
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
    return _error_response(
        502,
        "wikimedia_pageviews_unavailable",
        "Wikipedia pageview data is temporarily unavailable.",
    )


async def _wikipedia_summary_error_handler(
    request: Request,
    exc: WikipediaSummaryError,
) -> JSONResponse:
    logger.warning(
        "Audience analysis summary source failed unexpectedly: %s",
        type(exc).__name__,
    )
    return _error_response(
        502,
        "wikipedia_summaries_unavailable",
        "Wikipedia summaries are temporarily unavailable.",
    )


async def _source_integrity_error_handler(
    request: Request,
    exc: AudienceSourceIntegrityError,
) -> JSONResponse:
    logger.error("Audience source integrity failed: code=%s", exc.code)
    return _error_response(
        500,
        exc.code,
        "Audience source data failed deterministic validation.",
    )


async def _provider_error_handler(
    request: Request,
    exc: AudienceProviderError,
) -> JSONResponse:
    logger.warning(
        "Initial audience provider request failed: %s",
        type(exc).__name__,
    )
    return _error_response(
        502,
        "audience_provider_unavailable",
        "Audience generation is temporarily unavailable.",
    )


async def _invariant_error_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    logger.error("Audience analysis invariant failed: %s", type(exc).__name__)
    return _error_response(
        500,
        "analysis_invariant_failed",
        "Audience analysis produced an inconsistent internal result.",
    )


async def _request_validation_error_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    logger.info("Audience API request validation failed")
    return _error_response(
        422,
        "request_validation_failed",
        "The request was not valid.",
    )


async def _unexpected_error_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    logger.error("Unexpected audience API failure: %s", type(exc).__name__)
    return _error_response(
        500,
        "internal_error",
        "An unexpected internal error occurred.",
    )


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    payload = ApiErrorResponse(
        error=ApiErrorDetailResponse(code=code, message=message)
    )
    return JSONResponse(
        status_code=status_code,
        content=payload.model_dump(mode="json"),
    )


app = create_app()
