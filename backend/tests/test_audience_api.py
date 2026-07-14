"""Focused lifecycle and HTTP tests for the audience-analysis API."""

import asyncio
from datetime import date
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from fastapi.testclient import TestClient

from app.agent.audience_finalization import (
    INVALID_TOTAL_ANALYZED_VIEWS,
    AudienceSourceIntegrityError,
)
from app.agent.audience_provider import AudienceProviderError
from app.agent.audience_trace import AudienceTraceInvariantError
from app.api.audience_analysis import AudienceAnalysisResources
from app.audience_analysis import (
    ROUTING_PARTITION_MISMATCH,
    AudienceAnalysisInvariantError,
)
from app.main import create_app
from app.models import Article, AudienceSegment, TopicCluster
from app.services.wikimedia_client import WikimediaPageviewsError
from app.services.wikipedia_summary_client import WikipediaSummaryError


def make_article(title: str, views: int) -> Article:
    observed_date = date(2026, 7, 12)
    return Article(
        title=title,
        normalized_title=title,
        url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
        weekly_views=views,
        daily_views={observed_date: views},
        summary=f"{title} is a concise source summary.",
        analysis_start_date=date(2026, 7, 6),
        analysis_end_date=observed_date,
    )


def make_cluster(cluster_id: str, views: tuple[int, int]) -> TopicCluster:
    articles = [
        make_article(f"{cluster_id} Alpha", views[0]),
        make_article(f"{cluster_id} Beta", views[1]),
    ]
    return TopicCluster(
        id=cluster_id,
        name=f"Topic {cluster_id}",
        description=f"A deterministic description for {cluster_id}.",
        articles=articles,
        keywords=[cluster_id, "example"],
        total_views=sum(views),
        article_count=2,
        confidence_score=0.8,
    )


def make_topic_metrics(*, empty: bool = False) -> SimpleNamespace:
    count = 0 if empty else 6
    return SimpleNamespace(
        fetched_article_count=count,
        rejected_article_count=0 if empty else 1,
        eligible_article_count=0 if empty else 5,
        top_n_omitted_article_count=0,
        selected_article_count=0 if empty else 4,
        summary_available_article_count=0 if empty else 4,
        summary_missing_article_count=0,
        topic_cluster_count=0 if empty else 3,
        clustered_article_count=0 if empty else 6,
        unclustered_article_count=0 if empty else 1,
        selected_pageviews=0 if empty else 1_500,
    )


def make_funnel_metrics(*, empty: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        topic_cluster_count=0 if empty else 3,
        commercial_eligible_cluster_count=0 if empty else 2,
        commercial_skipped_cluster_count=0 if empty else 1,
        prepared_cluster_count=0 if empty else 2,
        final_segment_count=0 if empty else 1,
        provider_skipped_cluster_count=0 if empty else 1,
        validation_dropped_source_cluster_count=0,
        unmatched_provider_output_count=0 if empty else 1,
        commercial_eligible_pageviews=0 if empty else 1_000,
        represented_audience_pageviews=0 if empty else 700,
        commercial_skip_counts_by_reason=(
            {} if empty else {"low_topic_cohesion": 1}
        ),
    )


def make_workflow_metrics(*, empty: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        initial_decision_count=0 if empty else 3,
        initial_valid_decision_count=0 if empty else 2,
        initial_invalid_report_count=0 if empty else 1,
        revision_count=0,
        revision_requested_cluster_count=0,
        revision_decision_count=0,
        revision_valid_decision_count=0,
        final_valid_decision_count=0 if empty else 2,
        final_segment_count=0 if empty else 1,
        final_provider_skip_count=0 if empty else 1,
        dropped_source_cluster_count=0,
        dropped_unmatched_decision_count=0 if empty else 1,
        provider_call_count=0 if empty else 1,
        provider_input_tokens=0 if empty else 100,
        provider_output_tokens=0 if empty else 50,
        provider_total_tokens=0 if empty else 150,
        provider_elapsed_seconds=0.0 if empty else 0.4,
        validation_issue_count=0 if empty else 1,
        validation_issue_counts_by_code=(
            {} if empty else {"unknown_decision_cluster": 1}
        ),
        drop_counts_by_code=(
            {} if empty else {"unmatched_initial_decision": 1}
        ),
    )


def make_internal_result(*, empty: bool = False) -> SimpleNamespace:
    if empty:
        topic_result = SimpleNamespace(
            topics=(),
            unclustered_articles=(),
            rejected_articles=(),
            metrics=make_topic_metrics(empty=True),
        )
        workflow = SimpleNamespace(
            metrics=make_workflow_metrics(empty=True),
            segments=(),
            provider_skips=(),
            dropped_decisions=(),
            initial_validation_report=None,
            revision_validation_report=None,
        )
        return SimpleNamespace(
            topic_analysis=topic_result,
            preparation=SimpleNamespace(clusters=()),
            audience_workflow=workflow,
            metrics=make_funnel_metrics(empty=True),
            segments=(),
            commercial_skips=(),
            provider_skips=(),
            dropped_decisions=(),
            is_publishable=True,
        )

    created = make_cluster("created", (400, 300))
    skipped = make_cluster("provider-skip", (200, 100))
    commercial = make_cluster("commercial-skip", (120, 80))
    unclustered = make_article("Unclustered", 75)
    rejected = make_article("Main Page", 50)
    segment = AudienceSegment(
        id="audience-created",
        name="Created Followers",
        description="People following the created topic and its developments.",
        topic_cluster_ids=["created"],
        size_index=46.67,
        buying_power="medium",
        buying_power_reason="The audience has repeat category spending.",
        brand_categories=["Media", "Technology"],
        supporting_articles=list(created.articles),
        commercial_confidence=0.78,
        commercial_confidence_reason=(
            "The source articles provide coherent commercial evidence."
        ),
    )
    issue = SimpleNamespace(
        code="unknown_decision_cluster",
        reference_id=None,
    )
    dropped = SimpleNamespace(
        cluster_id="outside",
        source_cluster=None,
        decisions=(object(),),
        phase="initial",
        drop_code="unmatched_initial_decision",
        issues=(issue,),
    )
    topic_result = SimpleNamespace(
        topics=(created, skipped, commercial),
        unclustered_articles=(unclustered,),
        rejected_articles=(
            SimpleNamespace(article=rejected, reason="main_page"),
        ),
        metrics=make_topic_metrics(),
    )
    provider_skip = SimpleNamespace(
        cluster=skipped,
        reason="The topic did not support a sufficiently specific audience.",
    )
    unknown_invalid = SimpleNamespace(
        cluster_id="outside",
        source_cluster=None,
        decisions=(object(),),
        issues=(issue,),
    )
    initial_report = SimpleNamespace(
        valid_segments=(segment,),
        provider_skips=(provider_skip,),
        invalid_decisions=(unknown_invalid,),
    )
    workflow = SimpleNamespace(
        metrics=make_workflow_metrics(),
        segments=(segment,),
        provider_skips=(provider_skip,),
        dropped_decisions=(dropped,),
        initial_validation_report=initial_report,
        revision_validation_report=None,
    )
    return SimpleNamespace(
        topic_analysis=topic_result,
        preparation=SimpleNamespace(
            clusters=(
                SimpleNamespace(
                    cluster=created,
                    cluster_id="created",
                    context=SimpleNamespace(name=created.name),
                ),
                SimpleNamespace(
                    cluster=skipped,
                    cluster_id="provider-skip",
                    context=SimpleNamespace(name=skipped.name),
                ),
            )
        ),
        audience_workflow=workflow,
        metrics=make_funnel_metrics(),
        segments=(segment,),
        commercial_skips=(
            SimpleNamespace(
                cluster=commercial,
                reason="low_topic_cohesion",
            ),
        ),
        provider_skips=(provider_skip,),
        dropped_decisions=(dropped,),
        is_publishable=True,
    )


def make_partial_result() -> SimpleNamespace:
    result = make_internal_result()
    source_cluster = result.preparation.clusters[1].cluster
    dropped = SimpleNamespace(
        cluster_id="provider-skip",
        source_cluster=source_cluster,
        decisions=(),
        phase="revision",
        drop_code="revision_provider_failure",
        issues=(
            SimpleNamespace(
                code="missing_cluster_decision",
                reference_id=None,
            ),
        ),
    )
    workflow_metrics = make_workflow_metrics()
    workflow_metrics.final_valid_decision_count = 1
    workflow_metrics.final_provider_skip_count = 0
    workflow_metrics.dropped_source_cluster_count = 1
    workflow_metrics.dropped_unmatched_decision_count = 0
    workflow_metrics.revision_count = 1
    workflow_metrics.revision_requested_cluster_count = 1
    workflow_metrics.validation_issue_counts_by_code = {
        "missing_cluster_decision": 1
    }
    workflow_metrics.drop_counts_by_code = {"revision_provider_failure": 1}
    missing_issue = dropped.issues[0]
    result.audience_workflow = SimpleNamespace(
        metrics=workflow_metrics,
        segments=result.segments,
        provider_skips=(),
        dropped_decisions=(dropped,),
        initial_validation_report=SimpleNamespace(
            valid_segments=result.segments,
            provider_skips=(),
            invalid_decisions=(
                SimpleNamespace(
                    cluster_id="provider-skip",
                    source_cluster=source_cluster,
                    decisions=(),
                    issues=(missing_issue,),
                ),
            ),
        ),
        revision_validation_report=None,
    )
    result.provider_skips = ()
    result.dropped_decisions = (dropped,)
    result.metrics.provider_skipped_cluster_count = 0
    result.metrics.validation_dropped_source_cluster_count = 1
    result.metrics.unmatched_provider_output_count = 0
    return result


def make_resources() -> AudienceAnalysisResources:
    provider = SimpleNamespace(aclose=AsyncMock())
    return AudienceAnalysisResources(
        pageview_client=object(),  # type: ignore[arg-type]
        summary_client=object(),  # type: ignore[arg-type]
        encoder=object(),  # type: ignore[arg-type]
        audience_provider=provider,  # type: ignore[arg-type]
        analysis_lock=asyncio.Lock(),
    )


class ManagedAsyncResource:
    def __init__(self) -> None:
        self.enter_count = 0
        self.exit_count = 0

    async def __aenter__(self):
        self.enter_count += 1
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        self.exit_count += 1


class AudienceApiLifecycleTests(unittest.TestCase):
    def test_app_factory_is_import_safe_before_lifespan(self) -> None:
        with (
            patch("app.main.WikimediaPageviewsClient") as pageviews,
            patch("app.main.WikipediaSummaryClient") as summaries,
            patch("app.main.MiniLMArticleEncoder") as encoder,
            patch(
                "app.main.OpenAIAudienceProvider.from_environment"
            ) as provider,
        ):
            create_app()

        pageviews.assert_not_called()
        summaries.assert_not_called()
        encoder.assert_not_called()
        provider.assert_not_called()

    def test_production_lifespan_creates_reuses_and_closes_once(self) -> None:
        pageviews = ManagedAsyncResource()
        summaries = ManagedAsyncResource()
        provider = SimpleNamespace(aclose=AsyncMock())
        encoder = object()

        with (
            patch("app.main.WikimediaPageviewsClient", return_value=pageviews) as pv,
            patch("app.main.WikipediaSummaryClient", return_value=summaries) as sm,
            patch("app.main.MiniLMArticleEncoder", return_value=encoder) as enc,
            patch(
                "app.main.OpenAIAudienceProvider.from_environment",
                return_value=provider,
            ) as configured_provider,
        ):
            application = create_app()
            with TestClient(application) as client:
                resources = client.app.state.audience_analysis_resources
                self.assertIs(resources.pageview_client, pageviews)
                self.assertIs(resources.summary_client, summaries)
                self.assertIs(resources.encoder, encoder)
                self.assertIs(resources.audience_provider, provider)
                self.assertIsInstance(resources.analysis_lock, asyncio.Lock)

        pv.assert_called_once_with()
        sm.assert_called_once_with()
        enc.assert_called_once_with()
        configured_provider.assert_called_once_with()
        self.assertEqual(pageviews.enter_count, 1)
        self.assertEqual(pageviews.exit_count, 1)
        self.assertEqual(summaries.enter_count, 1)
        self.assertEqual(summaries.exit_count, 1)
        provider.aclose.assert_awaited_once_with()


class AudienceApiEndpointTests(unittest.TestCase):
    def test_serializes_public_output_and_reuses_injected_resources(self) -> None:
        resources = make_resources()
        internal_result = make_internal_result()
        analyze = AsyncMock(return_value=internal_result)
        application = create_app(resources=resources)

        with (
            patch("app.api.audience_analysis.analyze_audiences", new=analyze),
            TestClient(application) as client,
        ):
            first = client.post("/api/audience-analysis")
            second = client.post("/api/audience-analysis")

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(analyze.await_count, 2)
        for call in analyze.await_args_list:
            self.assertIs(call.args[0], resources.pageview_client)
            self.assertIs(call.args[1], resources.summary_client)
            self.assertIs(call.args[2], resources.encoder)
            self.assertIs(call.args[3], resources.audience_provider)
        resources.audience_provider.aclose.assert_not_awaited()

        payload = first.json()
        self.assertEqual([topic["id"] for topic in payload["topics"]], [
            "created",
            "provider-skip",
            "commercial-skip",
        ])
        self.assertEqual(payload["audience_segments"][0]["id"], "audience-created")
        self.assertEqual(payload["unclustered_articles"][0]["title"], "Unclustered")
        self.assertEqual(payload["rejected_articles"][0]["reason"], "main_page")
        self.assertEqual(
            payload["commercial_skips"][0]["reason"],
            "low_topic_cohesion",
        )
        self.assertEqual(
            payload["provider_skips"][0]["cluster_id"],
            "provider-skip",
        )
        self.assertEqual(payload["validation_drops"][0]["cluster_id"], "outside")
        self.assertEqual(
            [trace["cluster_id"] for trace in payload["audience_traces"]],
            ["created", "provider-skip", "outside"],
        )
        self.assertEqual(
            payload["audience_segments"][0]["trace_id"],
            payload["audience_traces"][0]["trace_id"],
        )
        self.assertEqual(
            payload["provider_skips"][0]["trace_id"],
            payload["audience_traces"][1]["trace_id"],
        )
        self.assertEqual(
            payload["validation_drops"][0]["trace_id"],
            payload["audience_traces"][2]["trace_id"],
        )
        self.assertNotIn(
            "commercial-skip",
            {trace["cluster_id"] for trace in payload["audience_traces"]},
        )
        self.assertTrue(payload["is_publishable"])
        self.assertEqual(
            payload["metrics"]["workflow"]["provider_total_tokens"],
            150,
        )
        response_text = first.text.casefold()
        for forbidden in (
            "preparation",
            "resolution_map",
            "raw_decision",
            "response_id",
            "api_key",
            "openai_api_key",
            "prompt",
        ):
            self.assertNotIn(forbidden, response_text)

    def test_empty_output_returns_200_without_internal_metadata(self) -> None:
        resources = make_resources()
        analyze = AsyncMock(return_value=make_internal_result(empty=True))
        application = create_app(resources=resources)

        with (
            patch("app.api.audience_analysis.analyze_audiences", new=analyze),
            TestClient(application) as client,
        ):
            response = client.post("/api/audience-analysis")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["topics"], [])
        self.assertEqual(payload["audience_segments"], [])
        self.assertEqual(payload["commercial_skips"], [])
        self.assertEqual(payload["provider_skips"], [])
        self.assertEqual(payload["validation_drops"], [])
        self.assertEqual(payload["audience_traces"], [])
        self.assertTrue(payload["is_publishable"])
        self.assertEqual(
            payload["metrics"]["workflow"]["provider_call_count"],
            0,
        )

    def test_revision_failure_partial_success_returns_200(self) -> None:
        resources = make_resources()
        analyze = AsyncMock(return_value=make_partial_result())
        application = create_app(resources=resources)

        with (
            patch("app.api.audience_analysis.analyze_audiences", new=analyze),
            TestClient(application) as client,
        ):
            response = client.post("/api/audience-analysis")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload["audience_segments"]), 1)
        self.assertEqual(payload["provider_skips"], [])
        self.assertEqual(
            payload["validation_drops"][0]["drop_code"],
            "revision_provider_failure",
        )
        self.assertEqual(
            [
                event["code"]
                for event in payload["audience_traces"][1]["events"]
            ],
            [
                "generation_requested",
                "validation_failed",
                "revision_requested",
                "revision_failed",
                "decision_dropped",
            ],
        )
        self.assertTrue(payload["is_publishable"])

    def test_trace_invariant_uses_safe_analysis_invariant_envelope(self) -> None:
        resources = make_resources()
        application = create_app(resources=resources)

        with (
            patch(
                "app.api.audience_analysis.analyze_audiences",
                new=AsyncMock(return_value=make_internal_result()),
            ),
            patch(
                "app.api.audience_analysis.build_audience_decision_traces",
                side_effect=AudienceTraceInvariantError(
                    "secret trace mismatch"
                ),
            ),
            TestClient(application) as client,
        ):
            response = client.post("/api/audience-analysis")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json(),
            {
                "error": {
                    "code": "analysis_invariant_failed",
                    "message": (
                        "Audience analysis produced an inconsistent "
                        "internal result."
                    ),
                }
            },
        )
        self.assertNotIn("secret trace mismatch", response.text)

    def test_domain_and_unexpected_errors_use_safe_stable_envelopes(self) -> None:
        resources = make_resources()
        application = create_app(resources=resources)
        cases = (
            (
                WikimediaPageviewsError("secret upstream response"),
                502,
                "wikimedia_pageviews_unavailable",
            ),
            (
                WikipediaSummaryError("secret summary response"),
                502,
                "wikipedia_summaries_unavailable",
            ),
            (
                AudienceSourceIntegrityError(INVALID_TOTAL_ANALYZED_VIEWS),
                500,
                INVALID_TOTAL_ANALYZED_VIEWS,
            ),
            (
                AudienceProviderError("secret prompt and server-key"),
                502,
                "audience_provider_unavailable",
            ),
            (
                AudienceAnalysisInvariantError(ROUTING_PARTITION_MISMATCH),
                500,
                "analysis_invariant_failed",
            ),
            (
                RuntimeError("secret environment value"),
                500,
                "internal_error",
            ),
        )

        with TestClient(application, raise_server_exceptions=False) as client:
            for error, status_code, code in cases:
                with self.subTest(error=type(error).__name__):
                    with patch(
                        "app.api.audience_analysis.analyze_audiences",
                        new=AsyncMock(side_effect=error),
                    ):
                        response = client.post("/api/audience-analysis")

                    self.assertEqual(response.status_code, status_code)
                    self.assertEqual(response.json()["error"]["code"], code)
                    response_text = response.text.casefold()
                    for secret in (
                        "secret upstream response",
                        "secret summary response",
                        "secret prompt",
                        "server-key",
                        "secret environment value",
                    ):
                        self.assertNotIn(secret, response_text)

    def test_request_validation_uses_safe_stable_envelope(self) -> None:
        application = create_app(resources=make_resources())

        @application.get("/validation-probe", include_in_schema=False)
        async def validation_probe(count: int) -> dict[str, int]:
            return {"count": count}

        with TestClient(application) as client:
            response = client.get("/validation-probe", params={"count": "bad"})

        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            response.json(),
            {
                "error": {
                    "code": "request_validation_failed",
                    "message": "The request was not valid.",
                }
            },
        )
        self.assertNotIn("bad", response.text)


if __name__ == "__main__":
    unittest.main()
