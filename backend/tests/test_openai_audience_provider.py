"""Mocked tests for the structured OpenAI audience provider."""

import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx
from openai import APIConnectionError

from app.agent.audience_provider import (
    AudienceRevisionIssue,
    AudienceRevisionRequest,
)
from app.agent.openai_audience_provider import (
    AudienceProviderError,
    DEFAULT_OPENAI_AUDIENCE_MODEL,
    MAX_OUTPUT_TOKENS,
    OPENAI_MAX_RETRIES,
    OPENAI_REQUEST_TIMEOUT_SECONDS,
    OpenAIAudienceProvider,
)
from app.models.audience_generation import (
    AudienceGenerationResponse,
    CompactClusterContext,
)


def make_context(cluster_id: str) -> CompactClusterContext:
    return CompactClusterContext.model_validate(
        {
            "cluster_id": cluster_id,
            "name": f"Topic {cluster_id}",
            "keywords": ["technology", "innovation"],
            "total_views": 50_000,
            "article_count": 2,
            "topic_confidence": 0.8,
            "articles": [
                {
                    "reference_id": f"{cluster_id}:a0",
                    "title": "First article",
                    "weekly_views": 30_000,
                    "summary": "A concise first article summary for the topic.",
                },
                {
                    "reference_id": f"{cluster_id}:a1",
                    "title": "Second article",
                    "weekly_views": 20_000,
                    "summary": "A concise second article summary for the topic.",
                },
            ],
        }
    )


def make_typed_response(cluster_id: str) -> AudienceGenerationResponse:
    return AudienceGenerationResponse.model_validate(
        {
            "decisions": [
                {
                    "decision": "create_audience",
                    "cluster_id": cluster_id,
                    "name": "Technology Innovation Followers",
                    "description": (
                        "People following practical developments in technology "
                        "and product innovation."
                    ),
                    "supporting_article_reference_ids": [
                        f"{cluster_id}:a0",
                        f"{cluster_id}:a1",
                    ],
                    "buying_power": "medium",
                    "buying_power_reason": (
                        "The subject suggests considered spending on useful "
                        "technology products."
                    ),
                    "brand_categories": ["Consumer technology"],
                    "commercial_confidence": 0.75,
                    "commercial_confidence_reason": (
                        "Two closely related articles provide coherent evidence "
                        "for the audience."
                    ),
                }
            ]
        }
    )


def make_api_response(
    *,
    parsed: AudienceGenerationResponse | None,
    status: str = "completed",
    output: list[object] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id="resp-audience-123",
        model="gpt-5.4-nano-2026-03-17",
        status=status,
        output=(
            [
                SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="output_text")],
                )
            ]
            if output is None
            else output
        ),
        output_parsed=parsed,
        usage=SimpleNamespace(
            input_tokens=321,
            output_tokens=123,
            total_tokens=444,
        ),
    )


class OpenAIAudienceProviderConfigurationTests(unittest.TestCase):
    @patch("app.agent.openai_audience_provider.AsyncOpenAI")
    def test_environment_configures_key_default_model_timeout_and_retries(
        self,
        async_openai: Mock,
    ) -> None:
        provider = OpenAIAudienceProvider.from_environment(
            {"OPENAI_API_KEY": "server-secret"}
        )

        async_openai.assert_called_once_with(
            api_key="server-secret",
            timeout=OPENAI_REQUEST_TIMEOUT_SECONDS,
            max_retries=OPENAI_MAX_RETRIES,
        )
        self.assertEqual(provider._model, DEFAULT_OPENAI_AUDIENCE_MODEL)
        self.assertTrue(provider._owns_client)

    @patch("app.agent.openai_audience_provider.AsyncOpenAI")
    def test_environment_uses_configured_model(self, async_openai: Mock) -> None:
        provider = OpenAIAudienceProvider.from_environment(
            {
                "OPENAI_API_KEY": "server-secret",
                "OPENAI_AUDIENCE_MODEL": "custom-small-model",
            }
        )

        self.assertEqual(provider._model, "custom-small-model")

    def test_missing_api_key_raises_safe_provider_error(self) -> None:
        with self.assertRaisesRegex(
            AudienceProviderError,
            "not configured",
        ):
            OpenAIAudienceProvider.from_environment({})


class OpenAIAudienceProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_aclose_preserves_caller_owned_injected_client(self) -> None:
        client = Mock()
        client.close = AsyncMock()
        provider = OpenAIAudienceProvider(client)

        await provider.aclose()
        await provider.aclose()

        client.close.assert_not_awaited()

    async def test_aclose_closes_explicitly_owned_client_once(self) -> None:
        client = Mock()
        client.close = AsyncMock()
        provider = OpenAIAudienceProvider(client, owns_client=True)

        await provider.aclose()
        await provider.aclose()

        client.close.assert_awaited_once_with()

    async def test_sends_all_contexts_once_and_returns_typed_metadata(
        self,
    ) -> None:
        contexts = [make_context("cluster-one"), make_context("cluster-two")]
        parsed = make_typed_response("cluster-one")
        api_response = make_api_response(parsed=parsed)
        client = Mock()
        client.responses.parse = AsyncMock(return_value=api_response)
        provider = OpenAIAudienceProvider(client, model="configured-model")

        with patch(
            "app.agent.openai_audience_provider.perf_counter",
            side_effect=[10.0, 10.25],
        ):
            result = await provider.generate(contexts)

        client.responses.parse.assert_awaited_once()
        request = client.responses.parse.await_args.kwargs
        self.assertEqual(request["model"], "configured-model")
        self.assertIs(request["text_format"], AudienceGenerationResponse)
        self.assertEqual(request["reasoning"], {"effort": "none"})
        self.assertIs(request["store"], False)
        self.assertEqual(request["timeout"], OPENAI_REQUEST_TIMEOUT_SECONDS)
        self.assertEqual(request["max_output_tokens"], MAX_OUTPUT_TOKENS)

        user_content = request["input"][1]["content"]
        self.assertIn('"cluster_id":"cluster-one"', user_content)
        self.assertIn('"cluster_id":"cluster-two"', user_content)
        self.assertNotIn("size_index", user_content)

        self.assertIs(result.response, parsed)
        self.assertEqual(result.model, "gpt-5.4-nano-2026-03-17")
        self.assertEqual(result.response_id, "resp-audience-123")
        self.assertEqual(result.elapsed_seconds, 0.25)
        self.assertEqual(result.usage.input_tokens, 321)
        self.assertEqual(result.usage.output_tokens, 123)
        self.assertEqual(result.usage.total_tokens, 444)

    async def test_sends_only_requested_revision_context_and_exact_issues(
        self,
    ) -> None:
        parsed = make_typed_response("cluster-one")
        api_response = make_api_response(parsed=parsed)
        client = Mock()
        client.responses.parse = AsyncMock(return_value=api_response)
        provider = OpenAIAudienceProvider(client, model="configured-model")
        revision_request = AudienceRevisionRequest(
            context=make_context("cluster-one"),
            previous_decisions=tuple(parsed.decisions),
            validation_issues=(
                AudienceRevisionIssue(
                    code="cross_cluster_supporting_reference",
                    reference_id="cluster-two:a0",
                ),
                AudienceRevisionIssue(
                    code="unknown_supporting_reference",
                    reference_id="cluster-one:a9",
                ),
            ),
        )

        result = await provider.revise([revision_request])

        client.responses.parse.assert_awaited_once()
        request = client.responses.parse.await_args.kwargs
        self.assertEqual(request["model"], "configured-model")
        self.assertIs(request["text_format"], AudienceGenerationResponse)
        self.assertEqual(request["reasoning"], {"effort": "none"})
        self.assertIs(request["store"], False)
        self.assertEqual(request["timeout"], OPENAI_REQUEST_TIMEOUT_SECONDS)
        self.assertEqual(request["max_output_tokens"], MAX_OUTPUT_TOKENS)

        user_content = request["input"][1]["content"]
        revision_payload = json.loads(user_content.split("\n", 1)[1])
        self.assertEqual(len(revision_payload), 1)
        self.assertEqual(
            set(revision_payload[0]),
            {"cluster_context", "previous_decisions", "validation_issues"},
        )
        self.assertEqual(
            revision_payload[0]["cluster_context"]["cluster_id"],
            "cluster-one",
        )
        self.assertEqual(len(revision_payload[0]["previous_decisions"]), 1)
        self.assertEqual(
            revision_payload[0]["validation_issues"],
            [
                {
                    "code": "cross_cluster_supporting_reference",
                    "reference_id": "cluster-two:a0",
                },
                {
                    "code": "unknown_supporting_reference",
                    "reference_id": "cluster-one:a9",
                },
            ],
        )
        self.assertNotIn('"cluster_id":"cluster-two"', user_content)
        self.assertNotIn("size_index", user_content)
        self.assertIs(result.response, parsed)

    async def test_empty_revision_does_not_call_api(self) -> None:
        client = Mock()
        client.responses.parse = AsyncMock()
        provider = OpenAIAudienceProvider(client)

        with self.assertRaisesRegex(
            AudienceProviderError,
            "requires at least one cluster",
        ):
            await provider.revise([])

        client.responses.parse.assert_not_awaited()

    async def test_provider_request_failure_is_safely_translated(self) -> None:
        client = Mock()
        source_error = APIConnectionError(
            request=httpx.Request(
                "POST",
                "https://api.openai.com/v1/responses",
            )
        )
        client.responses.parse = AsyncMock(side_effect=source_error)
        provider = OpenAIAudienceProvider(client)

        with self.assertRaisesRegex(
            AudienceProviderError,
            "request failed",
        ) as raised:
            await provider.generate([make_context("cluster-one")])

        self.assertIs(raised.exception.__cause__, source_error)
        self.assertNotIn("api.openai.com", str(raised.exception))

    async def test_refusal_incomplete_and_missing_parsed_output_are_isolated(
        self,
    ) -> None:
        parsed = make_typed_response("cluster-one")
        refusal_output = [
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="refusal", refusal="No")],
            )
        ]
        failure_cases = [
            (
                make_api_response(parsed=parsed, output=refusal_output),
                "was refused",
            ),
            (
                make_api_response(parsed=parsed, status="incomplete"),
                "incomplete response",
            ),
            (
                make_api_response(parsed=None),
                "no parsed output",
            ),
        ]

        for api_response, expected_message in failure_cases:
            with self.subTest(expected_message=expected_message):
                client = Mock()
                client.responses.parse = AsyncMock(return_value=api_response)
                provider = OpenAIAudienceProvider(client)

                with self.assertRaisesRegex(
                    AudienceProviderError,
                    expected_message,
                ):
                    await provider.generate([make_context("cluster-one")])


if __name__ == "__main__":
    unittest.main()
