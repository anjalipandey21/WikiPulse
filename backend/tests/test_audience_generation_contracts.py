"""Focused tests for internal audience-generation contracts."""

from collections.abc import Sequence
import unittest

from pydantic import ValidationError

from app.agent.audience_provider import (
    AudienceGenerationProvider,
    AudienceProviderResult,
    AudienceTokenUsage,
)
from app.models.audience_generation import (
    AudienceGenerationResponse,
    CompactArticleContext,
    CompactClusterContext,
    CreateAudienceDecision,
    SkipClusterDecision,
)


def article_context_data(reference_id: str) -> dict[str, object]:
    return {
        "reference_id": reference_id,
        "title": f"Article {reference_id}",
        "weekly_views": 1_000,
        "summary": "A concise source summary for commercial interpretation.",
    }


def cluster_context_data() -> dict[str, object]:
    return {
        "cluster_id": "candidate-world-cup",
        "name": "World Cup",
        "keywords": ["world cup", "football"],
        "total_views": 2_500_000,
        "article_count": 3,
        "topic_confidence": 0.82,
        "articles": [
            article_context_data("candidate-world-cup:a0"),
            article_context_data("candidate-world-cup:a1"),
            article_context_data("candidate-world-cup:a2"),
        ],
    }


def create_decision_data() -> dict[str, object]:
    return {
        "decision": "create_audience",
        "cluster_id": "candidate-world-cup",
        "name": "Global Football Followers",
        "description": (
            "Fans actively following international football tournaments and teams."
        ),
        "supporting_article_reference_ids": [
            "candidate-world-cup:a0",
            "candidate-world-cup:a1",
        ],
        "buying_power": "medium",
        "buying_power_reason": (
            "The audience spans broad income groups with repeat fan spending."
        ),
        "brand_categories": ["Sports apparel", "Streaming services"],
        "commercial_confidence": 0.78,
        "commercial_confidence_reason": (
            "Multiple related articles provide coherent evidence of audience interest."
        ),
    }


def skip_decision_data() -> dict[str, object]:
    return {
        "decision": "skip_cluster",
        "cluster_id": "candidate-ambiguous",
        "reason": (
            "The topic does not support a sufficiently specific commercial audience."
        ),
    }


class FakeAudienceProvider:
    def __init__(self, result: AudienceProviderResult) -> None:
        self.result = result
        self.received_contexts: Sequence[CompactClusterContext] | None = None

    async def generate(
        self,
        cluster_contexts: Sequence[CompactClusterContext],
    ) -> AudienceProviderResult:
        self.received_contexts = cluster_contexts
        return self.result


class AudienceGenerationContractTests(unittest.TestCase):
    def test_accepts_valid_create_and_skip_decisions(self) -> None:
        response = AudienceGenerationResponse.model_validate(
            {
                "decisions": [
                    create_decision_data(),
                    skip_decision_data(),
                ]
            }
        )

        self.assertIsInstance(response.decisions[0], CreateAudienceDecision)
        self.assertIsInstance(response.decisions[1], SkipClusterDecision)
        create_decision = response.decisions[0]
        self.assertEqual(
            create_decision.supporting_article_reference_ids,
            ["candidate-world-cup:a0", "candidate-world-cup:a1"],
        )
        self.assertEqual(create_decision.buying_power, "medium")

    def test_rejects_malformed_provider_outputs(self) -> None:
        malformed_cases = []

        unknown_decision = create_decision_data()
        unknown_decision["decision"] = "maybe"
        malformed_cases.append(unknown_decision)

        coerced_confidence = create_decision_data()
        coerced_confidence["commercial_confidence"] = "0.78"
        malformed_cases.append(coerced_confidence)

        extra_field = create_decision_data()
        extra_field["unsupported_field"] = "not allowed"
        malformed_cases.append(extra_field)

        missing_reason = skip_decision_data()
        del missing_reason["reason"]
        malformed_cases.append(missing_reason)

        for decision in malformed_cases:
            with self.subTest(decision=decision):
                with self.assertRaises(ValidationError):
                    AudienceGenerationResponse.model_validate(
                        {"decisions": [decision]}
                    )

    def test_rejects_duplicate_article_reference_ids(self) -> None:
        duplicate_output = create_decision_data()
        duplicate_output["supporting_article_reference_ids"] = [
            "candidate-world-cup:a0",
            "candidate-world-cup:a0",
        ]
        with self.assertRaisesRegex(ValidationError, "must be unique"):
            AudienceGenerationResponse.model_validate(
                {"decisions": [duplicate_output]}
            )

        duplicate_context = cluster_context_data()
        duplicate_context["articles"] = [
            article_context_data("candidate-world-cup:a0"),
            article_context_data("candidate-world-cup:a0"),
        ]
        with self.assertRaisesRegex(ValidationError, "must be unique"):
            CompactClusterContext.model_validate(duplicate_context)

    def test_enforces_text_list_and_confidence_bounds(self) -> None:
        invalid_decisions: list[dict[str, object]] = []

        short_description = create_decision_data()
        short_description["description"] = "Too short"
        invalid_decisions.append(short_description)

        long_name = create_decision_data()
        long_name["name"] = "A" * 81
        invalid_decisions.append(long_name)

        too_many_categories = create_decision_data()
        too_many_categories["brand_categories"] = [
            f"Category {index}" for index in range(6)
        ]
        invalid_decisions.append(too_many_categories)

        invalid_confidence = create_decision_data()
        invalid_confidence["commercial_confidence"] = 1.01
        invalid_decisions.append(invalid_confidence)

        too_few_references = create_decision_data()
        too_few_references["supporting_article_reference_ids"] = [
            "candidate-world-cup:a0"
        ]
        invalid_decisions.append(too_few_references)

        for decision in invalid_decisions:
            with self.subTest(decision=decision):
                with self.assertRaises(ValidationError):
                    AudienceGenerationResponse.model_validate(
                        {"decisions": [decision]}
                    )

        with self.assertRaises(ValidationError):
            AudienceGenerationResponse.model_validate(
                {"decisions": [skip_decision_data() for _ in range(7)]}
            )

    def test_decision_fields_are_mutually_exclusive(self) -> None:
        create_with_skip_reason = create_decision_data()
        create_with_skip_reason["reason"] = (
            "This skip-only reason must not appear on a create decision."
        )

        skip_with_create_fields = skip_decision_data()
        skip_with_create_fields["name"] = "Unexpected audience"
        skip_with_create_fields["brand_categories"] = ["Unexpected category"]

        for decision in (create_with_skip_reason, skip_with_create_fields):
            with self.subTest(decision=decision):
                with self.assertRaises(ValidationError):
                    AudienceGenerationResponse.model_validate(
                        {"decisions": [decision]}
                    )


class AudienceGenerationProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_provider_protocol_is_injectable_and_returns_typed_response(
        self,
    ) -> None:
        context = CompactClusterContext.model_validate(cluster_context_data())
        response = AudienceGenerationResponse.model_validate(
            {"decisions": [create_decision_data()]}
        )
        expected_result = AudienceProviderResult(
            response=response,
            model="mock-model",
            response_id="response-1",
            elapsed_seconds=0.25,
            usage=AudienceTokenUsage(
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
            ),
        )
        provider: AudienceGenerationProvider = FakeAudienceProvider(
            expected_result
        )

        result = await provider.generate([context])

        self.assertIs(result, expected_result)
        self.assertIs(result.response, response)
        self.assertEqual(provider.received_contexts, [context])


if __name__ == "__main__":
    unittest.main()
