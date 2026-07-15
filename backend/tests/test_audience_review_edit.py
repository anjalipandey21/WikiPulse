"""Focused Phase 1B analyst-edit workflow and recovery tests."""

import asyncio
from datetime import UTC, datetime, timedelta
from json import dumps
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from app.agent.audience_finalization import prepare_audience_clusters
from app.agent.audience_provider import (
    AnalystEditGenerationResponse,
    AnalystEditProviderRequest,
    AnalystEditProviderResult,
    AudienceTokenUsage,
)
from app.agent.audience_review_runtime import AudienceReviewRuntime, _payload_digest
from app.agent.audience_review_workflow import (
    AudienceReviewConflictError,
)
from app.models.audience_generation import CreateAudienceDecision, SkipClusterDecision
from app.models.audience_review import (
    AnalystEditableField,
    EditRecommendationReviewCommand,
    ReviewConflictCode,
    new_command_id,
    parse_review_command,
)
from tests.test_audience_review_runtime import (
    CheckpointBarrierSaver,
    MutableClock,
    assert_saver_is_safe,
)
from tests.test_audience_review_workflow import (
    FakeProvider,
    make_cluster,
    make_create,
    response,
)


def edited_create(
    cluster_id: str,
    *,
    description: str = (
        "People following this coherent topic with a sharper practical focus."
    ),
    references: list[str] | None = None,
) -> CreateAudienceDecision:
    original = make_create(cluster_id)
    return original.model_copy(
        deep=True,
        update={
            "description": description,
            "supporting_article_reference_ids": references
            or original.supporting_article_reference_ids,
        },
    )


def completed_edit(*decisions: object) -> AnalystEditProviderResult:
    return AnalystEditProviderResult(
        status="completed",
        response=AnalystEditGenerationResponse(decisions=list(decisions)),
        elapsed_seconds=0.1,
        usage=AudienceTokenUsage(20, 10, 30),
    )


class EditProvider(FakeProvider):
    def __init__(self, cluster_ids: tuple[str, ...]) -> None:
        super().__init__(response(*(make_create(value) for value in cluster_ids)))
        self.edit_result = completed_edit(edited_create(cluster_ids[0]))
        self.edit_calls = 0
        self.edit_requests: list[AnalystEditProviderRequest] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.block = False

    async def regenerate_from_analyst_edit(
        self,
        request: AnalystEditProviderRequest,
    ) -> AnalystEditProviderResult:
        self.edit_calls += 1
        self.edit_requests.append(request)
        self.started.set()
        if self.block:
            await self.release.wait()
        return self.edit_result


def edit_for(
    result,
    *,
    command_id: str | None = None,
    feedback: str | None = None,
    fields: tuple[AnalystEditableField, ...] = (
        AnalystEditableField.AUDIENCE_POSITIONING,
    ),
):
    pending = result.pending_review
    assert pending is not None
    return EditRecommendationReviewCommand(
        type="edit_recommendation",
        run_id=result.run_id,
        review_id=pending.review_id,
        cluster_id=pending.cluster_id,
        expected_version=pending.version,
        command_id=command_id or new_command_id(),
        feedback=feedback or "Make the audience description more practically focused.",
        fields_to_change=fields,
    )


class AnalystEditContractTests(unittest.TestCase):
    def test_edit_command_normalizes_feedback_and_field_order(self) -> None:
        base = {
            "type": "edit_recommendation",
            "run_id": "86f4a1d6-ed02-4f2d-b7cb-d2c4bac8fd0e",
            "review_id": "cc51cd39-7acb-5d4f-a2f4-12638c949afa",
            "cluster_id": "cluster-one",
            "expected_version": 1,
            "command_id": "8a4ceace-b90a-438c-bc71-cbb8ae4081a1",
            "feedback": "  Refine\u00a0the   positioning clearly.  ",
            "fields_to_change": [
                "commercial_confidence",
                "audience_positioning",
            ],
        }
        parsed = parse_review_command(base)

        self.assertEqual(parsed.feedback, "Refine the positioning clearly.")
        self.assertEqual(
            parsed.fields_to_change,
            (
                AnalystEditableField.AUDIENCE_POSITIONING,
                AnalystEditableField.COMMERCIAL_CONFIDENCE,
            ),
        )

    def test_edit_feedback_errors_are_redacted(self) -> None:
        secret = "PRIVATE-EDIT-SENTINEL"
        raw = {
            "type": "edit_recommendation",
            "run_id": "86f4a1d6-ed02-4f2d-b7cb-d2c4bac8fd0e",
            "review_id": "cc51cd39-7acb-5d4f-a2f4-12638c949afa",
            "cluster_id": "cluster-one",
            "expected_version": 1,
            "command_id": "8a4ceace-b90a-438c-bc71-cbb8ae4081a1",
            "feedback": secret + "\u200b",
            "fields_to_change": ["audience_positioning"],
        }
        with self.assertRaises(ValidationError) as raised:
            parse_review_command(raw)

        rendered = (
            str(raised.exception)
            + repr(raised.exception)
            + dumps(raised.exception.errors(), default=str)
            + raised.exception.json()
        )
        self.assertNotIn(secret, rendered)

        public_field_failure = dict(raw)
        public_field_failure["feedback"] = (
            secret + " remains valid private analyst guidance."
        )
        public_field_failure["run_id"] = "not-a-uuid"
        with self.assertRaises(ValidationError) as public_raised:
            parse_review_command(public_field_failure)
        public_rendered = (
            str(public_raised.exception)
            + repr(public_raised.exception)
            + dumps(public_raised.exception.errors(), default=str)
            + public_raised.exception.json()
        )
        self.assertNotIn(secret, public_rendered)

    def test_feedback_is_redacted_for_every_discriminator_failure(self) -> None:
        secret = "PRIVATE-DISCRIMINATOR-FEEDBACK-SENTINEL"
        base = {
            "run_id": "86f4a1d6-ed02-4f2d-b7cb-d2c4bac8fd0e",
            "review_id": "cc51cd39-7acb-5d4f-a2f4-12638c949afa",
            "cluster_id": "cluster-one",
            "expected_version": 1,
            "command_id": "8a4ceace-b90a-438c-bc71-cbb8ae4081a1",
            "feedback": (
                secret + " with otherwise valid private analyst guidance."
            ),
            "fields_to_change": ["audience_positioning"],
        }
        cases = {
            "missing type": dict(base),
            "unknown type": {**base, "type": "unknown_edit_type"},
            "non-string type": {**base, "type": 7},
            "malformed UUID": {
                **base,
                "type": "edit_recommendation",
                "run_id": "not-a-uuid",
            },
            "extra field": {
                **base,
                "type": "edit_recommendation",
                "unexpected": "not-allowed",
            },
        }

        for label, raw in cases.items():
            with self.subTest(label=label), self.assertRaises(
                ValidationError
            ) as raised:
                parse_review_command(raw)
            rendered = (
                str(raised.exception)
                + repr(raised.exception)
                + dumps(raised.exception.errors(), default=str)
                + raised.exception.json()
            )
            self.assertNotIn(secret, rendered)

    def test_duplicate_and_unknown_groups_are_rejected(self) -> None:
        base = {
            "type": "edit_recommendation",
            "run_id": "86f4a1d6-ed02-4f2d-b7cb-d2c4bac8fd0e",
            "review_id": "cc51cd39-7acb-5d4f-a2f4-12638c949afa",
            "cluster_id": "cluster-one",
            "expected_version": 1,
            "command_id": "8a4ceace-b90a-438c-bc71-cbb8ae4081a1",
            "feedback": "Refine this recommendation clearly.",
        }
        for groups in (
            ["audience_positioning", "audience_positioning"],
            ["not_allowed"],
        ):
            with self.subTest(groups=groups), self.assertRaises(ValidationError):
                parse_review_command({**base, "fields_to_change": groups})

    def test_equivalent_raw_and_model_commands_have_stable_digest(self) -> None:
        raw = {
            "type": "edit_recommendation",
            "run_id": "86f4a1d6-ed02-4f2d-b7cb-d2c4bac8fd0e",
            "review_id": "cc51cd39-7acb-5d4f-a2f4-12638c949afa",
            "cluster_id": "cluster-one",
            "expected_version": 1,
            "command_id": "8a4ceace-b90a-438c-bc71-cbb8ae4081a1",
            "feedback": "  Refine the positioning with useful clarity.  ",
            "fields_to_change": [
                "commercial_confidence",
                "audience_positioning",
            ],
        }
        parsed = parse_review_command(raw)
        reparsed = parse_review_command(parsed)

        self.assertEqual(
            _payload_digest(parsed.model_dump(mode="json")),
            _payload_digest(reparsed.model_dump(mode="json")),
        )


class AnalystEditRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.clock = MutableClock()
        self.runtime = AudienceReviewRuntime(
            clock=self.clock,
            default_ttl=timedelta(hours=1),
        )

    async def _start(self, *cluster_ids: str):
        provider = EditProvider(tuple(cluster_ids))
        preparation = prepare_audience_clusters(
            [make_cluster(value) for value in cluster_ids],
            total_analyzed_views=max(1_000, len(cluster_ids) * 300),
        )
        result = await self.runtime.start(preparation, provider)
        return result, provider

    async def test_successful_edit_publishes_once_and_replay_is_idempotent(self) -> None:
        started, provider = await self._start("first")
        command = edit_for(started)

        receipt = await self.runtime.submit_command(command)
        replay = await self.runtime.submit_command(command)
        result = await self.runtime.get_run(started.run_id)

        self.assertEqual(receipt.resulting_status, "published")
        self.assertTrue(replay.idempotent_replay)
        self.assertEqual(provider.edit_calls, 1)
        self.assertEqual(result.status, "completed")
        candidate = result.review_candidates[0]
        self.assertTrue(candidate.edit_attempted)
        self.assertEqual(candidate.version, 2)
        self.assertEqual(candidate.status, "published")
        self.assertIsNotNone(candidate.edited_recommendation)
        self.assertEqual(
            [event.code for event in candidate.trace.events[-5:]],
            [
                "review_requested",
                "analyst_edit_requested",
                "edited_decision_received",
                "edited_decision_validated",
                "edited_audience_published",
            ],
        )
        self.assertNotIn(command.feedback, result.model_dump_json())
        self.assertNotIn(command.feedback, receipt.model_dump_json())

    async def test_get_run_reports_editing_without_holding_run_lock(self) -> None:
        started, provider = await self._start("first")
        provider.block = True
        command = edit_for(started)
        submission = asyncio.create_task(self.runtime.submit_command(command))
        await provider.started.wait()

        editing = await self.runtime.get_run(started.run_id)

        self.assertEqual(editing.status, "editing")
        self.assertIsNone(editing.pending_review)
        self.assertIn(command.command_id, self.runtime._edit_operations)
        provider.release.set()
        receipt = await submission
        self.assertEqual(receipt.resulting_status, "published")
        self.assertNotIn(command.command_id, self.runtime._edit_operations)

    async def test_cancellation_during_provider_call_recovers_without_recall(self) -> None:
        started, provider = await self._start("first")
        provider.block = True
        command = edit_for(started)
        submission = asyncio.create_task(self.runtime.submit_command(command))
        await provider.started.wait()
        submission.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await submission
        provider.release.set()
        await asyncio.shield(self.runtime._edit_operations[command.command_id].provider_task)

        receipt = await self.runtime.submit_command(command)
        result = await self.runtime.get_run(started.run_id)

        self.assertEqual(receipt.resulting_status, "published")
        self.assertEqual(provider.edit_calls, 1)
        self.assertEqual(result.status, "completed")

    async def test_receipt_gap_reconstructs_without_duplicate_edit_events(self) -> None:
        cancellation = asyncio.CancelledError()

        async def cancel_after_terminal_checkpoint() -> None:
            raise cancellation

        runtime = AudienceReviewRuntime(
            clock=self.clock,
            _after_command_resume_hook=cancel_after_terminal_checkpoint,
        )
        provider = EditProvider(("first",))
        preparation = prepare_audience_clusters(
            [make_cluster("first")], total_analyzed_views=1_000
        )
        started = await runtime.start(preparation, provider)
        command = edit_for(started)
        with self.assertRaises(asyncio.CancelledError):
            await runtime.submit_command(command)
        runtime._after_command_resume_hook = None

        receipt = await runtime.submit_command(command)
        result = await runtime.get_run(started.run_id)

        self.assertTrue(receipt.idempotent_replay)
        self.assertEqual(provider.edit_calls, 1)
        event_codes = [event.code for event in result.review_candidates[0].trace.events]
        for code in (
            "analyst_edit_requested",
            "edited_decision_received",
            "edited_decision_validated",
            "edited_audience_published",
        ):
            self.assertEqual(event_codes.count(code), 1)

    async def test_terminal_checkpoint_replay_cleans_edit_operation(self) -> None:
        saver = CheckpointBarrierSaver("finalize_review_run")
        runtime = AudienceReviewRuntime(clock=self.clock)
        runtime._configure_checkpointer(saver)
        provider = EditProvider(("first",))
        preparation = prepare_audience_clusters(
            [make_cluster("first")], total_analyzed_views=1_000
        )
        started = await runtime.start(preparation, provider)
        command = edit_for(started)

        with self.assertRaises(asyncio.CancelledError):
            await runtime.submit_command(command)
        partial = await runtime._graph.aget_state(
            {"configurable": {"thread_id": started.thread_id}}
        )
        self.assertEqual(partial.next, ())
        self.assertTrue(partial.values["completed"])
        self.assertIn(command.command_id, runtime._edit_operations)
        self.assertIsNone(
            runtime._runs[started.run_id].receipts[command.command_id].receipt
        )

        receipt = await runtime.submit_command(command)
        replay = await runtime.submit_command(command)

        self.assertTrue(receipt.idempotent_replay)
        self.assertTrue(replay.idempotent_replay)
        self.assertEqual(provider.edit_calls, 1)
        self.assertNotIn(command.command_id, runtime._edit_operations)

    async def test_terminal_checkpoint_lookup_cleans_edit_operation(self) -> None:
        saver = CheckpointBarrierSaver("finalize_review_run")
        runtime = AudienceReviewRuntime(clock=self.clock)
        runtime._configure_checkpointer(saver)
        provider = EditProvider(("first",))
        preparation = prepare_audience_clusters(
            [make_cluster("first")], total_analyzed_views=1_000
        )
        started = await runtime.start(preparation, provider)
        command = edit_for(started)

        with self.assertRaises(asyncio.CancelledError):
            await runtime.submit_command(command)
        self.assertIn(command.command_id, runtime._edit_operations)

        result = await runtime.get_run(started.run_id)
        reservation = runtime._runs[started.run_id].receipts[command.command_id]

        self.assertEqual(result.status, "completed")
        self.assertIsNotNone(reservation.receipt)
        self.assertEqual(provider.edit_calls, 1)
        self.assertNotIn(command.command_id, runtime._edit_operations)

    async def test_cleanup_removes_only_matching_terminal_operation(self) -> None:
        first_provider = EditProvider(("first",))
        second_provider = EditProvider(("second",))
        first_provider.block = True
        second_provider.block = True
        first_started = await self.runtime.start(
            prepare_audience_clusters(
                [make_cluster("first")], total_analyzed_views=1_000
            ),
            first_provider,
        )
        second_started = await self.runtime.start(
            prepare_audience_clusters(
                [make_cluster("second")], total_analyzed_views=1_000
            ),
            second_provider,
        )
        first_command = edit_for(first_started)
        second_command = edit_for(second_started)
        first_submission = asyncio.create_task(
            self.runtime.submit_command(first_command)
        )
        second_submission = asyncio.create_task(
            self.runtime.submit_command(second_command)
        )
        await first_provider.started.wait()
        await second_provider.started.wait()

        first_provider.release.set()
        await first_submission

        self.assertNotIn(first_command.command_id, self.runtime._edit_operations)
        self.assertIn(second_command.command_id, self.runtime._edit_operations)
        self.assertEqual(first_provider.edit_calls, 1)
        self.assertEqual(second_provider.edit_calls, 1)

        second_provider.release.set()
        await second_submission
        self.assertNotIn(second_command.command_id, self.runtime._edit_operations)

    async def test_runtime_discriminator_errors_do_not_expose_feedback(self) -> None:
        secret = "PRIVATE-RUNTIME-FEEDBACK-SENTINEL"
        base = {
            "run_id": "86f4a1d6-ed02-4f2d-b7cb-d2c4bac8fd0e",
            "review_id": "cc51cd39-7acb-5d4f-a2f4-12638c949afa",
            "cluster_id": "cluster-one",
            "expected_version": 1,
            "command_id": "8a4ceace-b90a-438c-bc71-cbb8ae4081a1",
            "feedback": secret + " with valid private analyst guidance.",
            "fields_to_change": ["audience_positioning"],
        }
        for raw in (
            dict(base),
            {**base, "type": "unknown_edit_type"},
            {**base, "type": 7},
            {**base, "type": "edit_recommendation", "run_id": "bad"},
            {**base, "type": "edit_recommendation", "extra": True},
        ):
            with self.subTest(command_type=raw.get("type")), self.assertRaises(
                (ValidationError, TypeError)
            ) as raised:
                await self.runtime.submit_command(raw)
            error = raised.exception
            rendered = str(error) + repr(error)
            if isinstance(error, ValidationError):
                rendered += dumps(error.errors(), default=str) + error.json()
            self.assertNotIn(secret, rendered)

    async def test_cancellation_after_edit_intent_checkpoint_recovers(self) -> None:
        saver = CheckpointBarrierSaver("apply_edit_command")
        self.runtime._configure_checkpointer(saver)
        started, provider = await self._start("first")
        command = edit_for(started)

        with self.assertRaises(asyncio.CancelledError):
            await self.runtime.submit_command(command)
        partial = await self.runtime._graph.aget_state(
            {"configurable": {"thread_id": started.thread_id}}
        )
        self.assertEqual(partial.next, ("regenerate_and_finalize_edit",))
        self.assertEqual(partial.values["records"][0]["status"], "editing")
        self.assertEqual(provider.edit_calls, 0)

        receipt = await self.runtime.submit_command(command)
        self.assertEqual(receipt.resulting_status, "published")
        self.assertEqual(provider.edit_calls, 1)

    async def test_cancellation_during_queue_advancement_recovers(self) -> None:
        saver = CheckpointBarrierSaver("mark_review_pending")
        self.runtime._configure_checkpointer(saver)
        started, provider = await self._start("first", "second")
        command = edit_for(started)

        with self.assertRaises(asyncio.CancelledError):
            await self.runtime.submit_command(command)
        partial = await self.runtime._graph.aget_state(
            {"configurable": {"thread_id": started.thread_id}}
        )
        self.assertEqual(
            [item["status"] for item in partial.values["records"]],
            ["published", "pending_review"],
        )

        receipt = await self.runtime.submit_command(command)
        result = await self.runtime.get_run(started.run_id)
        self.assertTrue(receipt.idempotent_replay)
        self.assertEqual(provider.edit_calls, 1)
        self.assertEqual(result.pending_review.cluster_id, "second")
        first_events = [
            event.code for event in result.review_candidates[0].trace.events
        ]
        self.assertEqual(first_events.count("edited_audience_published"), 1)

    async def test_automatic_revision_valid_candidate_can_still_be_edited(self) -> None:
        provider = EditProvider(("first",))
        provider.initial = response(
            make_create("first", references=["first:a0", "missing:a9"])
        )
        provider.revision = response(make_create("first"))
        preparation = prepare_audience_clusters(
            [make_cluster("first")], total_analyzed_views=1_000
        )
        started = await self.runtime.start(preparation, provider)

        receipt = await self.runtime.submit_command(edit_for(started))
        result = await self.runtime.get_run(started.run_id)

        self.assertEqual(provider.revise_calls, 1)
        self.assertEqual(provider.edit_calls, 1)
        self.assertEqual(receipt.resulting_status, "published")
        self.assertEqual(result.metrics.revision_count, 1)
        self.assertTrue(result.review_candidates[0].edit_attempted)

    async def test_checkpoint_contains_no_provider_metadata_or_public_feedback(self) -> None:
        started, provider = await self._start("first")
        command = edit_for(
            started,
            feedback="Use a focused but private analyst positioning direction.",
        )
        await self.runtime.submit_command(command)
        result = await self.runtime.get_run(started.run_id)

        await assert_saver_is_safe(
            self,
            self.runtime,
            result.thread_id,
            expected_interrupt_count=1,
            forbidden_values=(
                "must-not-be-checkpointed",
                "PRIVATE-PROVIDER-SENTINEL",
            ),
        )
        self.assertNotIn(command.feedback, result.model_dump_json())
        latest = await self.runtime._graph.aget_state(
            {"configurable": {"thread_id": result.thread_id}}
        )
        self.assertIsNone(latest.values["active_edit"])
        self.assertEqual(latest.values["pending_command"], {})

    async def test_drop_codes_and_truthful_traces(self) -> None:
        cases = {
            "edit_provider_failed": AnalystEditProviderResult(
                status="provider_failed", response=None, elapsed_seconds=0, usage=None
            ),
            "edit_provider_refused": AnalystEditProviderResult(
                status="refused", response=None, elapsed_seconds=0, usage=None
            ),
            "edit_provider_missing_output": AnalystEditProviderResult(
                status="missing_output", response=None, elapsed_seconds=0, usage=None
            ),
            "edit_zero_decisions": completed_edit(),
            "edit_multiple_decisions": completed_edit(
                edited_create("first"), edited_create("first")
            ),
            "edit_wrong_cluster": completed_edit(edited_create("other")),
            "edit_provider_skip_not_allowed": completed_edit(
                SkipClusterDecision(
                    decision="skip_cluster",
                    cluster_id="first",
                    reason="The requested edit cannot support a safe audience.",
                )
            ),
            "edit_unsupported_references": completed_edit(
                edited_create("first", references=["first:a0", "unknown:a9"])
            ),
            "edit_intent_conformance_failed": completed_edit(make_create("first")),
        }
        for expected_code, edit_result in cases.items():
            with self.subTest(expected_code=expected_code):
                runtime = AudienceReviewRuntime(clock=MutableClock())
                provider = EditProvider(("first",))
                provider.edit_result = edit_result
                preparation = prepare_audience_clusters(
                    [make_cluster("first")], total_analyzed_views=1_000
                )
                started = await runtime.start(preparation, provider)

                receipt = await runtime.submit_command(edit_for(started))
                result = await runtime.get_run(started.run_id)

                self.assertEqual(
                    receipt.resulting_status, "edit_validation_dropped"
                )
                self.assertEqual(
                    result.edit_validation_drops[0].drop_code,
                    expected_code,
                )
                events = [event.code for event in result.review_candidates[0].trace.events]
                self.assertEqual(events.count("analyst_edit_dropped"), 1)
                self.assertEqual(
                    "edited_decision_received" in events,
                    expected_code
                    not in {
                        "edit_provider_failed",
                        "edit_provider_refused",
                        "edit_provider_missing_output",
                        "edit_zero_decisions",
                    },
                )

    async def test_validation_and_internal_failures_use_stable_drop_codes(self) -> None:
        validation_report = SimpleNamespace(
            valid_segments=(),
            provider_skips=(),
            invalid_decisions=(
                SimpleNamespace(
                    issues=(SimpleNamespace(code="synthetic_validation_issue"),)
                ),
            ),
        )
        cases = (
            (
                "edit_validation_failed",
                patch(
                    "app.agent.audience_review_workflow.finalize_audience_decisions",
                    return_value=validation_report,
                ),
            ),
            (
                "edit_internal_failure",
                patch(
                    "app.agent.audience_review_workflow._restore_preparation",
                    side_effect=RuntimeError("PRIVATE-INTERNAL-SENTINEL"),
                ),
            ),
        )
        for expected_code, boundary_patch in cases:
            with self.subTest(expected_code=expected_code):
                runtime = AudienceReviewRuntime(clock=MutableClock())
                provider = EditProvider(("first",))
                preparation = prepare_audience_clusters(
                    [make_cluster("first")], total_analyzed_views=1_000
                )
                started = await runtime.start(preparation, provider)
                with boundary_patch:
                    receipt = await runtime.submit_command(edit_for(started))
                result = await runtime.get_run(started.run_id)

                self.assertEqual(
                    receipt.resulting_status, "edit_validation_dropped"
                )
                self.assertEqual(
                    result.edit_validation_drops[0].drop_code,
                    expected_code,
                )
                self.assertNotIn(
                    "PRIVATE-INTERNAL-SENTINEL",
                    result.model_dump_json(),
                )

    async def test_unselected_and_order_only_changes_fail_conformance(self) -> None:
        original = make_create("first")
        cases = (
            (
                original.model_copy(
                    deep=True,
                    update={
                        "description": (
                            "People following this topic with a practical focus."
                        ),
                        "buying_power": "high",
                    },
                ),
                (AnalystEditableField.AUDIENCE_POSITIONING,),
            ),
            (
                original.model_copy(
                    deep=True,
                    update={
                        "supporting_article_reference_ids": [
                            "first:a1",
                            "first:a0",
                        ]
                    },
                ),
                (AnalystEditableField.SUPPORTING_EVIDENCE,),
            ),
            (
                original.model_copy(
                    deep=True,
                    update={"brand_categories": ["media"]},
                ),
                (AnalystEditableField.BRAND_CATEGORIES,),
            ),
        )
        for decision, fields in cases:
            with self.subTest(fields=fields):
                runtime = AudienceReviewRuntime(clock=MutableClock())
                provider = EditProvider(("first",))
                provider.edit_result = completed_edit(decision)
                preparation = prepare_audience_clusters(
                    [make_cluster("first")], total_analyzed_views=1_000
                )
                started = await runtime.start(preparation, provider)

                await runtime.submit_command(edit_for(started, fields=fields))
                result = await runtime.get_run(started.run_id)

                self.assertEqual(
                    result.edit_validation_drops[0].drop_code,
                    "edit_intent_conformance_failed",
                )

    async def test_expiry_during_edit_preserves_edit_and_expires_next(self) -> None:
        started, provider = await self._start("first", "second")
        provider.block = True
        command = edit_for(started)
        submission = asyncio.create_task(self.runtime.submit_command(command))
        await provider.started.wait()
        self.clock.value = datetime(2026, 7, 14, 14, tzinfo=UTC)
        editing = await self.runtime.get_run(started.run_id)
        self.assertEqual(editing.status, "editing")

        provider.release.set()
        receipt = await submission
        result = await self.runtime.get_run(started.run_id)

        self.assertEqual(receipt.resulting_status, "published")
        self.assertEqual(result.status, "expired")
        self.assertEqual(
            [candidate.status for candidate in result.review_candidates],
            ["published", "expired"],
        )

    async def test_due_before_edit_expires_without_provider_call(self) -> None:
        started, provider = await self._start("first")
        command = edit_for(started)
        self.clock.value = datetime(2026, 7, 14, 14, tzinfo=UTC)

        with self.assertRaises(AudienceReviewConflictError) as raised:
            await self.runtime.submit_command(command)

        self.assertEqual(raised.exception.code, ReviewConflictCode.RUN_EXPIRED)
        self.assertEqual(provider.edit_calls, 0)
        result = await self.runtime.get_run(started.run_id)
        self.assertEqual(result.status, "expired")

    async def test_changed_payload_conflicts_and_second_edit_is_not_accepted(self) -> None:
        started, provider = await self._start("first")
        command = edit_for(started)
        await self.runtime.submit_command(command)
        secret = "PRIVATE-CHANGED-FEEDBACK-SENTINEL"
        changed = command.model_copy(
            update={"feedback": secret + " with a materially different direction."}
        )

        with self.assertRaises(AudienceReviewConflictError) as raised:
            await self.runtime.submit_command(changed)

        self.assertEqual(raised.exception.code, ReviewConflictCode.COMMAND_ID_REUSED)
        self.assertEqual(provider.edit_calls, 1)
        result = await self.runtime.get_run(started.run_id)
        self.assertNotIn(secret, str(raised.exception) + repr(raised.exception))
        self.assertNotIn(secret, result.model_dump_json())

    async def test_changed_payload_conflicts_while_edit_is_in_flight(self) -> None:
        started, provider = await self._start("first")
        provider.block = True
        command = edit_for(started)
        submission = asyncio.create_task(self.runtime.submit_command(command))
        await provider.started.wait()
        changed = command.model_copy(
            update={"feedback": "Use a different private positioning direction."}
        )

        with self.assertRaises(AudienceReviewConflictError) as raised:
            await self.runtime.submit_command(changed)

        self.assertEqual(raised.exception.code, ReviewConflictCode.COMMAND_ID_REUSED)
        provider.release.set()
        await submission
        self.assertEqual(provider.edit_calls, 1)


if __name__ == "__main__":
    unittest.main()
