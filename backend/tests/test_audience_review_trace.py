"""Truthfulness and privacy tests for analyst-review traces."""

from datetime import UTC, datetime, timedelta
import unittest

from app.agent.audience_finalization import prepare_audience_clusters
from app.agent.audience_review_runtime import AudienceReviewRuntime
from app.models.audience_review import (
    ApproveReviewCommand,
    RejectReasonCode,
    RejectReviewCommand,
    new_command_id,
)
from tests.test_audience_review_workflow import (
    FakeProvider,
    make_cluster,
    make_create,
    response,
)
from tests.test_audience_review_runtime import assert_saver_is_safe


class AudienceReviewTraceTests(unittest.IsolatedAsyncioTestCase):
    async def _start(self, count: int = 1):
        clusters = [make_cluster(f"cluster-{index}") for index in range(count)]
        preparation = prepare_audience_clusters(
            clusters,
            total_analyzed_views=1_000,
        )
        clock_value = [datetime(2026, 7, 14, tzinfo=UTC)]
        runtime = AudienceReviewRuntime(
            clock=lambda: clock_value[0],
            default_ttl=timedelta(hours=1),
        )
        result = await runtime.start(
            preparation,
            FakeProvider(response(*(make_create(cluster.id) for cluster in clusters))),
        )
        return runtime, result, clock_value

    async def test_publish_trace_occurs_only_after_approval(self) -> None:
        runtime, result, _ = await self._start()
        trace = next(trace for trace in result.traces if trace.review_id is not None)
        self.assertEqual(trace.events[-1].code, "review_requested")
        self.assertNotIn("audience_published", [event.code for event in trace.events])

        pending = result.pending_review
        await runtime.submit_command(
            ApproveReviewCommand(
                type="approve",
                run_id=result.run_id,
                review_id=pending.review_id,
                cluster_id=pending.cluster_id,
                expected_version=1,
                command_id=new_command_id(),
            )
        )
        completed = await runtime.get_run(result.run_id)
        codes = [event.code for event in completed.traces[0].events]
        self.assertEqual(codes[-3:], [
            "review_requested",
            "analyst_approved",
            "audience_published",
        ])

    async def test_rejection_trace_exposes_code_but_not_private_note(self) -> None:
        runtime, result, _ = await self._start()
        pending = result.pending_review
        await runtime.submit_command(
            RejectReviewCommand(
                type="reject",
                run_id=result.run_id,
                review_id=pending.review_id,
                cluster_id=pending.cluster_id,
                expected_version=1,
                command_id=new_command_id(),
                reason_code=RejectReasonCode.OTHER,
                private_note="never public",
            )
        )
        completed = await runtime.get_run(result.run_id)
        event = completed.traces[0].events[-1]
        self.assertEqual(event.code, "analyst_rejected")
        self.assertEqual(event.outcome_code, "other")
        self.assertNotIn("never public", completed.model_dump_json())
        await assert_saver_is_safe(
            self,
            runtime,
            result.thread_id,
            expected_interrupt_count=1,
            forbidden_values=("never public",),
        )

    async def test_expiry_traces_current_and_queued_reviews(self) -> None:
        runtime, result, clock_value = await self._start(count=2)
        clock_value[0] += timedelta(hours=2)
        self.assertEqual(await runtime.expire_due_runs(), 1)
        expired = await runtime.get_run(result.run_id)
        review_traces = [trace for trace in expired.traces if trace.review_id]
        self.assertEqual(
            [trace.events[-1].code for trace in review_traces],
            ["review_expired", "review_expired"],
        )
        self.assertEqual(
            [trace.final_outcome for trace in review_traces],
            ["expired", "expired"],
        )
        self.assertEqual(
            [candidate.status for candidate in expired.review_candidates],
            ["expired", "expired"],
        )
        self.assertEqual(
            [
                sum(event.code == "review_requested" for event in trace.events)
                for trace in review_traces
            ],
            [1, 0],
        )

    async def test_review_requested_is_emitted_only_as_each_item_activates(
        self,
    ) -> None:
        runtime, result, _ = await self._start(count=3)
        self.assertEqual(
            [candidate.status for candidate in result.review_candidates],
            ["pending_review", "queued", "queued"],
        )
        self.assertEqual(
            [
                sum(event.code == "review_requested" for event in trace.events)
                for trace in result.traces
            ],
            [1, 0, 0],
        )

        await runtime.submit_command(
            ApproveReviewCommand(
                type="approve",
                run_id=result.run_id,
                review_id=result.pending_review.review_id,
                cluster_id=result.pending_review.cluster_id,
                expected_version=1,
                command_id=new_command_id(),
            )
        )
        second = await runtime.get_run(result.run_id)
        self.assertEqual(
            [candidate.status for candidate in second.review_candidates],
            ["published", "pending_review", "queued"],
        )
        self.assertEqual(
            [
                sum(event.code == "review_requested" for event in trace.events)
                for trace in second.traces
            ],
            [1, 1, 0],
        )


if __name__ == "__main__":
    unittest.main()
