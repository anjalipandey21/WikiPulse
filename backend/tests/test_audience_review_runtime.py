"""Focused lifecycle, concurrency, expiry, and checkpoint privacy tests."""

import asyncio
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from json import dumps
from threading import Event, Lock
import unittest
from unittest.mock import AsyncMock, patch

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Interrupt

from app.agent.audience_finalization import (
    AudiencePreparation,
    PreparedAudienceCluster,
    prepare_audience_clusters,
)
from app.agent.audience_review_runtime import (
    AudienceReviewRuntime,
    AudienceReviewRuntimeError,
)
from app.agent.audience_review_checkpointer import (
    LifecycleFencedCheckpointer,
    ReviewPersistenceFenceRejected,
)
from app.agent.audience_review_workflow import (
    AudienceReviewConflictError,
)
from app.models.audience_review import (
    ApproveReviewCommand,
    PendingAudienceReview,
    RejectReasonCode,
    RejectReviewCommand,
    ReviewConflictCode,
    new_command_id,
    new_run_id,
    review_id_for,
)
from tests.test_audience_review_workflow import (
    FakeProvider,
    make_cluster,
    make_create,
    response,
)


class MutableClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 7, 14, 12, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.value


class FailingProvider:
    def __init__(self, secret: str) -> None:
        self.secret = secret

    async def generate(self, _contexts):
        raise RuntimeError(self.secret)

    async def revise(self, _requests):
        raise AssertionError("revision is not expected")


class FailOnceAfterResume:
    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.call_count = 0

    async def __call__(self) -> None:
        self.call_count += 1
        if self.call_count == 1:
            raise self.error


class CheckpointBarrierSaver(InMemorySaver):
    """Cancel after selected real checkpoints have been committed."""

    def __init__(self, *targets: str) -> None:
        super().__init__()
        self.targets = list(targets)
        self.partial_values: dict[str, object] | None = None
        self.committed: list[tuple[str, dict[str, object]]] = []

    @staticmethod
    def _label(values: dict[str, object]) -> str | None:
        records = values.get("records", ())
        if (
            any(record.get("status") == "editing" for record in records)
            and values.get("last_applied_command") is None
            and "branch:to:regenerate_and_finalize_edit" in values
        ):
            return "apply_edit_command"
        if values.get("last_applied_command") is None or not records:
            return None
        if values.get("completed"):
            return "finalize_review_run"
        if any(
            record.get("status") == "pending_review"
            for record in records
        ):
            return "mark_review_pending"
        if "branch:to:mark_review_pending" in values or (
            "branch:to:finalize_review_run" in values
        ):
            return "apply_analyst_command"
        return None

    def put(self, config, checkpoint, metadata, new_versions):
        saved_config = super().put(
            config,
            checkpoint,
            metadata,
            new_versions,
        )
        values = checkpoint.get("channel_values", {})
        label = self._label(values)
        if self.targets and label == self.targets[0]:
            self.targets.pop(0)
            self.partial_values = dict(values)
            self.committed.append((label, dict(values)))
            raise asyncio.CancelledError
        return saved_config


class CancellationResistantSaverFenceBarrier:
    def __init__(self, target_kind: str | None = None) -> None:
        self.target_kind = target_kind
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.authorization_attempted = asyncio.Event()

    def install(self, runtime: AudienceReviewRuntime) -> None:
        checkpointer = runtime._fenced_checkpointer
        authorize = checkpointer._is_authorized

        def observed_authorization(authority: object) -> bool:
            result = authorize(authority)
            self.authorization_attempted.set()
            return result

        checkpointer._is_authorized = observed_authorization
        checkpointer._before_persist = self

    async def __call__(self, kind: str) -> None:
        if self.target_kind is not None and kind != self.target_kind:
            return
        self.entered.set()
        while not self.release.is_set():
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                continue


class BlockingRecordingSaver(InMemorySaver):
    def __init__(self, *, block_checkpoint: bool = False) -> None:
        super().__init__()
        self.block_checkpoint = block_checkpoint
        self.entered = Event()
        self.release = Event()
        self.calls: list[str] = []

    def put(self, config, checkpoint, metadata, new_versions):
        self.entered.set()
        if self.block_checkpoint:
            self.release.wait()
        self.calls.append("checkpoint")
        return config

    def put_writes(self, config, writes, task_id, task_path=""):
        self.calls.append("pending_writes")


class DeferredRecoveryRuntime(AudienceReviewRuntime):
    """Model a process interruption before best-effort stabilization runs."""

    async def _best_effort_stabilize_and_reconcile(
        self,
        record,
        lease,
    ) -> None:
        snapshot = await self._read_snapshot(record)
        self._assert_mutation_lease(lease)
        record.state = dict(snapshot.values)


def assert_safe_value(
    test: unittest.TestCase,
    value: object,
    *,
    forbidden_values: tuple[str, ...] = (),
    path: str = "root",
    allow_interrupt_key: bool = False,
) -> None:
    test.assertNotIsInstance(value, BaseException, path)
    test.assertNotIsInstance(value, Interrupt, path)
    if value is None or isinstance(value, (bool, int, float)):
        return
    if isinstance(value, str):
        for forbidden in forbidden_values:
            test.assertNotIn(forbidden, value, path)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            assert_safe_value(
                test,
                item,
                forbidden_values=forbidden_values,
                path=f"{path}[{index}]",
                allow_interrupt_key=allow_interrupt_key,
            )
        return
    if isinstance(value, dict):
        forbidden_keys = {
            "__error__",
            "private_note",
            "prompt",
            "response_id",
            "model",
            "client",
            "lock",
            "exception",
        }
        if not allow_interrupt_key:
            forbidden_keys.add("__interrupt__")
        test.assertTrue(forbidden_keys.isdisjoint(value), path)
        for key, item in value.items():
            assert_safe_value(
                test,
                key,
                forbidden_values=forbidden_values,
                path=f"{path}.key",
                allow_interrupt_key=allow_interrupt_key,
            )
            assert_safe_value(
                test,
                item,
                forbidden_values=forbidden_values,
                path=f"{path}.{key}",
                allow_interrupt_key=allow_interrupt_key,
            )
        return
    test.fail(f"unsafe checkpoint type at {path}: {type(value)!r}")


async def assert_saver_is_safe(
    test: unittest.TestCase,
    runtime: AudienceReviewRuntime,
    thread_id: str,
    *,
    expected_interrupt_count: int,
    forbidden_values: tuple[str, ...] = (),
) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    inspection = runtime._inspect_checkpointer(config)
    checkpoints = inspection.checkpoints
    test.assertGreater(len(checkpoints), 0)
    interrupt_count = 0
    for checkpoint_index, item in enumerate(checkpoints):
        assert_safe_value(
            test,
            item.checkpoint,
            forbidden_values=forbidden_values,
            path=f"checkpoint[{checkpoint_index}]",
            allow_interrupt_key=True,
        )
        for write_index, (_, channel, value) in enumerate(item.pending_writes):
            if channel == "__interrupt__":
                test.assertIsInstance(value, list)
                for interrupt in value:
                    test.assertIsInstance(interrupt, Interrupt)
                    pending = PendingAudienceReview.model_validate_json(
                        dumps(interrupt.value)
                    )
                    test.assertEqual(
                        pending.model_dump(mode="json"),
                        interrupt.value,
                    )
                    assert_safe_value(
                        test,
                        interrupt.value,
                        forbidden_values=forbidden_values,
                        path="interrupt.payload",
                    )
                    interrupt_count += 1
                continue
            test.assertNotEqual(channel, "__error__")
            assert_safe_value(
                test,
                value,
                forbidden_values=forbidden_values,
                path=f"pending_write[{write_index}].{channel}",
            )
    for blob_index, decoded in enumerate(inspection.decoded_blobs):
        assert_safe_value(
            test,
            decoded,
            forbidden_values=forbidden_values,
            path=f"blob[{blob_index}]",
        )
    test.assertEqual(interrupt_count, expected_interrupt_count)


def approve_for(result, *, command_id: str | None = None, version: int = 1):
    pending = result.pending_review
    assert pending is not None
    return ApproveReviewCommand(
        type="approve",
        run_id=result.run_id,
        review_id=pending.review_id,
        cluster_id=pending.cluster_id,
        expected_version=version,
        command_id=command_id or new_command_id(),
    )


def reject_for(
    result,
    *,
    command_id: str | None = None,
    reason_code: RejectReasonCode = RejectReasonCode.SAFETY_CONCERN,
):
    pending = result.pending_review
    assert pending is not None
    return RejectReviewCommand(
        type="reject",
        run_id=result.run_id,
        review_id=pending.review_id,
        cluster_id=pending.cluster_id,
        expected_version=pending.version,
        command_id=command_id or new_command_id(),
        reason_code=reason_code,
    )


class AudienceReviewRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.clock = MutableClock()
        self.runtime = AudienceReviewRuntime(
            clock=self.clock,
            default_ttl=timedelta(hours=1),
        )

    async def _start(self, *cluster_ids: str):
        preparation = prepare_audience_clusters(
            [make_cluster(cluster_id) for cluster_id in cluster_ids],
            total_analyzed_views=max(1_000, len(cluster_ids) * 300),
        )
        return await self.runtime.start(
            preparation,
            FakeProvider(response(*(make_create(value) for value in cluster_ids))),
        )

    async def test_explicit_client_run_id_is_preserved_with_runtime_timestamps(
        self,
    ) -> None:
        run_id = new_run_id()
        preparation = prepare_audience_clusters(
            [make_cluster("first")],
            total_analyzed_views=1_000,
        )

        result = await self.runtime.start(
            preparation,
            FakeProvider(response(make_create("first"))),
            run_id=run_id,
        )

        self.assertEqual(result.run_id, run_id)
        self.assertEqual(result.created_at, self.clock.value.isoformat())
        self.assertEqual(
            result.expires_at,
            (self.clock.value + timedelta(hours=1)).isoformat(),
        )

    def _install_checkpoint_barrier(
        self,
        *targets: str,
    ) -> CheckpointBarrierSaver:
        saver = CheckpointBarrierSaver(*targets)
        self.runtime._configure_checkpointer(saver)
        self._installed_saver = saver
        return saver

    def _install_apply_checkpoint_barrier(
        self,
    ) -> CheckpointBarrierSaver:
        return self._install_checkpoint_barrier("apply_analyst_command")

    def _defer_post_cancel_recovery(self) -> None:
        self.runtime = DeferredRecoveryRuntime(
            clock=self.clock,
            default_ttl=timedelta(hours=1),
        )

    async def _cancel_after_apply_checkpoint(
        self,
        command,
        *,
        expected_next: str,
    ):
        saver = self._installed_saver
        self.assertIsInstance(saver, CheckpointBarrierSaver)
        with self.assertRaises(asyncio.CancelledError):
            await self.runtime.submit_command(command)
        self.assertIsNotNone(saver.partial_values)
        self.assertIn(
            f"branch:to:{expected_next}",
            saver.partial_values,
        )
        self.assertIsNotNone(saver.partial_values["last_applied_command"])
        self.assertFalse(
            any(
                record["status"] == "pending_review"
                for record in saver.partial_values["records"]
            )
        )
        return saver.partial_values

    async def _authoritative_snapshot(self, thread_id: str):
        return await self.runtime._graph.aget_state(
            {"configurable": {"thread_id": thread_id}}
        )

    def _assert_scheduled_partial(
        self,
        snapshot,
        *,
        node: str,
        statuses: list[str],
    ) -> None:
        self.assertEqual(snapshot.next, (node,))
        self.assertEqual(len(snapshot.tasks), 1)
        self.assertEqual(snapshot.tasks[0].name, node)
        self.assertIsNone(snapshot.tasks[0].error)
        self.assertEqual(snapshot.tasks[0].interrupts, ())
        self.assertEqual(snapshot.interrupts, ())
        self.assertEqual(
            [record["status"] for record in snapshot.values["records"]],
            statuses,
        )
        self.assertIsNotNone(snapshot.values["last_applied_command"])

    async def test_approve_publishes_original_snapshot_unchanged(self) -> None:
        result = await self._start("one")
        original = result.pending_review.recommendation
        receipt = await self.runtime.submit_command(approve_for(result))
        completed = await self.runtime.get_run(result.run_id)

        self.assertEqual(receipt.resulting_status, "published")
        self.assertEqual(completed.status, "completed")
        self.assertEqual(completed.published_audiences, (original,))
        self.assertEqual(
            completed.published_audiences[0].model_dump(mode="json"),
            original.model_dump(mode="json"),
        )

    async def test_reject_is_distinct_and_private_note_is_not_public(self) -> None:
        result = await self._start("one")
        pending = result.pending_review
        command = RejectReviewCommand(
            type="reject",
            run_id=result.run_id,
            review_id=pending.review_id,
            cluster_id=pending.cluster_id,
            expected_version=1,
            command_id=new_command_id(),
            reason_code=RejectReasonCode.OTHER,
            private_note="private analyst detail",
        )
        await self.runtime.submit_command(command)
        completed = await self.runtime.get_run(result.run_id)
        serialized = completed.model_dump_json()

        self.assertEqual(len(completed.rejected_reviews), 1)
        self.assertEqual(completed.validation_drops, ())
        self.assertNotIn("private analyst detail", serialized)
        self.assertNotIn("private_note", serialized)

    async def test_exact_replay_returns_prior_receipt(self) -> None:
        result = await self._start("one")
        command = approve_for(result)
        first = await self.runtime.submit_command(command)
        replay = await self.runtime.submit_command(command)

        self.assertFalse(first.idempotent_replay)
        self.assertTrue(replay.idempotent_replay)
        self.assertEqual(first.command_id, replay.command_id)

    async def test_approve_recovers_after_commit_before_receipt(self) -> None:
        result = await self._start("one")
        command = approve_for(result)
        hook = FailOnceAfterResume(RuntimeError("post_resume_interruption"))
        self.runtime._after_command_resume_hook = hook

        with self.assertRaisesRegex(RuntimeError, "post_resume_interruption"):
            await self.runtime.submit_command(command)
        reservation = self.runtime._runs[result.run_id].receipts[
            command.command_id
        ]
        self.assertIsNotNone(reservation.receipt)

        replay = await self.runtime.submit_command(command)
        completed = await self.runtime.get_run(result.run_id)
        self.assertTrue(replay.idempotent_replay)
        self.assertEqual(replay.resulting_status, "published")
        self.assertEqual(hook.call_count, 1)
        codes = [event.code for event in completed.traces[0].events]
        self.assertEqual(codes.count("analyst_approved"), 1)
        self.assertEqual(codes.count("audience_published"), 1)
        await assert_saver_is_safe(
            self,
            self.runtime,
            result.thread_id,
            expected_interrupt_count=1,
        )

    async def test_approve_apply_checkpoint_cancellation_reaches_end(self) -> None:
        self._defer_post_cancel_recovery()
        self._install_apply_checkpoint_barrier()
        result = await self._start("one")
        command = approve_for(result)

        partial = await self._cancel_after_apply_checkpoint(
            command,
            expected_next="finalize_review_run",
        )
        self.assertEqual(
            [record["status"] for record in partial["records"]],
            ["published"],
        )
        completed = await self.runtime.get_run(result.run_id)
        repeated = await self.runtime.get_run(result.run_id)
        replay = await self.runtime.submit_command(command)
        snapshot = await self.runtime._graph.aget_state(
            {"configurable": {"thread_id": result.thread_id}}
        )

        self.assertEqual(completed.status, "completed")
        self.assertEqual(completed, repeated)
        self.assertEqual(snapshot.next, ())
        self.assertTrue(replay.idempotent_replay)
        self.assertEqual(replay.resulting_status, "published")
        codes = [event.code for event in completed.traces[0].events]
        self.assertEqual(codes.count("analyst_approved"), 1)
        self.assertEqual(codes.count("audience_published"), 1)

    async def test_reject_apply_checkpoint_cancellation_reaches_end(self) -> None:
        self._defer_post_cancel_recovery()
        self._install_apply_checkpoint_barrier()
        result = await self._start("one")
        command = reject_for(result)

        partial = await self._cancel_after_apply_checkpoint(
            command,
            expected_next="finalize_review_run",
        )
        self.assertEqual(partial["records"][0]["status"], "rejected")
        completed = await self.runtime.get_run(result.run_id)
        replay = await self.runtime.submit_command(command)

        self.assertEqual(completed.status, "completed")
        self.assertEqual(len(completed.rejected_reviews), 1)
        self.assertTrue(replay.idempotent_replay)
        self.assertEqual(replay.resulting_status, "rejected")
        codes = [event.code for event in completed.traces[0].events]
        self.assertEqual(codes.count("analyst_rejected"), 1)

    async def test_mark_checkpoint_cancellation_recovers_one_interrupt(
        self,
    ) -> None:
        self._defer_post_cancel_recovery()
        self._install_checkpoint_barrier("mark_review_pending")
        result = await self._start("first", "second")
        command = approve_for(result)

        with self.assertRaises(asyncio.CancelledError):
            await self.runtime.submit_command(command)
        partial = await self._authoritative_snapshot(result.thread_id)
        self._assert_scheduled_partial(
            partial,
            node="await_analyst_command",
            statuses=["published", "pending_review"],
        )
        self.assertEqual(
            partial.values["last_applied_command"]["command_id"],
            command.command_id,
        )

        replay = await self.runtime.submit_command(command)
        recovered = await self.runtime.get_run(result.run_id)
        repeated = await self.runtime.get_run(result.run_id)
        stable = await self._authoritative_snapshot(result.thread_id)

        self.assertTrue(replay.idempotent_replay)
        self.assertEqual(recovered, repeated)
        self.assertEqual(recovered.pending_review.cluster_id, "second")
        self.assertEqual(stable.next, ("await_analyst_command",))
        self.assertEqual(len(stable.tasks), 1)
        self.assertEqual(len(stable.tasks[0].interrupts), 1)
        self.assertEqual(len(stable.interrupts), 1)
        self.assertEqual(
            [candidate.version for candidate in recovered.review_candidates],
            [1, 1],
        )
        self.assertEqual(recovered.metrics, result.metrics)
        for trace in recovered.traces:
            codes = [event.code for event in trace.events]
            self.assertEqual(codes.count("review_requested"), 1)
        first_codes = [event.code for event in recovered.traces[0].events]
        self.assertEqual(first_codes.count("analyst_approved"), 1)
        self.assertEqual(first_codes.count("audience_published"), 1)

    async def test_finalize_checkpoint_cancellation_recovers_stable_end(
        self,
    ) -> None:
        self._defer_post_cancel_recovery()
        self._install_checkpoint_barrier("finalize_review_run")
        result = await self._start("one")
        command = reject_for(result)

        with self.assertRaises(asyncio.CancelledError):
            await self.runtime.submit_command(command)
        partial = await self._authoritative_snapshot(result.thread_id)
        self.assertEqual(partial.next, ())
        self.assertEqual(partial.tasks, ())
        self.assertEqual(partial.interrupts, ())
        self.assertTrue(partial.values["completed"])
        self.assertEqual(partial.values["records"][0]["status"], "rejected")
        self.assertEqual(
            partial.values["last_applied_command"]["command_id"],
            command.command_id,
        )

        replay = await self.runtime.submit_command(command)
        completed = await self.runtime.get_run(result.run_id)
        repeated = await self.runtime.get_run(result.run_id)

        self.assertTrue(replay.idempotent_replay)
        self.assertEqual(completed, repeated)
        self.assertEqual(completed.status, "completed")
        self.assertEqual(
            [candidate.version for candidate in completed.review_candidates],
            [1],
        )
        self.assertEqual(completed.metrics, result.metrics)
        codes = [event.code for event in completed.traces[0].events]
        self.assertEqual(codes.count("analyst_rejected"), 1)

    async def test_cancellation_during_stabilization_remains_recoverable(
        self,
    ) -> None:
        self._defer_post_cancel_recovery()
        saver = self._install_checkpoint_barrier(
            "apply_analyst_command",
            "mark_review_pending",
        )
        result = await self._start("first", "second")
        command = approve_for(result)

        await self._cancel_after_apply_checkpoint(
            command,
            expected_next="mark_review_pending",
        )
        apply_partial = await self._authoritative_snapshot(result.thread_id)
        self._assert_scheduled_partial(
            apply_partial,
            node="mark_review_pending",
            statuses=["published", "queued"],
        )
        with self.assertRaises(asyncio.CancelledError):
            await self.runtime.get_run(result.run_id)
        mark_partial = await self._authoritative_snapshot(result.thread_id)
        self._assert_scheduled_partial(
            mark_partial,
            node="await_analyst_command",
            statuses=["published", "pending_review"],
        )
        self.assertEqual(
            [label for label, _ in saver.committed],
            ["apply_analyst_command", "mark_review_pending"],
        )

        recovered = await self.runtime.get_run(result.run_id)
        repeated = await self.runtime.get_run(result.run_id)
        replay = await self.runtime.submit_command(command)

        self.assertEqual(recovered, repeated)
        self.assertTrue(replay.idempotent_replay)
        self.assertEqual(recovered.pending_review.cluster_id, "second")
        self.assertEqual(recovered.metrics, result.metrics)
        self.assertEqual(
            [candidate.version for candidate in recovered.review_candidates],
            [1, 1],
        )
        first_codes = [event.code for event in recovered.traces[0].events]
        second_codes = [event.code for event in recovered.traces[1].events]
        self.assertEqual(first_codes.count("analyst_approved"), 1)
        self.assertEqual(first_codes.count("audience_published"), 1)
        self.assertEqual(first_codes.count("review_requested"), 1)
        self.assertEqual(second_codes.count("review_requested"), 1)

    async def test_stable_boundaries_require_matching_task_metadata(
        self,
    ) -> None:
        result = await self._start("one")
        record = self.runtime._runs[result.run_id]
        stable = await self._authoritative_snapshot(result.thread_id)
        task = stable.tasks[0]
        changed_payload = dict(stable.interrupts[0].value)
        changed_payload["ordinal"] = 99
        malformed = (
            stable._replace(
                tasks=(task._replace(name="mark_review_pending"),)
            ),
            stable._replace(
                tasks=(task._replace(error=RuntimeError("task failed")),)
            ),
            stable._replace(tasks=(task._replace(interrupts=()),)),
            stable._replace(
                tasks=(
                    task._replace(
                        interrupts=(
                            Interrupt(
                                value=changed_payload,
                                id=stable.interrupts[0].id,
                            ),
                        )
                    ),
                )
            ),
        )
        for snapshot in malformed:
            with self.subTest(task=snapshot.tasks[0]):
                record.state = dict(snapshot.values)
                with self.assertRaisesRegex(
                    AudienceReviewRuntimeError,
                    "review_stabilization_failed",
                ):
                    self.runtime._is_stable_boundary(record, snapshot)

        bad = malformed[0]
        with patch.object(
            self.runtime,
            "_read_snapshot",
            new=AsyncMock(return_value=bad),
        ), patch.object(
            self.runtime._graph,
            "ainvoke",
            new=AsyncMock(),
        ) as continuation:
            lease = self.runtime._acquire_mutation_lease("get_reconcile")
            try:
                with self.assertRaisesRegex(
                    AudienceReviewRuntimeError,
                    "review_stabilization_failed",
                ):
                    await self.runtime._stabilize(record, lease)
            finally:
                self.runtime._release_mutation_lease(lease)
            continuation.assert_not_awaited()

        record.state = dict(stable.values)
        await self.runtime.submit_command(approve_for(result))
        terminal = await self._authoritative_snapshot(result.thread_id)
        record.state = dict(terminal.values)
        malformed_terminal = terminal._replace(tasks=(task,))
        with self.assertRaisesRegex(
            AudienceReviewRuntimeError,
            "review_stabilization_failed",
        ):
            self.runtime._is_stable_boundary(record, malformed_terminal)

    async def test_two_candidate_approve_checkpoint_recovery_activates_next(
        self,
    ) -> None:
        self._defer_post_cancel_recovery()
        self._install_apply_checkpoint_barrier()
        result = await self._start("first", "second")
        command = approve_for(result)

        partial = await self._cancel_after_apply_checkpoint(
            command,
            expected_next="mark_review_pending",
        )
        self.assertEqual(
            [record["status"] for record in partial["records"]],
            ["published", "queued"],
        )
        replay = await self.runtime.submit_command(command)
        current = await self.runtime.get_run(result.run_id)

        self.assertTrue(replay.idempotent_replay)
        self.assertEqual(current.pending_review.cluster_id, "second")
        self.assertEqual(
            [candidate.status for candidate in current.review_candidates],
            ["published", "pending_review"],
        )
        first_codes = [event.code for event in current.traces[0].events]
        self.assertEqual(first_codes.count("analyst_approved"), 1)
        self.assertEqual(first_codes.count("audience_published"), 1)
        await assert_saver_is_safe(
            self,
            self.runtime,
            result.thread_id,
            expected_interrupt_count=2,
        )
        changed = reject_for(result, command_id=command.command_id)
        with self.assertRaises(AudienceReviewConflictError) as raised:
            await self.runtime.submit_command(changed)
        self.assertEqual(
            raised.exception.code,
            ReviewConflictCode.COMMAND_ID_REUSED,
        )

    async def test_two_candidate_reject_checkpoint_lookup_activates_next(
        self,
    ) -> None:
        self._defer_post_cancel_recovery()
        self._install_apply_checkpoint_barrier()
        result = await self._start("first", "second")
        command = reject_for(result)

        partial = await self._cancel_after_apply_checkpoint(
            command,
            expected_next="mark_review_pending",
        )
        self.assertEqual(
            [record["status"] for record in partial["records"]],
            ["rejected", "queued"],
        )
        current = await self.runtime.get_run(result.run_id)
        repeated = await self.runtime.get_run(result.run_id)

        self.assertEqual(current, repeated)
        self.assertEqual(current.pending_review.cluster_id, "second")
        self.assertEqual(
            [candidate.status for candidate in current.review_candidates],
            ["rejected", "pending_review"],
        )
        self.assertEqual(
            [event.code for event in current.traces[1].events].count(
                "review_requested"
            ),
            1,
        )

    async def test_new_command_waits_for_queued_candidate_activation(self) -> None:
        self._defer_post_cancel_recovery()
        self._install_apply_checkpoint_barrier()
        result = await self._start("first", "second")
        await self._cancel_after_apply_checkpoint(
            reject_for(result),
            expected_next="mark_review_pending",
        )
        queued = result.review_candidates[1]
        command = ApproveReviewCommand(
            type="approve",
            run_id=result.run_id,
            review_id=queued.review_id,
            cluster_id=queued.cluster_id,
            expected_version=queued.version,
            command_id=new_command_id(),
        )

        receipt = await self.runtime.submit_command(command)
        completed = await self.runtime.get_run(result.run_id)

        self.assertEqual(receipt.resulting_status, "published")
        self.assertEqual(completed.status, "completed")
        self.assertEqual(
            [candidate.status for candidate in completed.review_candidates],
            ["rejected", "published"],
        )
        second_codes = [event.code for event in completed.traces[1].events]
        self.assertLess(
            second_codes.index("review_requested"),
            second_codes.index("analyst_approved"),
        )

    async def test_changed_payload_conflicts_while_command_reserved(self) -> None:
        result = await self._start("one")
        approve = approve_for(result)
        hook = FailOnceAfterResume(RuntimeError("receipt_not_stored"))
        self.runtime._after_command_resume_hook = hook
        with self.assertRaises(RuntimeError):
            await self.runtime.submit_command(approve)

        pending = result.pending_review
        reject = RejectReviewCommand(
            type="reject",
            run_id=result.run_id,
            review_id=pending.review_id,
            cluster_id=pending.cluster_id,
            expected_version=1,
            command_id=approve.command_id,
            reason_code=RejectReasonCode.SAFETY_CONCERN,
        )
        with self.assertRaises(AudienceReviewConflictError) as raised:
            await self.runtime.submit_command(reject)
        self.assertEqual(raised.exception.code, ReviewConflictCode.COMMAND_ID_REUSED)
        completed = await self.runtime.get_run(result.run_id)
        self.assertEqual(len(completed.published_audiences), 1)
        self.assertEqual(completed.rejected_reviews, ())

    async def test_private_note_participates_in_reserved_command_digest(self) -> None:
        result = await self._start("one")
        pending = result.pending_review
        command_id = new_command_id()
        first = RejectReviewCommand(
            type="reject",
            run_id=result.run_id,
            review_id=pending.review_id,
            cluster_id=pending.cluster_id,
            expected_version=1,
            command_id=command_id,
            reason_code=RejectReasonCode.OTHER,
            private_note="first private note",
        )
        hook = FailOnceAfterResume(RuntimeError("receipt_gap"))
        self.runtime._after_command_resume_hook = hook
        with self.assertRaises(RuntimeError):
            await self.runtime.submit_command(first)

        changed = first.model_copy(update={"private_note": "second private note"})
        with self.assertRaises(AudienceReviewConflictError) as raised:
            await self.runtime.submit_command(changed)
        self.assertEqual(raised.exception.code, ReviewConflictCode.COMMAND_ID_REUSED)

    async def test_multi_candidate_recovery_exposes_authoritative_next_review(
        self,
    ) -> None:
        result = await self._start("first", "second")
        command = approve_for(result)
        hook = FailOnceAfterResume(RuntimeError("receipt_gap"))
        self.runtime._after_command_resume_hook = hook
        with self.assertRaises(RuntimeError):
            await self.runtime.submit_command(command)

        replay = await self.runtime.submit_command(command)
        current = await self.runtime.get_run(result.run_id)
        authoritative = await self.runtime._graph.aget_state(
            {"configurable": {"thread_id": result.thread_id}}
        )
        record = self.runtime._runs[result.run_id]

        self.assertTrue(replay.idempotent_replay)
        self.assertEqual(replay.resulting_status, "published")
        self.assertEqual(current.pending_review.cluster_id, "second")
        self.assertEqual(record.state, dict(authoritative.values))
        self.assertEqual(hook.call_count, 1)

    async def test_cancellation_after_graph_resume_is_recoverable(self) -> None:
        result = await self._start("one")
        command = approve_for(result)
        hook = FailOnceAfterResume(asyncio.CancelledError())
        self.runtime._after_command_resume_hook = hook

        with self.assertRaises(asyncio.CancelledError):
            await self.runtime.submit_command(command)
        replay = await self.runtime.submit_command(command)
        completed = await self.runtime.get_run(result.run_id)

        self.assertTrue(replay.idempotent_replay)
        self.assertEqual(replay.resulting_status, "published")
        self.assertEqual(completed.status, "completed")
        self.assertEqual(hook.call_count, 1)

    async def test_command_id_payload_mismatch_conflicts(self) -> None:
        result = await self._start("one")
        command_id = new_command_id()
        await self.runtime.submit_command(approve_for(result, command_id=command_id))
        changed = approve_for(result, command_id=command_id, version=2)
        with self.assertRaises(AudienceReviewConflictError) as raised:
            await self.runtime.submit_command(changed)
        self.assertEqual(raised.exception.code, ReviewConflictCode.COMMAND_ID_REUSED)

    async def test_identity_version_thread_and_terminal_conflicts(self) -> None:
        result = await self._start("one")
        base = approve_for(result)
        cases = (
            (
                base.model_copy(
                    update={
                        "review_id": review_id_for(result.run_id, "other")
                    }
                ),
                ReviewConflictCode.REVIEW_ID_MISMATCH,
                None,
            ),
            (
                base.model_copy(update={"cluster_id": "different"}),
                ReviewConflictCode.CLUSTER_ID_MISMATCH,
                None,
            ),
            (
                base.model_copy(update={"expected_version": 2}),
                ReviewConflictCode.STALE_VERSION,
                None,
            ),
            (base, ReviewConflictCode.WRONG_THREAD, "wrong-thread"),
        )
        for command, expected, thread_id in cases:
            with self.subTest(expected=expected):
                with self.assertRaises(AudienceReviewConflictError) as raised:
                    await self.runtime.submit_command(command, thread_id=thread_id)
                self.assertEqual(raised.exception.code, expected)

        await self.runtime.submit_command(base)
        terminal = base.model_copy(update={"command_id": new_command_id()})
        with self.assertRaises(AudienceReviewConflictError) as raised:
            await self.runtime.submit_command(terminal)
        self.assertEqual(raised.exception.code, ReviewConflictCode.RUN_TERMINAL)

    async def test_private_note_is_absent_from_runtime_conflict(self) -> None:
        result = await self._start("one")
        pending = result.pending_review
        secret = "SECRET_VALID_PRIVATE_NOTE_c84b"
        command = RejectReviewCommand(
            type="reject",
            run_id=result.run_id,
            review_id=pending.review_id,
            cluster_id=pending.cluster_id,
            expected_version=2,
            command_id=new_command_id(),
            reason_code=RejectReasonCode.OTHER,
            private_note=secret,
        )
        with self.assertRaises(AudienceReviewConflictError) as raised:
            await self.runtime.submit_command(command)
        self.assertEqual(raised.exception.code, ReviewConflictCode.STALE_VERSION)
        self.assertNotIn(secret, str(raised.exception))
        self.assertNotIn(secret, repr(raised.exception))
        current = await self.runtime.get_run(result.run_id)
        self.assertNotIn(secret, current.model_dump_json())
        await assert_saver_is_safe(
            self,
            self.runtime,
            result.thread_id,
            expected_interrupt_count=1,
            forbidden_values=(secret,),
        )

    async def test_cross_run_review_identity_conflicts(self) -> None:
        first = await self._start("first")
        second = await self._start("second")
        command = approve_for(first).model_copy(
            update={
                "run_id": second.run_id,
                "command_id": new_command_id(),
            }
        )
        with self.assertRaises(AudienceReviewConflictError) as raised:
            await self.runtime.submit_command(command)
        self.assertEqual(raised.exception.code, ReviewConflictCode.REVIEW_ID_MISMATCH)

    async def test_concurrent_commands_accept_only_one_transition(self) -> None:
        result = await self._start("one")
        commands = [approve_for(result), approve_for(result)]
        outcomes = await asyncio.gather(
            *(self.runtime.submit_command(command) for command in commands),
            return_exceptions=True,
        )

        self.assertEqual(
            sum(not isinstance(outcome, Exception) for outcome in outcomes),
            1,
        )
        self.assertEqual(
            sum(isinstance(outcome, AudienceReviewConflictError) for outcome in outcomes),
            1,
        )

    async def test_expiry_is_external_and_preserves_prior_publication(self) -> None:
        result = await self._start("first", "second", "third")
        await self.runtime.submit_command(approve_for(result))
        waiting = await self.runtime.get_run(result.run_id)
        self.assertEqual(len(waiting.published_audiences), 1)
        self.assertEqual(
            [item.status for item in waiting.review_candidates],
            ["published", "pending_review", "queued"],
        )

        self.clock.value += timedelta(hours=2)
        record = self.runtime._runs[result.run_id]
        self.assertEqual(record.state["expired"], False)
        self.assertFalse(record.lock.locked())
        self.assertNotIn("task", record.__slots__)

        self.assertEqual(await self.runtime.expire_due_runs(), 1)
        expired = await self.runtime.get_run(result.run_id)
        self.assertEqual(expired.status, "expired")
        self.assertEqual(len(expired.published_audiences), 1)
        self.assertEqual(len(expired.expired_reviews), 2)
        self.assertEqual(
            [item.status for item in expired.review_candidates],
            ["published", "expired", "expired"],
        )

        with self.assertRaises(AudienceReviewConflictError) as raised:
            await self.runtime.submit_command(approve_for(waiting))
        self.assertEqual(raised.exception.code, ReviewConflictCode.RUN_EXPIRED)

    async def test_lookup_and_submission_each_detect_due_expiry(self) -> None:
        lookup_run = await self._start("lookup")
        self.clock.value += timedelta(hours=2)
        self.assertEqual(
            (await self.runtime.get_run(lookup_run.run_id)).status,
            "expired",
        )

        submit_run = await self._start("submit")
        self.clock.value += timedelta(hours=2)
        with self.assertRaises(AudienceReviewConflictError) as raised:
            await self.runtime.submit_command(approve_for(submit_run))
        self.assertEqual(raised.exception.code, ReviewConflictCode.RUN_EXPIRED)

    async def test_due_expiry_runs_before_approve_receipt_replay(self) -> None:
        result = await self._start("first", "second")
        command = approve_for(result)
        original = await self.runtime.submit_command(command)
        self.clock.value += timedelta(hours=2)

        replay = await self.runtime.submit_command(command)
        expired = await self.runtime.get_run(result.run_id)

        self.assertTrue(replay.idempotent_replay)
        self.assertEqual(
            replay.model_copy(update={"idempotent_replay": False}),
            original,
        )
        self.assertEqual(expired.status, "expired")
        self.assertEqual(
            [candidate.status for candidate in expired.review_candidates],
            ["published", "expired"],
        )
        self.assertEqual(len(expired.published_audiences), 1)

    async def test_due_expiry_runs_before_reject_receipt_replay(self) -> None:
        result = await self._start("first", "second", "third")
        command = reject_for(result)
        original = await self.runtime.submit_command(command)
        self.clock.value += timedelta(hours=2)

        replay = await self.runtime.submit_command(command)
        expired = await self.runtime.get_run(result.run_id)

        self.assertTrue(replay.idempotent_replay)
        self.assertEqual(
            replay.model_copy(update={"idempotent_replay": False}),
            original,
        )
        self.assertEqual(
            [candidate.status for candidate in expired.review_candidates],
            ["rejected", "expired", "expired"],
        )
        self.assertEqual(len(expired.rejected_reviews), 1)

    async def test_receipt_replay_before_ttl_keeps_next_review_pending(self) -> None:
        result = await self._start("first", "second")
        command = approve_for(result)
        await self.runtime.submit_command(command)

        replay = await self.runtime.submit_command(command)
        current = await self.runtime.get_run(result.run_id)

        self.assertTrue(replay.idempotent_replay)
        self.assertEqual(current.status, "pending_review")
        self.assertEqual(current.pending_review.cluster_id, "second")

    async def test_changed_replay_conflicts_when_due_without_duplicate_expiry(
        self,
    ) -> None:
        result = await self._start("first", "second")
        command = approve_for(result)
        await self.runtime.submit_command(command)
        self.clock.value += timedelta(hours=2)
        changed = reject_for(result, command_id=command.command_id)

        with self.assertRaises(AudienceReviewConflictError) as raised:
            await self.runtime.submit_command(changed)
        self.assertEqual(
            raised.exception.code,
            ReviewConflictCode.COMMAND_ID_REUSED,
        )
        replay, expired_count = await asyncio.gather(
            self.runtime.submit_command(command),
            self.runtime.expire_due_runs(),
        )
        expired = await self.runtime.get_run(result.run_id)

        self.assertTrue(replay.idempotent_replay)
        self.assertIn(expired_count, {0, 1})
        self.assertEqual(expired.status, "expired")
        second_codes = [event.code for event in expired.traces[1].events]
        self.assertEqual(second_codes.count("review_expired"), 1)

    async def test_checkpoint_contains_only_safe_projected_state(self) -> None:
        result = await self._start("one")
        await assert_saver_is_safe(
            self,
            self.runtime,
            result.thread_id,
            expected_interrupt_count=1,
            forbidden_values=(
                "AudienceProviderResult",
                "must-not-be-checkpointed",
            ),
        )

    async def test_automatic_failure_is_projected_without_raw_error(self) -> None:
        secret = "SECRET_PROVIDER_FAILURE_7fb31"
        runtime = AudienceReviewRuntime(clock=self.clock)
        preparation = prepare_audience_clusters(
            [make_cluster("failure")],
            total_analyzed_views=1_000,
        )
        with self.assertRaises(AudienceReviewRuntimeError) as raised:
            await runtime.start(preparation, FailingProvider(secret))
        self.assertEqual(str(raised.exception), "review_analysis_failed")

        record = next(iter(runtime._runs.values()))
        result = await runtime.get_run(record.run_id)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.failure_code, "automatic_workflow_failed")
        self.assertEqual(result.traces, ())
        self.assertNotIn(secret, result.model_dump_json())
        await assert_saver_is_safe(
            self,
            runtime,
            record.thread_id,
            expected_interrupt_count=0,
            forbidden_values=(secret,),
        )

    async def test_raw_mapping_crosses_discriminated_runtime_boundary(self) -> None:
        result = await self._start("one")
        command = approve_for(result).model_dump(mode="json")
        receipt = await self.runtime.submit_command(command)
        self.assertEqual(receipt.command_type, "approve")
        self.assertEqual(receipt.resulting_status, "published")

    async def test_three_candidate_run_has_one_terminal_outcome_each(self) -> None:
        result = await self._start("first", "second", "third")
        self.assertEqual(
            [item.status for item in result.review_candidates],
            ["pending_review", "queued", "queued"],
        )
        self.assertEqual(
            sum(item.status == "pending_review" for item in result.review_candidates),
            1,
        )
        await self.runtime.submit_command(approve_for(result))
        second = await self.runtime.get_run(result.run_id)
        pending = second.pending_review
        await self.runtime.submit_command(
            RejectReviewCommand(
                type="reject",
                run_id=result.run_id,
                review_id=pending.review_id,
                cluster_id=pending.cluster_id,
                expected_version=1,
                command_id=new_command_id(),
                reason_code=RejectReasonCode.INSUFFICIENT_EVIDENCE,
            )
        )
        third = await self.runtime.get_run(result.run_id)
        await self.runtime.submit_command(approve_for(third))
        completed = await self.runtime.get_run(result.run_id)

        self.assertEqual(completed.status, "completed")
        self.assertEqual(
            [item.status for item in completed.review_candidates],
            ["published", "rejected", "published"],
        )
        self.assertFalse(
            any(
                item.status in {"queued", "pending_review"}
                for item in completed.review_candidates
            )
        )
        for trace in completed.traces:
            if trace.review_id is not None:
                self.assertEqual(
                    sum(
                        event.code in {
                            "audience_published",
                            "analyst_rejected",
                            "review_expired",
                        }
                        for event in trace.events
                    ),
                    1,
                )

    async def test_returned_nested_snapshots_cannot_be_mutated(self) -> None:
        result = await self._start("one")
        pending = result.pending_review
        with self.assertRaises(TypeError):
            pending.evidence[0] = pending.evidence[1]
        with self.assertRaises(TypeError):
            pending.evidence[0].article.daily_views[0] = (
                pending.evidence[0].article.daily_views[0]
            )
        with self.assertRaises(TypeError):
            result.metrics.validation_issue_counts_by_code[0:0] = ()
        with self.assertRaises(TypeError):
            result.metrics.drop_counts_by_code[0:0] = ()
        with self.assertRaises(TypeError):
            result.traces[0:0] = ()

        authoritative = await self.runtime._graph.aget_state(
            {"configurable": {"thread_id": result.thread_id}}
        )
        restored = await self.runtime.get_run(result.run_id)
        self.assertEqual(
            self.runtime._runs[result.run_id].state,
            dict(authoritative.values),
        )
        self.assertEqual(restored.model_dump_json(), result.model_dump_json())

    async def test_ttl_none_zero_negative_and_short_positive(self) -> None:
        default_result = await self._start("default")
        default_record = self.runtime._runs[default_result.run_id]
        self.assertEqual(
            default_record.expires_at - default_record.created_at,
            timedelta(hours=1),
        )
        preparation = prepare_audience_clusters(
            [make_cluster("ttl")],
            total_analyzed_views=1_000,
        )
        provider = FakeProvider(response(make_create("ttl")))
        for invalid in (timedelta(0), timedelta(microseconds=-1)):
            with self.subTest(ttl=invalid):
                before = set(self.runtime._runs)
                with self.assertRaises(ValueError):
                    await self.runtime.start(preparation, provider, ttl=invalid)
                self.assertEqual(set(self.runtime._runs), before)

        short = await self.runtime.start(
            preparation,
            FakeProvider(response(make_create("ttl"))),
            ttl=timedelta(microseconds=1),
        )
        short_record = self.runtime._runs[short.run_id]
        self.assertGreater(short_record.expires_at, short_record.created_at)

    async def test_naive_clock_is_rejected_without_registry_or_saver_residue(self) -> None:
        runtime = AudienceReviewRuntime(
            clock=lambda: datetime(2026, 7, 14, 12),
        )
        preparation = prepare_audience_clusters(
            [make_cluster("one")],
            total_analyzed_views=1_000,
        )
        with self.assertRaises(AudienceReviewRuntimeError) as raised:
            await runtime.start(
                preparation,
                FakeProvider(response(make_create("one"))),
            )
        self.assertEqual(str(raised.exception), "invalid_review_clock")
        self.assertEqual(runtime._runs, {})
        self.assertEqual(runtime._starting_run_ids, set())
        inspection = runtime._inspect_checkpointer()
        self.assertEqual(inspection.storage, ())
        self.assertEqual(inspection.writes, ())
        self.assertEqual(inspection.blobs, ())

    async def test_naive_clock_during_expiry_check_is_rejected_safely(self) -> None:
        result = await self._start("one")
        before = self.runtime._runs[result.run_id].state
        self.clock.value = datetime(2026, 7, 14, 12)
        with self.assertRaises(AudienceReviewRuntimeError) as raised:
            await self.runtime.get_run(result.run_id)
        self.assertEqual(str(raised.exception), "invalid_review_clock")
        self.assertEqual(self.runtime._runs[result.run_id].state, before)

    async def test_failed_projection_is_removed_and_same_run_id_can_retry(self) -> None:
        run_id = new_run_id()
        runtime = AudienceReviewRuntime(
            clock=self.clock,
            run_id_factory=lambda: run_id,
        )
        preparation = prepare_audience_clusters(
            [make_cluster("one")],
            total_analyzed_views=1_000,
        )
        secret = "SECRET_PROJECTION_FAILURE_b8f1"
        with patch(
            "app.agent.audience_review_workflow._project_workflow",
            side_effect=ValueError(secret),
        ):
            with self.assertRaises(AudienceReviewRuntimeError) as raised:
                await runtime.start(
                    preparation,
                    FakeProvider(response(make_create("one"))),
                )
        self.assertEqual(str(raised.exception), "review_analysis_failed")
        self.assertEqual(runtime._runs, {})
        self.assertEqual(runtime._starting_run_ids, set())
        inspection = runtime._inspect_checkpointer()
        self.assertEqual(inspection.storage, ())
        self.assertEqual(inspection.writes, ())
        self.assertEqual(inspection.blobs, ())
        self.assertNotIn(secret, str(raised.exception))

        retry = await runtime.start(
            preparation,
            FakeProvider(response(make_create("one"))),
        )
        self.assertEqual(retry.run_id, run_id)
        self.assertEqual(retry.status, "pending_review")

    async def test_missing_reference_owner_fails_without_runtime_residue(
        self,
    ) -> None:
        valid = prepare_audience_clusters(
            [make_cluster("one")],
            total_analyzed_views=1_000,
        )
        prepared = valid.clusters[0]
        malformed = AudiencePreparation(
            clusters=(
                PreparedAudienceCluster(
                    cluster=prepared.cluster,
                    context=prepared.context,
                    cluster_id=prepared.cluster_id,
                    cluster_pageviews=prepared.cluster_pageviews,
                    evidence_reference_ids=prepared.evidence_reference_ids,
                    resolution_map=prepared.resolution_map,
                ),
            ),
            total_analyzed_views=valid.total_analyzed_views,
            reference_cluster_ids={},
        )

        with self.assertRaises(AudienceReviewRuntimeError) as raised:
            await self.runtime.start(
                malformed,
                FakeProvider(response(make_create("one"))),
            )

        self.assertEqual(str(raised.exception), "review_initialization_failed")
        self.assertEqual(self.runtime._runs, {})
        self.assertEqual(self.runtime._starting_run_ids, set())
        inspection = self.runtime._inspect_checkpointer()
        self.assertEqual(inspection.storage, ())
        self.assertEqual(inspection.writes, ())
        self.assertEqual(inspection.blobs, ())

    async def test_missing_checkpoint_is_a_stable_runtime_conflict(self) -> None:
        saver = InMemorySaver()
        self.runtime._configure_checkpointer(saver)
        result = await self._start("one")
        await saver.adelete_thread(result.thread_id)
        with self.assertRaises(AudienceReviewConflictError) as raised:
            await self.runtime.get_run(result.run_id)
        self.assertEqual(
            raised.exception.code,
            ReviewConflictCode.CHECKPOINT_NOT_FOUND,
        )

    async def test_persistence_gate_has_atomic_write_or_shutdown_ordering(
        self,
    ) -> None:
        gate = Lock()
        authority = {"valid": True}
        saver = BlockingRecordingSaver(block_checkpoint=True)
        fenced = LifecycleFencedCheckpointer(
            saver,
            persistence_gate=gate,
            is_authorized=lambda _capability: authority["valid"],
        )
        shutdown_attempted = Event()
        invalidated = Event()

        def write_checkpoint() -> None:
            async def invoke() -> None:
                with fenced.bind("write-first"):
                    await fenced.aput({}, {}, {}, {})

            asyncio.run(invoke())

        def invalidate() -> None:
            shutdown_attempted.set()
            with gate:
                authority["valid"] = False
                invalidated.set()

        writer = asyncio.create_task(asyncio.to_thread(write_checkpoint))
        await asyncio.to_thread(saver.entered.wait)
        shutdown = asyncio.create_task(asyncio.to_thread(invalidate))
        await asyncio.to_thread(shutdown_attempted.wait)
        self.assertFalse(invalidated.is_set())
        saver.release.set()
        await asyncio.gather(writer, shutdown)
        self.assertEqual(saver.calls, ["checkpoint"])
        self.assertTrue(invalidated.is_set())

        blocked = BlockingRecordingSaver()
        shutdown_first = LifecycleFencedCheckpointer(
            blocked,
            persistence_gate=Lock(),
            is_authorized=lambda _capability: False,
        )
        with self.assertRaises(ReviewPersistenceFenceRejected):
            await shutdown_first.aput({}, {}, {}, {})
        with shutdown_first.bind("shutdown-first"):
            with self.assertRaises(ReviewPersistenceFenceRejected):
                await shutdown_first.aput({}, {}, {}, {})
            with self.assertRaises(ReviewPersistenceFenceRejected):
                await shutdown_first.aput_writes({}, (), "task")
        with self.assertRaises(ReviewPersistenceFenceRejected):
            shutdown_first.put({}, {}, {}, {})
        with self.assertRaises(ReviewPersistenceFenceRejected):
            shutdown_first.put_writes({}, (), "task")
        self.assertEqual(blocked.calls, [])

    async def test_preclose_pending_write_cannot_gain_postclose_checkpoint(
        self,
    ) -> None:
        gate = Lock()
        authority = {"valid": True}
        saver = BlockingRecordingSaver()
        fenced = LifecycleFencedCheckpointer(
            saver,
            persistence_gate=gate,
            is_authorized=lambda _capability: authority["valid"],
        )
        with fenced.bind("partial-write"):
            await fenced.aput_writes({}, (), "task")
            with gate:
                authority["valid"] = False
            with self.assertRaises(ReviewPersistenceFenceRejected):
                await fenced.aput({}, {}, {}, {})
        self.assertEqual(saver.calls, ["pending_writes"])

    async def test_stabilization_cannot_persist_after_runtime_closed(
        self,
    ) -> None:
        self._defer_post_cancel_recovery()
        self._install_apply_checkpoint_barrier()
        result = await self._start("first", "second")
        with self.assertRaises(asyncio.CancelledError):
            await self.runtime.submit_command(approve_for(result))
        record = self.runtime._runs[result.run_id]
        before = self.runtime._inspect_checkpointer()
        before_snapshot = await self.runtime._graph.aget_state(
            self.runtime._config(record.thread_id)
        )
        barrier = CancellationResistantSaverFenceBarrier()
        barrier.install(self.runtime)
        lookup = asyncio.create_task(self.runtime.get_run(result.run_id))
        await asyncio.wait_for(barrier.entered.wait(), timeout=0.5)
        with (
            patch(
                "app.agent.audience_review_runtime."
                "REVIEW_RUNTIME_SHUTDOWN_TIMEOUT_SECONDS",
                0.003,
            ),
            patch(
                "app.agent.audience_review_runtime."
                "REVIEW_RUNTIME_CANCEL_TIMEOUT_SECONDS",
                0.003,
            ),
        ):
            await self.runtime.aclose()
        closed = self.runtime._inspect_checkpointer()
        self.assertEqual(closed, before)
        barrier.release.set()
        await asyncio.wait_for(
            barrier.authorization_attempted.wait(),
            timeout=0.5,
        )
        await asyncio.wait_for(
            asyncio.gather(lookup, return_exceptions=True),
            timeout=0.5,
        )
        after_snapshot = await self.runtime._graph.aget_state(
            self.runtime._config(record.thread_id)
        )
        self.assertEqual(self.runtime._inspect_checkpointer(), closed)
        self.assertEqual(after_snapshot.values, before_snapshot.values)
        self.assertEqual(after_snapshot.next, before_snapshot.next)
        self.assertEqual(after_snapshot.tasks, before_snapshot.tasks)
        self.assertFalse(
            any(
                record["status"] == "pending_review"
                for record in after_snapshot.values["records"]
            )
        )

    async def test_runtime_exposes_only_detached_saver_inspection(self) -> None:
        result = await self._start("one")
        self.assertFalse(hasattr(self.runtime, "checkpointer"))
        self.assertFalse(
            any(
                isinstance(getattr(self.runtime, name), InMemorySaver)
                for name in dir(self.runtime)
                if not name.startswith("_")
            )
        )
        self.assertFalse(
            any(
                isinstance(
                    getattr(self.runtime._fenced_checkpointer, name),
                    InMemorySaver,
                )
                for name in dir(self.runtime._fenced_checkpointer)
                if not name.startswith("_")
            )
        )
        before = self.runtime._inspect_checkpointer()
        detached = self.runtime._inspect_checkpointer(
            self.runtime._config(result.thread_id)
        )
        self.assertGreater(len(detached.checkpoints), 0)
        detached.checkpoints[0].checkpoint["caller_mutation"] = True
        with self.assertRaises((AttributeError, TypeError)):
            detached.storage += (("caller", "mutation"),)
        fenced = self.runtime._fenced_checkpointer
        with self.assertRaises(ReviewPersistenceFenceRejected):
            fenced.put({}, {}, {}, {})
        with self.assertRaises(ReviewPersistenceFenceRejected):
            fenced.put_writes({}, (), "task")
        with self.assertRaises(ReviewPersistenceFenceRejected):
            fenced.delete_thread(result.thread_id)
        with self.assertRaises(ReviewPersistenceFenceRejected):
            await fenced.aput({}, {}, {}, {})
        with self.assertRaises(ReviewPersistenceFenceRejected):
            await fenced.aput_writes({}, (), "task")
        with self.assertRaises(ReviewPersistenceFenceRejected):
            await fenced.adelete_thread(result.thread_id)
        self.assertEqual(self.runtime._inspect_checkpointer(), before)

    async def test_missing_thread_reads_leave_no_saver_residue(self) -> None:
        config = {
            "configurable": {
                "thread_id": "wikipulse-review-v1:missing",
                "checkpoint_ns": "missing-namespace",
            }
        }
        before = self.runtime._inspect_checkpointer()
        fenced = self.runtime._fenced_checkpointer
        self.assertIsNone(fenced.get(config))
        self.assertIsNone(fenced.get_tuple(config))
        self.assertEqual(tuple(fenced.list(config)), ())
        self.assertIsNone(await fenced.aget(config))
        self.assertIsNone(await fenced.aget_tuple(config))
        self.assertEqual(
            tuple([item async for item in fenced.alist(config)]),
            (),
        )
        with self.assertRaises(AudienceReviewConflictError):
            await self.runtime.get_run(new_run_id())
        self.assertEqual(self.runtime._inspect_checkpointer(), before)

    async def test_existing_thread_reads_match_native_inmemory_saver(self) -> None:
        saver = InMemorySaver()
        self.runtime._configure_checkpointer(saver)
        result = await self._start("one")
        config = self.runtime._config(result.thread_id)
        native_tuple = saver.get_tuple(config)
        wrapped_tuple = self.runtime._fenced_checkpointer.get_tuple(config)
        self.assertEqual(wrapped_tuple, native_tuple)
        self.assertEqual(
            tuple(self.runtime._fenced_checkpointer.list(config)),
            tuple(saver.list(config)),
        )

    async def test_owned_cleanup_fences_pending_and_checkpoint_writes(
        self,
    ) -> None:
        for target_kind in ("pending_writes", "checkpoint"):
            with self.subTest(target_kind=target_kind):
                runtime = AudienceReviewRuntime()
                barrier = CancellationResistantSaverFenceBarrier(target_kind)
                barrier.install(runtime)
                run_id = new_run_id()
                operation = asyncio.create_task(
                    runtime.start(
                        prepare_audience_clusters(
                            [make_cluster("one")],
                            total_analyzed_views=1_000,
                        ),
                        FakeProvider(response(make_create("one"))),
                        run_id=run_id,
                    )
                )
                await asyncio.wait_for(barrier.entered.wait(), timeout=0.5)
                active = next(iter(runtime._active_operations.values()))
                ownership = next(iter(active.cleanup_threads.values()))
                self.assertTrue(
                    await runtime._discard_thread(
                        lease=active.lease,
                        run_id=ownership.run_id,
                        thread_id=ownership.thread_id,
                        checkpoint_ns=ownership.checkpoint_ns,
                    )
                )
                cleaned = runtime._inspect_checkpointer()
                self.assertEqual(cleaned.storage, ())
                self.assertEqual(cleaned.writes, ())
                self.assertEqual(cleaned.blobs, ())
                self.assertFalse(
                    await runtime._discard_thread(
                        lease=active.lease,
                        run_id=ownership.run_id,
                        thread_id=ownership.thread_id,
                        checkpoint_ns=ownership.checkpoint_ns,
                    )
                )
                barrier.release.set()
                await asyncio.wait_for(
                    barrier.authorization_attempted.wait(),
                    timeout=0.5,
                )
                outcome = (
                    await asyncio.wait_for(
                        asyncio.gather(operation, return_exceptions=True),
                        timeout=0.5,
                    )
                )[0]
                self.assertIsInstance(outcome, AudienceReviewRuntimeError)
                self.assertEqual(runtime._inspect_checkpointer(), cleaned)
                self.assertEqual(runtime._active_operations, {})
                await runtime.aclose()

    async def test_cleanup_requires_exact_ownership_and_isolates_threads(
        self,
    ) -> None:
        first = await self._start("first")
        first_config = self.runtime._config(first.thread_id)
        first_before = self.runtime._inspect_checkpointer(first_config)
        barrier = CancellationResistantSaverFenceBarrier("checkpoint")
        barrier.install(self.runtime)
        second_run_id = new_run_id()
        operation = asyncio.create_task(
            self.runtime.start(
                prepare_audience_clusters(
                    [make_cluster("second")],
                    total_analyzed_views=1_000,
                ),
                FakeProvider(response(make_create("second"))),
                run_id=second_run_id,
            )
        )
        await asyncio.wait_for(barrier.entered.wait(), timeout=0.5)
        active = next(iter(self.runtime._active_operations.values()))
        ownership = next(iter(active.cleanup_threads.values()))
        unchanged = self.runtime._inspect_checkpointer()
        attempts = (
            (replace(active.lease, operation_id=new_run_id()),
             ownership.run_id, ownership.thread_id, ownership.checkpoint_ns),
            (replace(active.lease, generation=active.lease.generation + 1),
             ownership.run_id, ownership.thread_id, ownership.checkpoint_ns),
            (active.lease, new_run_id(), ownership.thread_id,
             ownership.checkpoint_ns),
            (active.lease, ownership.run_id, f"{ownership.thread_id}-wrong",
             ownership.checkpoint_ns),
            (active.lease, ownership.run_id, ownership.thread_id,
             "wrong-namespace"),
        )
        for lease, run_id, thread_id, namespace in attempts:
            self.assertFalse(
                await self.runtime._discard_thread(
                    lease=lease,
                    run_id=run_id,
                    thread_id=thread_id,
                    checkpoint_ns=namespace,
                )
            )
            self.assertEqual(self.runtime._inspect_checkpointer(), unchanged)
        self.assertTrue(
            await self.runtime._discard_thread(
                lease=active.lease,
                run_id=ownership.run_id,
                thread_id=ownership.thread_id,
                checkpoint_ns=ownership.checkpoint_ns,
            )
        )
        self.assertEqual(
            self.runtime._inspect_checkpointer(first_config),
            first_before,
        )
        barrier.release.set()
        await asyncio.wait_for(
            barrier.authorization_attempted.wait(),
            timeout=0.5,
        )
        await asyncio.wait_for(
            asyncio.gather(operation, return_exceptions=True),
            timeout=0.5,
        )
        self.assertEqual(
            self.runtime._inspect_checkpointer(first_config),
            first_before,
        )


if __name__ == "__main__":
    unittest.main()
