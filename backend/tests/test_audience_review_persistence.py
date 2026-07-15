"""Phase 3 restart recovery tests using an official SQLite saver."""

from __future__ import annotations

import asyncio
from datetime import timedelta
import importlib.util
from pathlib import Path
import tempfile
import unittest

from app.agent.audience_finalization import prepare_audience_clusters
from app.agent.audience_review_runtime import (
    AudienceReviewRuntime,
    AudienceReviewRuntimeError,
)
from app.agent.audience_review_store import AudienceReviewDurableStore
from app.models.audience_review import ReviewConflictCode, new_run_id
from app.agent.audience_review_workflow import AudienceReviewConflictError
from tests.test_audience_review_edit import (
    EditProvider,
    completed_edit,
    edit_for,
)
from tests.test_audience_review_runtime import (
    MutableClock,
    approve_for,
    reject_for,
)
from tests.test_audience_review_workflow import (
    FakeProvider,
    make_cluster,
    make_create,
    response,
)


HAS_SQLITE_SAVER = importlib.util.find_spec(
    "langgraph.checkpoint.sqlite"
) is not None


class DurableStoreSchemaTests(unittest.TestCase):
    def test_schema_reopening_and_incomplete_start_cleanup_are_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "review.db")
            first = AudienceReviewDurableStore(path)
            self.assertTrue(first.claim_start("run", "digest", "2026-01-01T00:00:00+00:00"))
            first.close()
            second = AudienceReviewDurableStore(path)
            self.assertEqual(second.discard_incomplete_starts(), ("run",))
            self.assertEqual(second.discard_incomplete_starts(), ())
            second.close()


@unittest.skipUnless(
    HAS_SQLITE_SAVER,
    "official langgraph-checkpoint-sqlite dependency is not installed",
)
class AudienceReviewRestartTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = str(Path(self.directory.name) / "review.db")
        self.clock = MutableClock()

    async def asyncTearDown(self) -> None:
        self.directory.cleanup()

    def provider(self, *cluster_ids: str) -> FakeProvider:
        return FakeProvider(
            response(*(make_create(cluster_id) for cluster_id in cluster_ids))
        )

    async def start(self, runtime, provider, *cluster_ids: str):
        run_id = new_run_id()
        preparation = prepare_audience_clusters(
            [make_cluster(cluster_id) for cluster_id in cluster_ids],
            total_analyzed_views=max(1_000, len(cluster_ids) * 300),
        )
        result = await runtime.start(
            preparation,
            provider,
            run_id=run_id,
            start_request_digest="start-digest",
            ttl=timedelta(hours=1),
        )
        return result

    def runtime(self) -> AudienceReviewRuntime:
        return AudienceReviewRuntime(
            durable_path=self.path,
            clock=self.clock,
            default_ttl=timedelta(hours=1),
        )

    async def reopen(self, provider):
        runtime = self.runtime()
        calls_before = getattr(provider, "calls", None)
        await runtime.hydrate(provider)
        self.assertEqual(getattr(provider, "calls", None), calls_before)
        return runtime

    async def test_pending_and_start_identity_survive_restart(self):
        provider = self.provider("one")
        first = self.runtime()
        started = await self.start(first, provider, "one")
        await first.aclose()

        second = await self.reopen(provider)
        restored = await second.get_run(started.run_id)
        self.assertEqual(restored.pending_review, started.pending_review)
        self.assertFalse(
            await second.claim_start_request(started.run_id, "start-digest")
        )
        with self.assertRaises(AudienceReviewRuntimeError) as conflict:
            await second.claim_start_request(started.run_id, "other-digest")
        self.assertEqual(
            getattr(conflict.exception, "code", None),
            "review_start_request_conflict",
        )
        with self.assertRaises(AudienceReviewConflictError):
            await second.submit_command(
                approve_for(restored).model_copy(update={"run_id": new_run_id()})
            )
        await second.aclose()

    async def test_application_level_restart_smoke_continues_pending_run(self):
        provider = self.provider("one")
        first = self.runtime()
        started = await self.start(first, provider, "one")
        expected = {
            "pending": started.pending_review,
            "expires_at": started.expires_at,
            "traces": started.traces,
            "metrics": started.metrics,
        }
        await first.aclose()

        second = await self.reopen(provider)
        restored = await second.get_run(started.run_id)
        self.assertEqual(restored.pending_review, expected["pending"])
        self.assertEqual(restored.expires_at, expected["expires_at"])
        self.assertEqual(restored.traces, expected["traces"])
        self.assertEqual(restored.metrics, expected["metrics"])
        receipt = await second.submit_command(approve_for(restored))
        self.assertEqual(receipt.resulting_status, "published")
        self.assertEqual((await second.get_run(started.run_id)).status, "completed")
        await second.aclose()

    async def test_completed_receipt_and_conflict_survive_restart(self):
        provider = self.provider("one")
        first = self.runtime()
        started = await self.start(first, provider, "one")
        command = approve_for(started)
        original = await first.submit_command(command)
        await first.aclose()

        second = await self.reopen(provider)
        replay = await second.submit_command(command)
        self.assertEqual(replay.model_copy(update={"idempotent_replay": False}), original)
        self.assertTrue(replay.idempotent_replay)
        changed = command.model_copy(update={"expected_version": 2})
        with self.assertRaises(AudienceReviewConflictError) as raised:
            await second.submit_command(changed)
        self.assertEqual(raised.exception.code, ReviewConflictCode.COMMAND_ID_REUSED)
        self.assertEqual((await second.get_run(started.run_id)).status, "completed")
        await second.aclose()

    async def test_rejected_and_expired_outcomes_survive_restart(self):
        provider = self.provider("one")
        first = self.runtime()
        rejected = await self.start(first, provider, "one")
        await first.submit_command(reject_for(rejected))
        await first.aclose()
        second = await self.reopen(provider)
        self.assertEqual(len((await second.get_run(rejected.run_id)).rejected_reviews), 1)
        await second.aclose()

        third = self.runtime()
        pending = await self.start(third, provider, "one")
        original_expiry = pending.expires_at
        await third.aclose()
        self.clock.value += timedelta(hours=2)
        fourth = await self.reopen(provider)
        expired = await fourth.get_run(pending.run_id)
        self.assertEqual(expired.status, "expired")
        self.assertEqual(expired.expires_at, original_expiry)
        await fourth.aclose()

    async def test_analyst_edited_publication_survives_restart(self):
        provider = EditProvider(("one",))
        first = self.runtime()
        started = await self.start(first, provider, "one")
        await first.submit_command(edit_for(started))
        self.assertEqual(provider.edit_calls, 1)
        await first.aclose()
        second = await self.reopen(provider)
        restored = await second.get_run(started.run_id)
        self.assertEqual(restored.status, "completed")
        self.assertEqual(len(restored.published_audiences), 1)
        self.assertEqual(provider.edit_calls, 1)
        await second.aclose()

    async def test_edit_drop_and_private_feedback_receipt_survive_restart(self):
        provider = EditProvider(("one",))
        provider.edit_result = completed_edit()
        first = self.runtime()
        started = await self.start(first, provider, "one")
        secret = "PRIVATE-RESTART-FEEDBACK-SENTINEL"
        await first.submit_command(edit_for(started, feedback=secret))
        await first.aclose()
        second = await self.reopen(provider)
        restored = await second.get_run(started.run_id)
        self.assertEqual(restored.status, "completed")
        self.assertEqual(len(restored.edit_validation_drops), 1)
        store = second._durable_store
        self.assertIsNotNone(store)
        receipt_text = "".join(
            item.receipt_json for item in store.load_receipts(started.run_id)
        )
        self.assertNotIn(secret, receipt_text)
        await second.aclose()

    async def test_interrupted_edit_hydrates_failed_without_provider_replay(self):
        provider = EditProvider(("one",))
        provider.block = True
        first = self.runtime()
        started = await self.start(first, provider, "one")
        submission = asyncio.create_task(
            first.submit_command(edit_for(started))
        )
        await provider.started.wait()
        await first.aclose()
        await asyncio.gather(submission, return_exceptions=True)
        self.assertEqual(provider.edit_calls, 1)

        second = await self.reopen(provider)
        restored = await second.get_run(started.run_id)
        self.assertEqual(restored.status, "failed")
        self.assertIsNone(restored.pending_review)
        self.assertEqual(provider.edit_calls, 1)
        await second.aclose()

    async def test_unknown_run_remains_missing(self):
        runtime = self.runtime()
        await runtime.hydrate(self.provider("one"))
        with self.assertRaises(AudienceReviewConflictError) as raised:
            await runtime.get_run(new_run_id())
        self.assertEqual(raised.exception.code, ReviewConflictCode.RUN_NOT_FOUND)
        await runtime.aclose()
