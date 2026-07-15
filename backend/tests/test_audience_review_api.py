"""Focused FastAPI tests for the Phase 2A analyst-review boundary."""

import asyncio
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import timedelta
from datetime import UTC, datetime
import inspect
import json
from types import SimpleNamespace
import threading
import time
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from pydantic import BaseModel, ValidationError

from app.agent.audience_finalization import (
    AudienceSourceIntegrityError,
    prepare_audience_clusters,
)
from app.agent.audience_provider import AnalystEditProviderResult
from app.agent.audience_review_runtime import (
    AudienceReviewRuntime,
    AudienceReviewRuntimeError,
)
from app.api.audience_analysis import AudienceAnalysisResources
from app.api.audience_reviews import AudienceReviewResources
from app.main import create_app
from app.models.audience_review import new_command_id, review_id_for
from app.models.audience_review_api import AudienceReviewStartRequest
from tests.test_audience_review_edit import EditProvider, completed_edit, edited_create
from tests.test_audience_review_runtime import FailingProvider, MutableClock
from tests.test_audience_review_workflow import (
    FakeProvider,
    make_cluster,
    make_create,
    make_skip,
    response,
)


FORBIDDEN_PUBLIC_KEYS = {
    "active_edit",
    "checkpoint",
    "command_digest",
    "request_digest",
    "start_request_digest",
    "fields_to_change",
    "feedback",
    "last_applied_command",
    "operation_id",
    "provider_task",
    "graph_task",
    "private_note",
    "prompt",
    "response_id",
    "thread_id",
}
PRIVATE_VALUE_MARKERS = ("PRIVATE-", "SECRET_", "SENTINEL")
FORBIDDEN_OBJECT_NAMES = {
    "_CommandReservation",
    "_EditOperation",
    "_RunRecord",
    "_StartReservation",
    "Interrupt",
    "StateSnapshot",
}


def assert_public_safe(test: unittest.TestCase, value: object) -> None:
    test.assertNotIsInstance(value, BaseException)
    test.assertNotIsInstance(value, asyncio.Future)
    test.assertNotIsInstance(value, (asyncio.Lock, asyncio.Event))
    test.assertFalse(inspect.iscoroutine(value))
    test.assertNotIn(type(value).__name__, FORBIDDEN_OBJECT_NAMES)
    if not isinstance(value, BaseModel):
        object_name = type(value).__name__.lower()
        test.assertFalse(object_name.endswith("provider"))
        test.assertNotIn("client", object_name)
    test.assertFalse(type(value).__module__.startswith("langgraph"))
    if isinstance(value, str):
        for marker in PRIVATE_VALUE_MARKERS:
            test.assertNotIn(marker, value)
    elif isinstance(value, BaseModel):
        for name in type(value).model_fields:
            assert_public_safe(test, name)
            assert_public_safe(test, getattr(value, name))
    elif isinstance(value, dict):
        test.assertTrue(FORBIDDEN_PUBLIC_KEYS.isdisjoint(value))
        for key, item in value.items():
            assert_public_safe(test, key)
            assert_public_safe(test, item)
    elif isinstance(value, list):
        for item in value:
            assert_public_safe(test, item)
    elif isinstance(value, tuple):
        for item in value:
            assert_public_safe(test, item)


def preparation(*cluster_ids: str):
    return prepare_audience_clusters(
        [make_cluster(cluster_id) for cluster_id in cluster_ids],
        total_analyzed_views=max(1_000, len(cluster_ids) * 300),
    )


def make_resources(
    provider,
    *,
    ttl: timedelta = timedelta(hours=1),
    clock=None,
):
    lock = asyncio.Lock()
    analysis = AudienceAnalysisResources(
        pageview_client=object(),  # type: ignore[arg-type]
        summary_client=object(),  # type: ignore[arg-type]
        encoder=object(),  # type: ignore[arg-type]
        audience_provider=provider,
        analysis_lock=lock,
    )
    runtime = AudienceReviewRuntime(
        clock=clock,
        default_ttl=ttl,
        provider_call_lock=lock,
    )
    return analysis, AudienceReviewResources(analysis=analysis, runtime=runtime)


def start_payload(run_id: str | None = None) -> dict[str, str]:
    return {"run_id": run_id or str(uuid4())}


def command_for(
    run: dict[str, object],
    command_type: str,
    *,
    command_id: str | None = None,
) -> dict[str, object]:
    current = run["current_review"]
    assert isinstance(current, dict)
    command: dict[str, object] = {
        "type": command_type,
        "command_id": command_id or new_command_id(),
        "review_id": current["review_id"],
        "cluster_id": current["cluster_id"],
        "expected_version": current["expected_version"],
    }
    if command_type == "reject":
        command["reason_code"] = "not_commercially_useful"
    elif command_type == "edit_recommendation":
        command["feedback"] = "Make the positioning more practically focused."
        command["fields_to_change"] = ["audience_positioning"]
    return command


def internal_validation_error(value: str = "invalid") -> ValidationError:
    try:
        AudienceReviewStartRequest.model_validate({"run_id": value})
    except ValidationError as exc:
        return exc
    raise AssertionError("expected a validation error")


class BlockingEditProvider(EditProvider):
    def __init__(self) -> None:
        super().__init__(("first",))
        self.thread_started = threading.Event()
        self.thread_release = threading.Event()

    async def regenerate_from_analyst_edit(self, request):
        self.edit_calls += 1
        self.edit_requests.append(request)
        self.thread_started.set()
        await asyncio.to_thread(self.thread_release.wait)
        return self.edit_result


class CancellationResistantPersistenceBarrier:
    """Pause immediately before the production persistence gate."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.authorization_attempted = asyncio.Event()
        self.kind: str | None = None
        self.cancel_count = 0

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
        if self.kind is not None:
            return
        self.kind = kind
        self.entered.set()
        while not self.release.is_set():
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancel_count += 1


class CancellationResistantEditProvider(EditProvider):
    def __init__(self, *, delayed: bool) -> None:
        super().__init__(("first",))
        self.delayed = delayed
        self.after_cancel = asyncio.Event()
        self.release_after_cancel = asyncio.Event()

    async def regenerate_from_analyst_edit(self, request):
        self.edit_calls += 1
        self.edit_requests.append(request)
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.after_cancel.set()
            while not self.release_after_cancel.is_set():
                try:
                    await self.release_after_cancel.wait()
                except asyncio.CancelledError:
                    self.after_cancel.set()
            return self.edit_result


class CancellationResistantStartProvider(FakeProvider):
    def __init__(self) -> None:
        super().__init__(response(make_create("first")))
        self.started = asyncio.Event()
        self.after_cancel = asyncio.Event()
        self.release_after_close = asyncio.Event()
        self.cleanup_complete = asyncio.Event()
        self.cleanup_calls = 0
        self.closed = False
        self.used_after_close = False

    async def generate(self, contexts):
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.after_cancel.set()
            while not self.release_after_close.is_set():
                try:
                    await self.release_after_close.wait()
                except asyncio.CancelledError:
                    self.after_cancel.set()
        self.used_after_close = self.closed
        return await super().generate(contexts)

    async def aclose(self) -> None:
        self.cleanup_calls += 1
        self.closed = True
        self.cleanup_complete.set()


class ManagedAsyncResource:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        return None


class AudienceReviewApiTests(unittest.TestCase):
    def _application(self, provider):
        analysis, review = make_resources(provider)
        return create_app(resources=analysis, review_resources=review), review

    def test_production_shutdown_closes_review_runtime_before_provider(self) -> None:
        events: list[str] = []
        provider = SimpleNamespace(
            aclose=AsyncMock(side_effect=lambda: events.append("provider"))
        )
        async def close_runtime() -> None:
            events.append("runtime")
            await provider.aclose()

        runtime = SimpleNamespace(
            aclose=AsyncMock(side_effect=close_runtime)
        )
        with (
            patch("app.main.WikimediaPageviewsClient", return_value=ManagedAsyncResource()),
            patch("app.main.WikipediaSummaryClient", return_value=ManagedAsyncResource()),
            patch("app.main.MiniLMArticleEncoder", return_value=object()),
            patch(
                "app.main.OpenAIAudienceProvider.from_environment",
                return_value=provider,
            ),
            patch(
                "app.main.AudienceReviewRuntime",
                return_value=runtime,
            ) as runtime_factory,
        ):
            application = create_app()
            with TestClient(application) as client:
                self.assertIs(
                    client.app.state.audience_review_resources.runtime,
                    runtime,
                )

        self.assertEqual(events, ["runtime", "provider"])
        self.assertIs(
            runtime_factory.call_args.kwargs["provider_cleanup"],
            provider.aclose,
        )
        runtime.aclose.assert_awaited_once_with()
        provider.aclose.assert_awaited_once_with()

    def test_runtime_shutdown_drains_inflight_edit_without_task_leak(self) -> None:
        async def scenario() -> None:
            provider = EditProvider(("first",))
            provider.block = True
            runtime = AudienceReviewRuntime()
            started = await runtime.start(preparation("first"), provider)
            pending = started.pending_review
            assert pending is not None
            submission = asyncio.create_task(
                runtime.submit_command(
                    {
                        "type": "edit_recommendation",
                        "run_id": started.run_id,
                        "command_id": new_command_id(),
                        "review_id": pending.review_id,
                        "cluster_id": pending.cluster_id,
                        "expected_version": pending.version,
                        "feedback": (
                            "Make the positioning more practically focused."
                        ),
                        "fields_to_change": ["audience_positioning"],
                    }
                )
            )
            await provider.started.wait()
            await runtime.aclose()
            provider.release.set()
            await asyncio.gather(submission, return_exceptions=True)
            self.assertEqual(provider.edit_calls, 1)
            self.assertEqual(runtime._edit_operations, {})
            self.assertEqual(
                runtime._result(runtime._runs[started.run_id]).status,
                "pending_review",
            )
            self.assertEqual(runtime._active_operations, {})

        asyncio.run(scenario())

    def test_runtime_shutdown_rejects_new_work_and_is_empty_when_idle(self) -> None:
        async def scenario() -> None:
            runtime = AudienceReviewRuntime()
            await runtime.aclose()
            self.assertEqual(runtime._edit_operations, {})
            self.assertEqual(runtime._start_reservations, {})
            with self.assertRaises(AudienceReviewRuntimeError) as raised:
                await runtime.start(preparation(), FakeProvider(None))
            self.assertEqual(raised.exception.code, "review_runtime_closed")

        asyncio.run(scenario())

    def test_runtime_shutdown_bounds_delayed_and_resistant_cancellation(
        self,
    ) -> None:
        async def run_case(*, release_during_cancel: bool) -> None:
            provider = CancellationResistantEditProvider(
                delayed=release_during_cancel
            )
            runtime = AudienceReviewRuntime()
            started = await runtime.start(preparation("first"), provider)
            command = command_for(
                {
                    "run_id": started.run_id,
                    "current_review": {
                        "review_id": started.pending_review.review_id,
                        "cluster_id": started.pending_review.cluster_id,
                        "expected_version": started.pending_review.version,
                    },
                },
                "edit_recommendation",
            )
            command["run_id"] = started.run_id
            submission = asyncio.create_task(runtime.submit_command(command))
            await provider.started.wait()
            began = time.monotonic()
            with (
                patch(
                    "app.agent.audience_review_runtime."
                    "REVIEW_RUNTIME_SHUTDOWN_TIMEOUT_SECONDS",
                    0.005,
                ),
                patch(
                    "app.agent.audience_review_runtime."
                    "REVIEW_RUNTIME_CANCEL_TIMEOUT_SECONDS",
                    0.02 if release_during_cancel else 0.005,
                ),
            ):
                closing = asyncio.create_task(runtime.aclose())
                await provider.after_cancel.wait()
                if release_during_cancel:
                    provider.release_after_cancel.set()
                await closing
            elapsed = time.monotonic() - began
            self.assertLess(elapsed, 0.1)
            self.assertTrue(provider.after_cancel.is_set())
            self.assertEqual(runtime._edit_operations, {})
            if not release_during_cancel:
                provider.release_after_cancel.set()
            await asyncio.gather(submission, return_exceptions=True)

        asyncio.run(run_case(release_during_cancel=True))
        with self.assertLogs(
            "app.agent.audience_review_runtime",
            level="WARNING",
        ) as logs:
            asyncio.run(run_case(release_during_cancel=False))
        rendered = "\n".join(logs.output)
        self.assertIn("detached", rendered)
        self.assertIn("unresponsive task", rendered)

    def test_shutdown_is_shared_idempotent_and_fences_late_provider(self) -> None:
        async def scenario() -> None:
            provider = CancellationResistantEditProvider(delayed=False)
            cleanup_calls: list[str] = []
            cleanup_complete = asyncio.Event()

            async def cleanup_provider() -> None:
                cleanup_calls.append("provider")
                cleanup_complete.set()

            runtime = AudienceReviewRuntime(provider_cleanup=cleanup_provider)
            started = await runtime.start(preparation("first"), provider)
            pending = started.pending_review
            assert pending is not None
            command_id = new_command_id()
            submission = asyncio.create_task(
                runtime.submit_command(
                    {
                        "type": "edit_recommendation",
                        "run_id": started.run_id,
                        "command_id": command_id,
                        "review_id": pending.review_id,
                        "cluster_id": pending.cluster_id,
                        "expected_version": pending.version,
                        "feedback": (
                            "Make the positioning more practically focused."
                        ),
                        "fields_to_change": ["audience_positioning"],
                    }
                )
            )
            await provider.started.wait()
            record = runtime._runs[started.run_id]
            before = (await runtime.get_run(started.run_id)).model_dump_json()

            with (
                patch(
                    "app.agent.audience_review_runtime."
                    "REVIEW_RUNTIME_SHUTDOWN_TIMEOUT_SECONDS",
                    0.005,
                ),
                patch(
                    "app.agent.audience_review_runtime."
                    "REVIEW_RUNTIME_CANCEL_TIMEOUT_SECONDS",
                    0.005,
                ),
            ):
                first = asyncio.create_task(runtime.aclose())
                second = asyncio.create_task(runtime.aclose())
                await provider.after_cancel.wait()
                await asyncio.gather(first, second)
                shared_shutdown = runtime._shutdown_task
                await runtime.aclose()
                self.assertIs(runtime._shutdown_task, shared_shutdown)

            after_close = runtime._result(record).model_dump_json()
            self.assertEqual(after_close, before)
            self.assertNotIn(command_id, record.receipts)
            self.assertEqual(provider.edit_calls, 1)
            self.assertEqual(runtime._edit_operations, {})
            self.assertEqual(runtime._start_reservations, {})

            provider.release_after_cancel.set()
            await asyncio.gather(submission, return_exceptions=True)
            await asyncio.wait_for(cleanup_complete.wait(), timeout=0.1)
            self.assertEqual(runtime._result(record).model_dump_json(), before)
            self.assertNotIn(command_id, record.receipts)
            self.assertEqual(provider.edit_calls, 1)
            self.assertEqual(cleanup_calls, ["provider"])
            await runtime.aclose()
            self.assertEqual(cleanup_calls, ["provider"])

        with self.assertLogs(
            "app.agent.audience_review_runtime",
            level="WARNING",
        ):
            asyncio.run(scenario())

    def test_shutdown_does_not_reconstruct_missing_edit_operation(self) -> None:
        async def scenario() -> None:
            provider = EditProvider(("first",))
            provider.block = True
            runtime = AudienceReviewRuntime()
            started = await runtime.start(preparation("first"), provider)
            pending = started.pending_review
            assert pending is not None
            command_id = new_command_id()
            submission = asyncio.create_task(
                runtime.submit_command(
                    {
                        "type": "edit_recommendation",
                        "run_id": started.run_id,
                        "command_id": command_id,
                        "review_id": pending.review_id,
                        "cluster_id": pending.cluster_id,
                        "expected_version": pending.version,
                        "feedback": (
                            "Make the positioning more practically focused."
                        ),
                        "fields_to_change": ["audience_positioning"],
                    }
                )
            )
            await provider.started.wait()
            operation = runtime._edit_operations.pop(command_id)
            for task in (operation.provider_task, operation.graph_task, submission):
                if task is not None:
                    task.cancel()
            await asyncio.gather(
                *(task for task in (
                    operation.provider_task,
                    operation.graph_task,
                    submission,
                ) if task is not None),
                return_exceptions=True,
            )
            before = runtime._result(runtime._runs[started.run_id]).model_dump_json()
            await runtime.aclose()
            self.assertEqual(provider.edit_calls, 1)
            self.assertEqual(runtime._edit_operations, {})
            self.assertEqual(runtime._runs[started.run_id].receipts, {})
            self.assertEqual(
                runtime._result(runtime._runs[started.run_id]).model_dump_json(),
                before,
            )

        asyncio.run(scenario())

    def test_cancelled_shutdown_caller_waits_for_shared_bounded_decision(
        self,
    ) -> None:
        async def scenario() -> None:
            provider = CancellationResistantEditProvider(delayed=False)
            runtime = AudienceReviewRuntime()
            started = await runtime.start(preparation("first"), provider)
            pending = started.pending_review
            assert pending is not None
            submission = asyncio.create_task(
                runtime.submit_command(
                    {
                        "type": "edit_recommendation",
                        "run_id": started.run_id,
                        "command_id": new_command_id(),
                        "review_id": pending.review_id,
                        "cluster_id": pending.cluster_id,
                        "expected_version": pending.version,
                        "feedback": (
                            "Make the positioning more practically focused."
                        ),
                        "fields_to_change": ["audience_positioning"],
                    }
                )
            )
            await provider.started.wait()
            with (
                patch(
                    "app.agent.audience_review_runtime."
                    "REVIEW_RUNTIME_SHUTDOWN_TIMEOUT_SECONDS",
                    0.005,
                ),
                patch(
                    "app.agent.audience_review_runtime."
                    "REVIEW_RUNTIME_CANCEL_TIMEOUT_SECONDS",
                    0.005,
                ),
            ):
                closing = asyncio.create_task(runtime.aclose())
                await provider.after_cancel.wait()
                closing.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await closing
            self.assertEqual(runtime._lifecycle, "closed")
            self.assertTrue(runtime._shutdown_task.done())
            provider.release_after_cancel.set()
            await asyncio.gather(submission, return_exceptions=True)
            await runtime.aclose()
            self.assertEqual(provider.edit_calls, 1)

        with self.assertLogs(
            "app.agent.audience_review_runtime",
            level="WARNING",
        ):
            asyncio.run(scenario())

    def test_shutdown_fences_start_before_graph_and_inside_provider(self) -> None:
        async def state(runtime: AudienceReviewRuntime) -> dict[str, object]:
            return {
                "runs": tuple(runtime._runs),
                "starting": tuple(sorted(runtime._starting_run_ids)),
                "start_reservations": tuple(runtime._start_reservations),
                "edit_operations": tuple(runtime._edit_operations),
                "active_operations": tuple(runtime._active_operations),
                "saver": runtime._inspect_checkpointer(),
            }

        async def before_graph() -> None:
            provider = FakeProvider(response(make_create("first")))
            runtime = AudienceReviewRuntime()
            entered = asyncio.Event()
            release = asyncio.Event()
            invoke = runtime._graph.ainvoke

            async def blocked_invoke(*args, **kwargs):
                entered.set()
                await release.wait()
                return await invoke(*args, **kwargs)

            runtime._graph.ainvoke = blocked_invoke
            starting = asyncio.create_task(
                runtime.start(preparation("first"), provider)
            )
            await entered.wait()
            await runtime.aclose()
            closed = await state(runtime)
            self.assertEqual(runtime._lifecycle, "closed")
            self.assertEqual(provider.generate_calls, 0)
            self.assertEqual(closed["runs"], ())
            self.assertEqual(closed["starting"], ())
            self.assertEqual(closed["active_operations"], ())
            self.assertEqual(closed["saver"].storage, ())
            self.assertEqual(closed["saver"].writes, ())
            self.assertEqual(closed["saver"].blobs, ())

            release.set()
            outcomes = await asyncio.gather(starting, return_exceptions=True)
            self.assertIsInstance(outcomes[0], AudienceReviewRuntimeError)
            self.assertEqual(await state(runtime), closed)
            self.assertEqual(provider.generate_calls, 0)

        async def inside_provider() -> None:
            provider = CancellationResistantStartProvider()
            runtime = AudienceReviewRuntime(provider_cleanup=provider.aclose)
            starting = asyncio.create_task(
                runtime.start(preparation("first"), provider)
            )
            await provider.started.wait()
            with (
                patch(
                    "app.agent.audience_review_runtime."
                    "REVIEW_RUNTIME_SHUTDOWN_TIMEOUT_SECONDS",
                    0.005,
                ),
                patch(
                    "app.agent.audience_review_runtime."
                    "REVIEW_RUNTIME_CANCEL_TIMEOUT_SECONDS",
                    0.005,
                ),
                self.assertLogs(
                    "app.agent.audience_review_runtime",
                    level="WARNING",
                ),
            ):
                await runtime.aclose()
            await provider.after_cancel.wait()
            closed = await state(runtime)
            self.assertEqual(provider.cleanup_calls, 0)
            self.assertFalse(provider.closed)
            self.assertEqual(closed["runs"], ())
            self.assertEqual(closed["active_operations"], ())
            self.assertEqual(closed["saver"].storage, ())

            provider.release_after_close.set()
            outcomes = await asyncio.gather(starting, return_exceptions=True)
            self.assertIsInstance(outcomes[0], AudienceReviewRuntimeError)
            await asyncio.wait_for(
                provider.cleanup_complete.wait(),
                timeout=0.1,
            )
            self.assertFalse(provider.used_after_close)
            self.assertEqual(provider.cleanup_calls, 1)
            self.assertEqual(await state(runtime), closed)
            await runtime.aclose()
            self.assertEqual(provider.cleanup_calls, 1)

        async def during_api_preparation() -> None:
            provider = FakeProvider(response(make_create("first")))
            application, review = self._application(provider)
            entered = asyncio.Event()
            release = asyncio.Event()

            async def blocked_preparation(*_args, **_kwargs):
                entered.set()
                await release.wait()
                return SimpleNamespace(preparation=preparation("first"))

            transport = ASGITransport(app=application)
            with patch(
                "app.api.audience_reviews.prepare_audience_analysis",
                new=blocked_preparation,
            ):
                async with (
                    application.router.lifespan_context(application),
                    AsyncClient(
                        transport=transport,
                        base_url="http://testserver",
                    ) as client,
                ):
                    request = asyncio.create_task(
                        client.post(
                            "/api/audience-reviews",
                            json={"run_id": str(uuid4())},
                        )
                    )
                    await entered.wait()
                    self.assertEqual(len(review.runtime._active_operations), 1)
                    await review.runtime.aclose()
                    closed = await state(review.runtime)
                    release.set()
                    await asyncio.gather(request, return_exceptions=True)
                    self.assertEqual(await state(review.runtime), closed)
                    self.assertEqual(closed["runs"], ())
                    self.assertEqual(closed["start_reservations"], ())
                    self.assertEqual(closed["active_operations"], ())
                    self.assertEqual(provider.generate_calls, 0)

        asyncio.run(before_graph())
        asyncio.run(inside_provider())
        asyncio.run(during_api_preparation())

    def test_shutdown_fences_commands_edit_and_get_expiry(self) -> None:
        async def authoritative(runtime, record) -> tuple[object, ...]:
            snapshot = await runtime._graph.aget_state(
                runtime._config(record.thread_id)
            )
            tasks = tuple(
                (
                    task.name,
                    task.error,
                    tuple(interrupt.value for interrupt in task.interrupts),
                )
                for task in snapshot.tasks
            )
            receipts = tuple(
                sorted(
                    (
                        command_id,
                        reservation.payload_digest,
                        (
                            None
                            if reservation.receipt is None
                            else reservation.receipt.model_dump_json()
                        ),
                    )
                    for command_id, reservation in record.receipts.items()
                )
            )
            return (
                runtime._result(record).model_dump_json(),
                json.dumps(snapshot.values, sort_keys=True),
                snapshot.next,
                tasks,
                tuple(interrupt.value for interrupt in snapshot.interrupts),
                receipts,
            )

        async def run_case(kind: str) -> None:
            clock = MutableClock()
            provider = (
                EditProvider(("first",))
                if kind == "edit_recommendation"
                else FakeProvider(response(make_create("first")))
            )
            runtime = AudienceReviewRuntime(
                clock=clock,
                default_ttl=timedelta(seconds=1),
            )
            started = await runtime.start(preparation("first"), provider)
            pending = started.pending_review
            assert pending is not None
            record = runtime._runs[started.run_id]
            before = await authoritative(runtime, record)
            entered = asyncio.Event()
            release = asyncio.Event()
            stabilize = runtime._stabilize_and_reconcile
            blocked = False

            async def blocked_stabilize(target, lease):
                nonlocal blocked
                await stabilize(target, lease)
                if not blocked:
                    blocked = True
                    entered.set()
                    await release.wait()

            runtime._stabilize_and_reconcile = blocked_stabilize
            if kind == "get":
                clock.value += timedelta(seconds=2)
                operation = asyncio.create_task(runtime.get_run(started.run_id))
                command_id = None
            else:
                public = {
                    "run_id": started.run_id,
                    "current_review": {
                        "review_id": pending.review_id,
                        "cluster_id": pending.cluster_id,
                        "expected_version": pending.version,
                    },
                }
                command_id = new_command_id()
                command = command_for(
                    public,
                    kind,
                    command_id=command_id,
                )
                command["run_id"] = started.run_id
                operation = asyncio.create_task(runtime.submit_command(command))

            await entered.wait()
            await runtime.aclose()
            closed = await authoritative(runtime, record)
            self.assertEqual(closed, before)
            self.assertEqual(runtime._active_operations, {})
            self.assertEqual(runtime._edit_operations, {})
            self.assertEqual(runtime._start_reservations, {})
            if command_id is not None:
                self.assertNotIn(command_id, record.receipts)
            if kind == "edit_recommendation":
                self.assertEqual(provider.edit_calls, 0)

            release.set()
            await asyncio.gather(operation, return_exceptions=True)
            self.assertEqual(await authoritative(runtime, record), closed)
            self.assertEqual(runtime._active_operations, {})
            self.assertEqual(runtime._edit_operations, {})
            if command_id is not None:
                self.assertNotIn(command_id, record.receipts)
            await runtime.aclose()
            self.assertEqual(await authoritative(runtime, record), closed)

        for kind in ("approve", "reject", "edit_recommendation", "get"):
            with self.subTest(kind=kind):
                asyncio.run(run_case(kind))

    def test_persistence_fence_blocks_post_close_graph_writes(self) -> None:
        def saver_state(runtime: AudienceReviewRuntime) -> object:
            return runtime._inspect_checkpointer()

        async def close_while_blocked(runtime: AudienceReviewRuntime) -> None:
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
                await runtime.aclose()

        async def command_case(kind: str) -> None:
            clock = MutableClock()
            provider = (
                EditProvider(("first",))
                if kind == "edit_recommendation"
                else FakeProvider(response(make_create("first")))
            )
            runtime = AudienceReviewRuntime(
                clock=clock,
                default_ttl=timedelta(seconds=1),
            )
            started = await runtime.start(preparation("first"), provider)
            pending = started.pending_review
            assert pending is not None
            record = runtime._runs[started.run_id]
            before_result = runtime._result(record).model_dump_json()
            barrier = CancellationResistantPersistenceBarrier()
            barrier.install(runtime)
            if kind == "expire":
                clock.value += timedelta(seconds=2)
                operation = asyncio.create_task(runtime.expire_due_runs())
                command_id = None
            else:
                command_id = new_command_id()
                command = {
                    "type": kind,
                    "run_id": started.run_id,
                    "command_id": command_id,
                    "review_id": pending.review_id,
                    "cluster_id": pending.cluster_id,
                    "expected_version": pending.version,
                }
                if kind == "reject":
                    command["reason_code"] = "not_commercially_useful"
                elif kind == "edit_recommendation":
                    command["feedback"] = (
                        "Make the positioning more practically focused."
                    )
                    command["fields_to_change"] = ["audience_positioning"]
                operation = asyncio.create_task(runtime.submit_command(command))

            await asyncio.wait_for(barrier.entered.wait(), timeout=0.5)
            await close_while_blocked(runtime)
            closed_saver = saver_state(runtime)
            closed_result = runtime._result(record).model_dump_json()
            self.assertEqual(closed_result, before_result)
            self.assertEqual(runtime._active_operations, {})
            self.assertEqual(runtime._edit_operations, {})

            barrier.release.set()
            await asyncio.wait_for(
                barrier.authorization_attempted.wait(),
                timeout=0.5,
            )
            outcome = (await asyncio.wait_for(
                asyncio.gather(operation, return_exceptions=True),
                timeout=0.5,
            ))[0]
            self.assertIsInstance(
                outcome,
                (AudienceReviewRuntimeError, asyncio.CancelledError),
            )
            if isinstance(outcome, AudienceReviewRuntimeError):
                self.assertEqual(outcome.code, "review_runtime_closed")
            self.assertEqual(saver_state(runtime), closed_saver)
            self.assertEqual(runtime._result(record).model_dump_json(), closed_result)
            self.assertEqual(record.receipts, {})
            if kind == "edit_recommendation":
                self.assertEqual(provider.edit_calls, 0)
            if command_id is not None:
                self.assertNotIn(command_id, record.receipts)

        async def start_case() -> None:
            provider = FakeProvider(response(make_create("first")))
            runtime = AudienceReviewRuntime()
            barrier = CancellationResistantPersistenceBarrier()
            barrier.install(runtime)
            operation = asyncio.create_task(
                runtime.start(preparation("first"), provider)
            )
            await asyncio.wait_for(barrier.entered.wait(), timeout=0.5)
            await close_while_blocked(runtime)
            closed_saver = saver_state(runtime)
            self.assertEqual(closed_saver.storage, ())
            self.assertEqual(closed_saver.writes, ())
            self.assertEqual(closed_saver.blobs, ())
            barrier.release.set()
            await asyncio.wait_for(
                barrier.authorization_attempted.wait(),
                timeout=0.5,
            )
            outcome = (await asyncio.wait_for(
                asyncio.gather(operation, return_exceptions=True),
                timeout=0.5,
            ))[0]
            self.assertIsInstance(outcome, AudienceReviewRuntimeError)
            self.assertEqual(outcome.code, "review_runtime_closed")
            self.assertEqual(saver_state(runtime), closed_saver)
            self.assertEqual(runtime._runs, {})
            self.assertEqual(runtime._start_reservations, {})
            self.assertEqual(runtime._active_operations, {})
            self.assertEqual(provider.generate_calls, 0)

        asyncio.run(start_case())
        for kind in ("approve", "reject", "edit_recommendation", "expire"):
            with self.subTest(kind=kind):
                asyncio.run(command_case(kind))

    def test_persistence_fence_blocks_late_edit_result_commit(self) -> None:
        async def scenario() -> None:
            provider = EditProvider(("first",))
            provider.block = True
            runtime = AudienceReviewRuntime()
            started = await runtime.start(preparation("first"), provider)
            pending = started.pending_review
            assert pending is not None
            command_id = new_command_id()
            command = {
                "type": "edit_recommendation",
                "run_id": started.run_id,
                "command_id": command_id,
                "review_id": pending.review_id,
                "cluster_id": pending.cluster_id,
                "expected_version": pending.version,
                "feedback": "Make the positioning more practically focused.",
                "fields_to_change": ["audience_positioning"],
            }
            operation = asyncio.create_task(runtime.submit_command(command))
            await provider.started.wait()
            record = runtime._runs[started.run_id]
            editing = await runtime.get_run(started.run_id)
            self.assertEqual(editing.status, "editing")
            barrier = CancellationResistantPersistenceBarrier()
            barrier.install(runtime)
            provider.release.set()
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
                await runtime.aclose()
            closed_saver = runtime._inspect_checkpointer()
            closed_result = runtime._result(record).model_dump_json()
            barrier.release.set()
            await asyncio.wait_for(
                barrier.authorization_attempted.wait(),
                timeout=0.5,
            )
            await asyncio.wait_for(
                asyncio.gather(operation, return_exceptions=True),
                timeout=0.5,
            )
            self.assertEqual(runtime._inspect_checkpointer(), closed_saver)
            self.assertEqual(runtime._result(record).model_dump_json(), closed_result)
            self.assertEqual(runtime._result(record).status, "editing")
            self.assertNotIn(command_id, record.receipts)
            self.assertEqual(provider.edit_calls, 1)

        asyncio.run(scenario())

    def test_client_run_id_starts_once_and_replays_without_preparation(self) -> None:
        provider = FakeProvider(response(make_create("first")))
        application, review = self._application(provider)
        run_id = str(uuid4())
        prepared = SimpleNamespace(preparation=preparation("first"))
        prepare = AsyncMock(return_value=prepared)

        with (
            patch(
                "app.api.audience_reviews.prepare_audience_analysis",
                new=prepare,
            ),
            TestClient(application) as client,
        ):
            first = client.post("/api/audience-reviews", json={"run_id": run_id})
            replay = client.post("/api/audience-reviews", json={"run_id": run_id})
            self.assertFalse(review.analysis.analysis_lock.locked())

        self.assertEqual(first.status_code, 201)
        self.assertEqual(replay.status_code, 200)
        self.assertEqual(first.headers["location"], f"/api/audience-reviews/{run_id}")
        self.assertEqual(first.headers["x-idempotent-replay"], "false")
        self.assertEqual(replay.headers["x-idempotent-replay"], "true")
        self.assertEqual(first.json(), replay.json())
        self.assertEqual(prepare.await_count, 1)
        self.assertEqual(provider.generate_calls, 1)
        self.assertEqual(first.json()["run_id"], run_id)
        self.assertEqual(first.json()["status"], "pending_review")
        self.assertEqual(first.json()["progress"]["current_position"], 1)
        self.assertEqual(first.json()["current_review"]["edit_available"], True)
        for response_value in (first, replay):
            self.assertEqual(response_value.headers["cache-control"], "no-store")
            self.assertEqual(response_value.headers["x-content-type-options"], "nosniff")
            assert_public_safe(self, response_value.json())

    def test_concurrent_start_replay_executes_preparation_once(self) -> None:
        provider = FakeProvider(response(make_create("first")))
        application, _ = self._application(provider)
        payload = start_payload()
        prepare = AsyncMock(
            return_value=SimpleNamespace(preparation=preparation("first"))
        )
        with (
            patch(
                "app.api.audience_reviews.prepare_audience_analysis",
                new=prepare,
            ),
            TestClient(application) as client,
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            futures = [
                executor.submit(
                    client.post,
                    "/api/audience-reviews",
                    json=payload,
                )
                for _ in range(2)
            ]
            responses = [future.result(timeout=2) for future in futures]

        self.assertEqual(sorted(item.status_code for item in responses), [200, 201])
        self.assertEqual(
            sorted(item.headers["x-idempotent-replay"] for item in responses),
            ["false", "true"],
        )
        self.assertEqual(prepare.await_count, 1)
        self.assertEqual(provider.generate_calls, 1)

    def test_start_digest_normalizes_default_and_rejects_conflict(self) -> None:
        provider = FakeProvider(response(make_create("first")))
        application, _ = self._application(provider)
        run_id = str(uuid4())
        prepare = AsyncMock(
            return_value=SimpleNamespace(preparation=preparation("first"))
        )
        with (
            patch(
                "app.api.audience_reviews.prepare_audience_analysis",
                new=prepare,
            ),
            TestClient(application) as client,
        ):
            created = client.post(
                "/api/audience-reviews",
                json={"run_id": run_id},
            )
            equivalent = client.post(
                "/api/audience-reviews",
                json={"run_id": run_id, "ttl_seconds": 3600},
            )
            conflict = client.post(
                "/api/audience-reviews",
                json={"run_id": run_id, "ttl_seconds": 3601},
            )

        self.assertEqual(created.status_code, 201)
        self.assertEqual(equivalent.status_code, 200)
        self.assertEqual(equivalent.headers["x-idempotent-replay"], "true")
        self.assertEqual(conflict.status_code, 409)
        self.assertEqual(
            conflict.json()["error"]["code"],
            "review_start_request_conflict",
        )
        self.assertEqual(prepare.await_count, 1)
        self.assertEqual(provider.generate_calls, 1)

    def test_ttl_preflight_rejects_unrepresentable_expiry_before_work(self) -> None:
        provider = FakeProvider(response(make_create("first")))
        near_max = datetime.max.replace(tzinfo=UTC) - timedelta(seconds=2)
        clock_calls: list[datetime] = []

        def clock() -> datetime:
            clock_calls.append(near_max)
            return near_max

        analysis, review = make_resources(
            provider,
            ttl=timedelta(seconds=1),
            clock=clock,
        )
        application = create_app(
            resources=analysis,
            review_resources=review,
        )
        prepare = AsyncMock(
            return_value=SimpleNamespace(preparation=preparation("first"))
        )
        retry_run_id = str(uuid4())
        with (
            patch(
                "app.api.audience_reviews.prepare_audience_analysis",
                new=prepare,
            ),
            TestClient(application) as client,
        ):
            conversion_overflow = client.post(
                "/api/audience-reviews",
                json={"run_id": str(uuid4()), "ttl_seconds": 10**100},
            )
            expiry_overflow = client.post(
                "/api/audience-reviews",
                json={"run_id": retry_run_id, "ttl_seconds": 3},
            )
            self.assertEqual(prepare.await_count, 0)
            self.assertEqual(review.runtime._start_reservations, {})
            self.assertEqual(review.runtime._runs, {})
            inspection = review.runtime._inspect_checkpointer()
            self.assertEqual(inspection.storage, ())
            self.assertEqual(inspection.writes, ())
            self.assertEqual(inspection.blobs, ())
            valid_retry = client.post(
                "/api/audience-reviews",
                json={"run_id": retry_run_id, "ttl_seconds": 1},
            )
            maximum_adjacent = client.post(
                "/api/audience-reviews",
                json={"run_id": str(uuid4()), "ttl_seconds": 2},
            )

        for invalid in (conversion_overflow, expiry_overflow):
            self.assertEqual(invalid.status_code, 422)
            self.assertEqual(
                invalid.json()["error"]["code"],
                "invalid_review_start",
            )
        self.assertEqual(valid_retry.status_code, 201)
        self.assertEqual(maximum_adjacent.status_code, 201)
        self.assertEqual(prepare.await_count, 2)
        self.assertEqual(provider.generate_calls, 2)
        self.assertEqual(clock_calls, [near_max, near_max, near_max])
        self.assertEqual(review.runtime._start_reservations, {})

    def test_concurrent_conflicting_starts_accept_one_configuration(self) -> None:
        provider = FakeProvider(response(make_create("first")))
        application, _ = self._application(provider)
        run_id = str(uuid4())
        prepare = AsyncMock(
            return_value=SimpleNamespace(preparation=preparation("first"))
        )
        payloads = (
            {"run_id": run_id, "ttl_seconds": 3600},
            {"run_id": run_id, "ttl_seconds": 3601},
        )
        with (
            patch(
                "app.api.audience_reviews.prepare_audience_analysis",
                new=prepare,
            ),
            TestClient(application) as client,
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            responses = [
                future.result(timeout=2)
                for future in (
                    executor.submit(
                        client.post,
                        "/api/audience-reviews",
                        json=payload,
                    )
                    for payload in payloads
                )
            ]

        self.assertEqual(sorted(item.status_code for item in responses), [201, 409])
        self.assertEqual(prepare.await_count, 1)
        self.assertEqual(provider.generate_calls, 1)

    def test_failed_start_releases_reservation_for_exact_retry(self) -> None:
        provider = FakeProvider(response(make_create("first")))
        application, review = self._application(provider)
        run_id = str(uuid4())
        secret = "PRIVATE-START-RETRY-SENTINEL"
        prepare = AsyncMock(
            side_effect=[
                RuntimeError(secret),
                SimpleNamespace(preparation=preparation("first")),
            ]
        )
        with (
            patch(
                "app.api.audience_reviews.prepare_audience_analysis",
                new=prepare,
            ),
            TestClient(application) as client,
        ):
            failed = client.post(
                "/api/audience-reviews",
                json={"run_id": run_id},
            )
            retried = client.post(
                "/api/audience-reviews",
                json={"run_id": run_id},
            )

        self.assertEqual(failed.status_code, 500)
        self.assertEqual(failed.json()["error"]["code"], "internal_error")
        self.assertNotIn(secret, failed.text)
        self.assertEqual(retried.status_code, 201)
        self.assertEqual(prepare.await_count, 2)
        self.assertEqual(provider.generate_calls, 1)
        self.assertEqual(review.runtime._start_reservations, {})

    def test_cancelled_start_waiter_leaves_no_reservation_or_run(self) -> None:
        async def scenario() -> None:
            provider = FakeProvider(response(make_create("first")))
            application, review = self._application(provider)
            run_id = str(uuid4())
            prepare = AsyncMock(
                return_value=SimpleNamespace(preparation=preparation("first"))
            )
            transport = ASGITransport(app=application)
            with patch(
                "app.api.audience_reviews.prepare_audience_analysis",
                new=prepare,
            ):
                async with (
                    application.router.lifespan_context(application),
                    AsyncClient(
                        transport=transport,
                        base_url="http://testserver",
                    ) as client,
                ):
                    await review.analysis.analysis_lock.acquire()
                    request = asyncio.create_task(
                        client.post(
                            "/api/audience-reviews",
                            json={"run_id": run_id},
                        )
                    )
                    while run_id not in review.runtime._start_reservations:
                        await asyncio.sleep(0)
                    request.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await request
                    review.analysis.analysis_lock.release()
                    self.assertNotIn(run_id, review.runtime._start_reservations)
                    self.assertNotIn(run_id, review.runtime._runs)
                    retried = await client.post(
                        "/api/audience-reviews",
                        json={"run_id": run_id},
                    )

            self.assertEqual(retried.status_code, 201)
            self.assertEqual(prepare.await_count, 1)
            self.assertEqual(provider.generate_calls, 1)

        asyncio.run(scenario())

    def test_lost_start_response_replays_published_run_once(self) -> None:
        async def scenario() -> None:
            provider = FakeProvider(response(make_create("first")))
            application, review = self._application(provider)
            run_id = str(uuid4())
            prepare = AsyncMock(
                return_value=SimpleNamespace(preparation=preparation("first"))
            )
            released = asyncio.Event()
            finish_release = asyncio.Event()
            original_release = review.runtime.release_start_request

            async def release_with_barrier(
                supplied_run_id: str,
                request_digest: str,
            ) -> None:
                await original_release(supplied_run_id, request_digest)
                released.set()
                await finish_release.wait()

            transport = ASGITransport(app=application)
            with (
                patch(
                    "app.api.audience_reviews.prepare_audience_analysis",
                    new=prepare,
                ),
                patch.object(
                    review.runtime,
                    "release_start_request",
                    new=release_with_barrier,
                ),
            ):
                async with (
                    application.router.lifespan_context(application),
                    AsyncClient(
                        transport=transport,
                        base_url="http://testserver",
                    ) as client,
                ):
                    request = asyncio.create_task(
                        client.post(
                            "/api/audience-reviews",
                            json={"run_id": run_id},
                        )
                    )
                    await released.wait()
                    self.assertIn(run_id, review.runtime._runs)
                    request.cancel()
                    finish_release.set()
                    with self.assertRaises(asyncio.CancelledError):
                        await request
                    replay = await client.post(
                        "/api/audience-reviews",
                        json={"run_id": run_id},
                    )

            self.assertEqual(replay.status_code, 200)
            self.assertEqual(replay.headers["x-idempotent-replay"], "true")
            self.assertEqual(prepare.await_count, 1)
            self.assertEqual(provider.generate_calls, 1)

        asyncio.run(scenario())

    def test_cancelled_command_after_terminal_checkpoint_recovers_via_get(
        self,
    ) -> None:
        async def scenario() -> None:
            provider = FakeProvider(response(make_create("first")))
            entered = asyncio.Event()
            release = asyncio.Event()

            async def after_resume() -> None:
                entered.set()
                await release.wait()

            lock = asyncio.Lock()
            analysis = AudienceAnalysisResources(
                pageview_client=object(),  # type: ignore[arg-type]
                summary_client=object(),  # type: ignore[arg-type]
                encoder=object(),  # type: ignore[arg-type]
                audience_provider=provider,
                analysis_lock=lock,
            )
            runtime = AudienceReviewRuntime(
                _after_command_resume_hook=after_resume,
                provider_call_lock=lock,
            )
            review = AudienceReviewResources(
                analysis=analysis,
                runtime=runtime,
            )
            application = create_app(
                resources=analysis,
                review_resources=review,
            )
            prepare = AsyncMock(
                return_value=SimpleNamespace(preparation=preparation("first"))
            )
            transport = ASGITransport(app=application)
            with patch(
                "app.api.audience_reviews.prepare_audience_analysis",
                new=prepare,
            ):
                async with (
                    application.router.lifespan_context(application),
                    AsyncClient(
                        transport=transport,
                        base_url="http://testserver",
                    ) as client,
                ):
                    started = (
                        await client.post(
                            "/api/audience-reviews",
                            json=start_payload(),
                        )
                    ).json()
                    command = command_for(started, "approve")
                    submission = asyncio.create_task(
                        client.post(
                            f"/api/audience-reviews/{started['run_id']}/commands",
                            json=command,
                        )
                    )
                    await entered.wait()
                    submission.cancel()
                    release.set()
                    with self.assertRaises(asyncio.CancelledError):
                        await submission
                    recovered = await client.get(
                        f"/api/audience-reviews/{started['run_id']}"
                    )
                    replay = await client.post(
                        f"/api/audience-reviews/{started['run_id']}/commands",
                        json=command,
                    )

            self.assertEqual(recovered.status_code, 200)
            self.assertEqual(recovered.json()["status"], "completed")
            self.assertEqual(len(recovered.json()["published_audiences"]), 1)
            self.assertEqual(replay.status_code, 200)
            self.assertTrue(replay.json()["receipt"]["idempotent_replay"])
            self.assertEqual(provider.generate_calls, 1)

        asyncio.run(scenario())

    def test_invalid_start_is_safe_and_empty_run_completes(self) -> None:
        provider = FakeProvider(None)
        application, _ = self._application(provider)
        prepare = AsyncMock(
            return_value=SimpleNamespace(preparation=preparation())
        )
        with (
            patch(
                "app.api.audience_reviews.prepare_audience_analysis",
                new=prepare,
            ),
            TestClient(application) as client,
        ):
            malformed = client.post(
                "/api/audience-reviews",
                json={"run_id": "00000000-0000-1000-8000-000000000000"},
            )
            extra = client.post(
                "/api/audience-reviews",
                json={"run_id": str(uuid4()), "ttl": 999},
            )
            invalid_ttls = [
                client.post(
                    "/api/audience-reviews",
                    json={"run_id": str(uuid4()), "ttl_seconds": value},
                )
                for value in (0, -1, True)
            ]
            completed = client.post(
                "/api/audience-reviews",
                json=start_payload(),
            )

        self.assertEqual(malformed.status_code, 422)
        self.assertEqual(malformed.json()["error"]["code"], "invalid_review_start")
        self.assertNotIn("input", malformed.text)
        self.assertEqual(extra.status_code, 422)
        self.assertEqual(extra.json()["error"]["code"], "invalid_review_start")
        for invalid_ttl in invalid_ttls:
            self.assertEqual(invalid_ttl.status_code, 422)
            self.assertEqual(
                invalid_ttl.json()["error"]["code"],
                "invalid_review_start",
            )
        self.assertEqual(completed.status_code, 201)
        self.assertEqual(completed.json()["status"], "completed")
        self.assertTrue(completed.json()["is_complete"])
        self.assertIsNone(completed.json()["current_review"])
        self.assertEqual(provider.generate_calls, 0)

    def test_domain_and_internal_failures_are_not_client_validation_errors(self) -> None:
        provider = FakeProvider(response(make_create("first")))
        application, review = self._application(provider)
        source_run_id = str(uuid4())
        internal_run_id = str(uuid4())
        with TestClient(application) as client:
            with patch(
                "app.api.audience_reviews.prepare_audience_analysis",
                new=AsyncMock(
                    side_effect=AudienceSourceIntegrityError(
                        "invalid_total_analyzed_views"
                    )
                ),
            ):
                source = client.post(
                    "/api/audience-reviews",
                    json={"run_id": source_run_id},
                )
            with patch(
                "app.api.audience_reviews.prepare_audience_analysis",
                new=AsyncMock(side_effect=TypeError("PRIVATE-START-INTERNAL")),
            ):
                internal_start = client.post(
                    "/api/audience-reviews",
                    json={"run_id": internal_run_id},
                )
            with patch(
                "app.api.audience_reviews.prepare_audience_analysis",
                new=AsyncMock(
                    return_value=SimpleNamespace(
                        preparation=preparation("first")
                    )
                ),
            ):
                started = client.post(
                    "/api/audience-reviews",
                    json=start_payload(),
                ).json()
            with patch(
                "app.api.audience_reviews.map_audience_review_run",
                side_effect=ValueError("PRIVATE-GET-INTERNAL"),
            ):
                internal_get = client.get(
                    f"/api/audience-reviews/{started['run_id']}"
                )
            command = command_for(started, "approve")
            with patch.object(
                review.runtime,
                "submit_command",
                new=AsyncMock(side_effect=KeyError("PRIVATE-COMMAND-INTERNAL")),
            ):
                internal_command = client.post(
                    f"/api/audience-reviews/{started['run_id']}/commands",
                    json=command,
                )

        self.assertEqual(source.status_code, 500)
        self.assertEqual(
            source.json()["error"]["code"],
            "invalid_total_analyzed_views",
        )
        for response_value in (internal_start, internal_get, internal_command):
            self.assertEqual(response_value.status_code, 500)
            self.assertEqual(response_value.json()["error"]["code"], "internal_error")
            self.assertNotIn("PRIVATE-", response_value.text)

    def test_internal_pydantic_failures_are_server_errors(self) -> None:
        provider = FakeProvider(response(make_create("first")))
        application, review = self._application(provider)
        prepare = AsyncMock(
            return_value=SimpleNamespace(preparation=preparation("first"))
        )
        secret = "PRIVATE-PYDANTIC-INTERNAL-SENTINEL"
        error = internal_validation_error(secret)
        with (
            patch(
                "app.api.audience_reviews.prepare_audience_analysis",
                new=prepare,
            ),
            self.assertLogs("app.api.audience_reviews") as logs,
            TestClient(application) as client,
        ):
            started = client.post(
                "/api/audience-reviews",
                json=start_payload(),
            ).json()
            with patch.object(
                review.runtime,
                "start",
                new=AsyncMock(side_effect=error),
            ):
                start_failure = client.post(
                    "/api/audience-reviews",
                    json=start_payload(),
                )
            with patch(
                "app.api.audience_reviews.map_audience_review_run",
                side_effect=error,
            ):
                get_failure = client.get(
                    f"/api/audience-reviews/{started['run_id']}"
                )
            with patch.object(
                review.runtime,
                "submit_command",
                new=AsyncMock(side_effect=error),
            ):
                command_failure = client.post(
                    f"/api/audience-reviews/{started['run_id']}/commands",
                    json=command_for(started, "approve"),
                )

        for response_value in (start_failure, get_failure, command_failure):
            self.assertEqual(response_value.status_code, 500)
            self.assertEqual(
                response_value.json()["error"]["code"],
                "internal_error",
            )
            self.assertNotIn(secret, response_value.text)
            self.assertNotIn("input", response_value.text)
            assert_public_safe(self, response_value.json())
        self.assertNotIn(secret, "\n".join(logs.output))

    def test_safe_start_failure_is_recoverable_as_failed_run(self) -> None:
        secret = "PRIVATE-START-FAILURE-SENTINEL"
        provider = FailingProvider(secret)
        application, _ = self._application(provider)
        run_id = str(uuid4())
        prepare = AsyncMock(
            return_value=SimpleNamespace(preparation=preparation("first"))
        )
        with (
            patch(
                "app.api.audience_reviews.prepare_audience_analysis",
                new=prepare,
            ),
            TestClient(application) as client,
        ):
            failed = client.post(
                "/api/audience-reviews",
                json={"run_id": run_id},
            )
            missing = client.get(f"/api/audience-reviews/{run_id}")

        self.assertEqual(failed.status_code, 500)
        self.assertEqual(failed.json()["error"]["code"], "review_start_failed")
        self.assertEqual(missing.status_code, 200)
        self.assertEqual(missing.json()["status"], "failed")
        self.assertEqual(
            missing.json()["failure_code"],
            "automatic_workflow_failed",
        )
        self.assertNotIn(secret, failed.text + missing.text)

    def test_get_is_idempotent_and_unknown_is_safe(self) -> None:
        provider = FakeProvider(response(make_create("first")))
        application, _ = self._application(provider)
        payload = start_payload()
        prepare = AsyncMock(
            return_value=SimpleNamespace(preparation=preparation("first"))
        )
        with (
            patch(
                "app.api.audience_reviews.prepare_audience_analysis",
                new=prepare,
            ),
            TestClient(application) as client,
        ):
            started = client.post("/api/audience-reviews", json=payload)
            first = client.get(f"/api/audience-reviews/{payload['run_id']}")
            second = client.get(f"/api/audience-reviews/{payload['run_id']}")
            missing = client.get(f"/api/audience-reviews/{uuid4()}")

        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json(), second.json())
        self.assertEqual(first.json(), started.json())
        self.assertEqual(provider.generate_calls, 1)
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(missing.json()["error"]["code"], "review_run_not_found")
        self.assertEqual(missing.headers["cache-control"], "no-store")

    def test_missing_injected_review_resources_use_safe_review_error(self) -> None:
        analysis, _ = make_resources(FakeProvider(None))
        application = create_app(resources=analysis)

        with TestClient(application) as client:
            response_value = client.get(f"/api/audience-reviews/{uuid4()}")

        self.assertEqual(response_value.status_code, 500)
        self.assertEqual(
            response_value.json()["error"]["code"],
            "review_state_unavailable",
        )
        self.assertEqual(response_value.headers["cache-control"], "no-store")
        self.assertEqual(
            response_value.headers["x-content-type-options"],
            "nosniff",
        )

    def test_approve_retry_and_changed_body_conflict(self) -> None:
        provider = FakeProvider(response(make_create("first")))
        application, _ = self._application(provider)
        prepare = AsyncMock(
            return_value=SimpleNamespace(preparation=preparation("first"))
        )
        with (
            patch(
                "app.api.audience_reviews.prepare_audience_analysis",
                new=prepare,
            ),
            TestClient(application) as client,
        ):
            started = client.post("/api/audience-reviews", json=start_payload()).json()
            command = command_for(started, "approve")
            approved = client.post(
                f"/api/audience-reviews/{started['run_id']}/commands",
                json=command,
            )
            replay = client.post(
                f"/api/audience-reviews/{started['run_id']}/commands",
                json=command,
            )
            changed = dict(command)
            changed["type"] = "reject"
            changed["reason_code"] = "safety_concern"
            conflict = client.post(
                f"/api/audience-reviews/{started['run_id']}/commands",
                json=changed,
            )

        self.assertEqual(approved.status_code, 200)
        self.assertEqual(approved.json()["receipt"]["resulting_status"], "published")
        self.assertFalse(approved.json()["receipt"]["idempotent_replay"])
        self.assertTrue(replay.json()["receipt"]["idempotent_replay"])
        self.assertEqual(len(replay.json()["run"]["published_audiences"]), 1)
        self.assertEqual(
            replay.json()["run"]["published_audiences"][0]["publication_source"],
            "original",
        )
        self.assertEqual(conflict.status_code, 409)
        self.assertEqual(
            conflict.json()["error"]["code"],
            "review_command_id_reused",
        )

    def test_approve_advances_exactly_one_next_candidate(self) -> None:
        provider = FakeProvider(
            response(make_create("first"), make_create("second"))
        )
        application, _ = self._application(provider)
        prepare = AsyncMock(
            return_value=SimpleNamespace(
                preparation=preparation("first", "second")
            )
        )
        with (
            patch(
                "app.api.audience_reviews.prepare_audience_analysis",
                new=prepare,
            ),
            TestClient(application) as client,
        ):
            started = client.post("/api/audience-reviews", json=start_payload()).json()
            advanced = client.post(
                f"/api/audience-reviews/{started['run_id']}/commands",
                json=command_for(started, "approve"),
            )

        payload = advanced.json()["run"]
        self.assertEqual(payload["status"], "pending_review")
        self.assertEqual(payload["current_review"]["cluster_id"], "second")
        self.assertEqual(payload["progress"], {
            "total_reviews": 2,
            "completed_reviews": 1,
            "queued_reviews": 0,
            "current_position": 2,
        })
        event_codes = [
            event["code"]
            for trace in payload["journey"]
            for event in trace["events"]
        ]
        self.assertEqual(event_codes.count("audience_published"), 1)
        self.assertEqual(event_codes.count("review_requested"), 2)

    def test_concurrent_commands_accept_one_transition(self) -> None:
        provider = FakeProvider(response(make_create("first")))
        application, _ = self._application(provider)
        prepare = AsyncMock(
            return_value=SimpleNamespace(preparation=preparation("first"))
        )
        with (
            patch(
                "app.api.audience_reviews.prepare_audience_analysis",
                new=prepare,
            ),
            TestClient(application) as client,
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            started = client.post("/api/audience-reviews", json=start_payload()).json()
            commands = (
                command_for(started, "approve"),
                command_for(started, "reject"),
            )
            futures = [
                executor.submit(
                    client.post,
                    f"/api/audience-reviews/{started['run_id']}/commands",
                    json=command,
                )
                for command in commands
            ]
            responses = [future.result(timeout=2) for future in futures]
            authoritative = client.get(
                f"/api/audience-reviews/{started['run_id']}"
            ).json()

        self.assertEqual(sorted(item.status_code for item in responses), [200, 409])
        self.assertEqual(
            len(authoritative["published_audiences"])
            + len(authoritative["rejected_reviews"]),
            1,
        )

    def test_identity_version_and_expiry_errors_are_stable(self) -> None:
        clock = MutableClock()
        provider = FakeProvider(response(make_create("first")))
        analysis, review = make_resources(
            provider,
            ttl=timedelta(seconds=1),
            clock=clock,
        )
        application = create_app(
            resources=analysis,
            review_resources=review,
        )
        prepare = AsyncMock(
            return_value=SimpleNamespace(preparation=preparation("first"))
        )
        with (
            patch(
                "app.api.audience_reviews.prepare_audience_analysis",
                new=prepare,
            ),
            TestClient(application) as client,
        ):
            started = client.post("/api/audience-reviews", json=start_payload()).json()
            base = command_for(started, "approve")
            stale = dict(base, expected_version=2)
            stale_response = client.post(
                f"/api/audience-reviews/{started['run_id']}/commands",
                json=stale,
            )
            wrong = dict(
                base,
                command_id=new_command_id(),
                review_id=review_id_for(started["run_id"], "other"),
            )
            wrong_response = client.post(
                f"/api/audience-reviews/{started['run_id']}/commands",
                json=wrong,
            )
            clock.value += timedelta(seconds=2)
            expired = client.post(
                f"/api/audience-reviews/{started['run_id']}/commands",
                json=dict(base, command_id=new_command_id()),
            )
            fetched = client.get(
                f"/api/audience-reviews/{started['run_id']}"
            )

        self.assertEqual(stale_response.status_code, 409)
        self.assertEqual(
            stale_response.json()["error"]["code"],
            "review_version_conflict",
        )
        self.assertEqual(wrong_response.status_code, 409)
        self.assertEqual(
            wrong_response.json()["error"]["code"],
            "review_identity_conflict",
        )
        self.assertEqual(expired.status_code, 410)
        self.assertEqual(expired.json()["error"]["code"], "review_run_expired")
        self.assertEqual(fetched.json()["status"], "expired")
        self.assertEqual(len(fetched.json()["expired_reviews"]), 1)

    def test_reject_private_note_never_crosses_public_boundary(self) -> None:
        secret = "PRIVATE-REJECT-API-SENTINEL"
        provider = FakeProvider(response(make_create("first")))
        application, _ = self._application(provider)
        prepare = AsyncMock(
            return_value=SimpleNamespace(preparation=preparation("first"))
        )
        with (
            patch(
                "app.api.audience_reviews.prepare_audience_analysis",
                new=prepare,
            ),
            TestClient(application) as client,
        ):
            started = client.post("/api/audience-reviews", json=start_payload()).json()
            command = command_for(started, "reject")
            command["reason_code"] = "other"
            command["private_note"] = secret + " is private analyst context."
            rejected = client.post(
                f"/api/audience-reviews/{started['run_id']}/commands",
                json=command,
            )

        self.assertEqual(rejected.status_code, 200)
        self.assertEqual(rejected.json()["receipt"]["resulting_status"], "rejected")
        self.assertEqual(
            rejected.json()["run"]["rejected_reviews"][0]["reason_code"],
            "other",
        )
        self.assertNotIn(secret, rejected.text)
        assert_public_safe(self, rejected.json())

    def test_invalid_private_commands_return_only_generic_422(self) -> None:
        secret = "PRIVATE-COMMAND-API-SENTINEL"
        note_secret = "PRIVATE-NOTE-API-SENTINEL"
        provider = FakeProvider(response(make_create("first")))
        application, _ = self._application(provider)
        prepare = AsyncMock(
            return_value=SimpleNamespace(preparation=preparation("first"))
        )
        with (
            patch(
                "app.api.audience_reviews.prepare_audience_analysis",
                new=prepare,
            ),
            self.assertLogs("app.api.audience_reviews") as logs,
            TestClient(application) as client,
        ):
            started = client.post("/api/audience-reviews", json=start_payload()).json()
            malformed = command_for(started, "edit_recommendation")
            malformed["feedback"] = secret + "\u200b"
            response_value = client.post(
                f"/api/audience-reviews/{started['run_id']}/commands",
                json=malformed,
            )
            invalid_reject = command_for(started, "reject")
            invalid_reject["private_note"] = note_secret + "\u0085"
            note_response = client.post(
                f"/api/audience-reviews/{started['run_id']}/commands",
                json=invalid_reject,
            )

        self.assertEqual(response_value.status_code, 422)
        self.assertEqual(
            response_value.json(),
            {
                "error": {
                    "code": "invalid_review_command",
                    "message": "The review command is invalid.",
                }
            },
        )
        self.assertNotIn(secret, response_value.text)
        self.assertEqual(note_response.status_code, 422)
        self.assertNotIn(note_secret, note_response.text)
        rendered_logs = "\n".join(logs.output)
        self.assertNotIn(secret, rendered_logs)
        self.assertNotIn(note_secret, rendered_logs)

    def test_duplicate_json_keys_are_rejected_recursively_without_leakage(
        self,
    ) -> None:
        secret = "PRIVATE-DUPLICATE-JSON-SENTINEL"
        provider = FakeProvider(response(make_create("first")))
        application, _ = self._application(provider)
        prepare = AsyncMock(
            return_value=SimpleNamespace(preparation=preparation("first"))
        )
        with (
            patch(
                "app.api.audience_reviews.prepare_audience_analysis",
                new=prepare,
            ),
            self.assertLogs("app.api.audience_reviews") as logs,
            TestClient(application) as client,
        ):
            started = client.post(
                "/api/audience-reviews",
                json=start_payload(),
            ).json()
            current = started["current_review"]
            assert isinstance(current, dict)
            common = (
                f'"command_id":"{new_command_id()}",'
                f'"review_id":"{current["review_id"]}",'
                f'"cluster_id":"{current["cluster_id"]}",'
                f'"expected_version":{current["expected_version"]}'
            )
            cases = {
                "type": (
                    '{"type":"reject","type":"approve",'
                    f"{common}}}"
                ),
                "command_id": (
                    '{"type":"approve","command_id":"'
                    f'{new_command_id()}","command_id":"{secret}",'
                    f'"review_id":"{current["review_id"]}",'
                    f'"cluster_id":"{current["cluster_id"]}",'
                    f'"expected_version":{current["expected_version"]}}}'
                ),
                "review_id": (
                    '{"type":"approve",'
                    f'"command_id":"{new_command_id()}",'
                    f'"review_id":"{current["review_id"]}",'
                    f'"review_id":"{secret}",'
                    f'"cluster_id":"{current["cluster_id"]}",'
                    f'"expected_version":{current["expected_version"]}}}'
                ),
                "cluster_id": (
                    '{"type":"approve",'
                    f'"command_id":"{new_command_id()}",'
                    f'"review_id":"{current["review_id"]}",'
                    f'"cluster_id":"{current["cluster_id"]}",'
                    f'"cluster_id":"{secret}",'
                    f'"expected_version":{current["expected_version"]}}}'
                ),
                "expected_version": (
                    '{"type":"approve",'
                    f'"command_id":"{new_command_id()}",'
                    f'"review_id":"{current["review_id"]}",'
                    f'"cluster_id":"{current["cluster_id"]}",'
                    f'"expected_version":{current["expected_version"]},'
                    '"expected_version":2}'
                ),
                "feedback": (
                    '{"type":"edit_recommendation",'
                    f"{common},"
                    f'"feedback":"Safe request.","feedback":"{secret}",'
                    '"fields_to_change":["audience_positioning"]}'
                ),
                "private_note": (
                    '{"type":"reject",'
                    f"{common},"
                    '"reason_code":"other",'
                    f'"private_note":"Safe note.","private_note":"{secret}"}}'
                ),
                "fields_to_change": (
                    '{"type":"edit_recommendation",'
                    f"{common},"
                    '"feedback":"Safe request.",'
                    '"fields_to_change":["audience_positioning"],'
                    '"fields_to_change":["brand_categories"]}'
                ),
                "nested": (
                    '{"type":"approve",'
                    f"{common},"
                    f'"nested":{{"value":"safe","value":"{secret}"}}}}'
                ),
            }
            responses = [
                client.post(
                    f"/api/audience-reviews/{started['run_id']}/commands",
                    content=raw,
                    headers={"content-type": "application/json"},
                )
                for raw in cases.values()
            ]
            authoritative = client.get(
                f"/api/audience-reviews/{started['run_id']}"
            )

        for response_value in responses:
            self.assertEqual(response_value.status_code, 422)
            self.assertEqual(
                response_value.json()["error"]["code"],
                "invalid_review_command",
            )
            self.assertNotIn(secret, response_value.text)
            self.assertNotIn(secret, repr(dict(response_value.headers)))
        rendered_logs = "\n".join(logs.output)
        self.assertNotIn(secret, rendered_logs)
        self.assertEqual(authoritative.json()["status"], "pending_review")
        self.assertEqual(authoritative.json()["published_audiences"], [])
        self.assertEqual(authoritative.json()["rejected_reviews"], [])
        assert_public_safe(self, authoritative.json())

    def test_command_media_types_json_shapes_and_size_limit(self) -> None:
        provider = FakeProvider(
            response(
                make_create("one"),
                make_create("two"),
                make_create("three"),
                make_create("four"),
            )
        )
        application, _ = self._application(provider)
        prepare = AsyncMock(
            return_value=SimpleNamespace(
                preparation=preparation("one", "two", "three", "four")
            )
        )
        with (
            patch(
                "app.api.audience_reviews.prepare_audience_analysis",
                new=prepare,
            ),
            TestClient(application) as client,
        ):
            started = client.post(
                "/api/audience-reviews",
                json=start_payload(),
            ).json()
            malformed_responses = [
                client.post(
                    f"/api/audience-reviews/{started['run_id']}/commands",
                    content=body,
                    headers={"content-type": content_type},
                )
                for content_type, body in (
                    ("application/json", "{"),
                    ("application/json; charset=utf-8", "[]"),
                    ("application/problem+json", "null"),
                )
            ]
            unsupported = [
                client.post(
                    f"/api/audience-reviews/{started['run_id']}/commands",
                    content="{}",
                    headers=({} if content_type is None else {
                        "content-type": content_type
                    }),
                )
                for content_type in (
                    None,
                    "text/plain",
                    "application/x-www-form-urlencoded",
                    "multipart/form-data; boundary=safe",
                    "application/xml",
                    "application/+json",
                )
            ]
            body_identity = command_for(started, "approve")
            body_identity["run_id"] = started["run_id"]
            body_identity_response = client.post(
                f"/api/audience-reviews/{started['run_id']}/commands",
                json=body_identity,
            )

            exact_command = command_for(started, "approve")
            exact_raw = json.dumps(exact_command, separators=(",", ":"))
            exact_raw += " " * (16_384 - len(exact_raw.encode("utf-8")))
            exact = client.post(
                f"/api/audience-reviews/{started['run_id']}/commands",
                content=exact_raw,
                headers={"content-type": "application/json"},
            )
            advanced = exact.json()["run"]
            over_command = command_for(advanced, "approve")
            over_raw = json.dumps(over_command, separators=(",", ":"))
            over_raw += " " * (16_385 - len(over_raw.encode("utf-8")))
            over = client.post(
                f"/api/audience-reviews/{started['run_id']}/commands",
                content=over_raw,
                headers={"content-type": "application/json"},
            )

        for response_value in malformed_responses:
            self.assertEqual(response_value.status_code, 422)
            self.assertEqual(
                response_value.json()["error"]["code"],
                "invalid_review_command",
            )
        for response_value in unsupported:
            self.assertEqual(response_value.status_code, 415)
            self.assertEqual(
                response_value.json()["error"]["code"],
                "unsupported_media_type",
            )
        self.assertEqual(body_identity_response.status_code, 422)
        self.assertEqual(
            body_identity_response.json()["error"]["code"],
            "invalid_review_command",
        )
        self.assertEqual(exact.status_code, 200)
        self.assertEqual(over.status_code, 422)
        self.assertEqual(
            over.json()["error"]["code"],
            "invalid_review_command",
        )

    def test_openapi_documents_safe_discriminated_command_contract(self) -> None:
        application, _ = self._application(FakeProvider(None))
        operation = application.openapi()["paths"][
            "/api/audience-reviews/{run_id}/commands"
        ]["post"]
        schema = operation["requestBody"]["content"]["application/json"][
            "schema"
        ]

        self.assertEqual(schema["discriminator"]["propertyName"], "type")
        self.assertEqual(len(schema["oneOf"]), 3)
        rendered = json.dumps(schema, sort_keys=True)
        for command_type in ("approve", "reject", "edit_recommendation"):
            self.assertIn(command_type, rendered)
        variants = {item["title"]: item for item in schema["oneOf"]}
        self.assertTrue(
            variants["RejectReviewApiCommand"]["properties"]["private_note"][
                "writeOnly"
            ]
        )
        self.assertTrue(
            variants["EditRecommendationReviewApiCommand"]["properties"][
                "feedback"
            ]["writeOnly"]
        )
        self.assertNotIn("$ref", rendered)
        for forbidden in (
            "run_id",
            "command_digest",
            "start_request_digest",
            "thread_id",
            "checkpoint",
        ):
            self.assertNotIn(f'"{forbidden}"', rendered)
        self.assertEqual(
            set(operation["responses"]),
            {"200", "404", "409", "410", "415", "422", "500"},
        )

    def test_edit_success_and_terminal_drop_are_200(self) -> None:
        validation_report = SimpleNamespace(
            valid_segments=(),
            provider_skips=(),
            invalid_decisions=(
                SimpleNamespace(
                    issues=(SimpleNamespace(code="synthetic_validation_issue"),)
                ),
            ),
        )
        cases = {
            "published": (
                completed_edit(edited_create("first")),
                nullcontext(),
            ),
            "edit_provider_failed": (
                AnalystEditProviderResult(
                    status="provider_failed",
                    response=None,
                    elapsed_seconds=0,
                    usage=None,
                ),
                nullcontext(),
            ),
            "edit_provider_refused": (
                AnalystEditProviderResult(
                    status="refused",
                    response=None,
                    elapsed_seconds=0,
                    usage=None,
                ),
                nullcontext(),
            ),
            "edit_provider_missing_output": (
                AnalystEditProviderResult(
                    status="missing_output",
                    response=None,
                    elapsed_seconds=0,
                    usage=None,
                ),
                nullcontext(),
            ),
            "edit_zero_decisions": (completed_edit(), nullcontext()),
            "edit_multiple_decisions": (
                completed_edit(
                    edited_create("first"),
                    edited_create("first"),
                ),
                nullcontext(),
            ),
            "edit_wrong_cluster": (
                completed_edit(edited_create("other")),
                nullcontext(),
            ),
            "edit_provider_skip_not_allowed": (
                completed_edit(make_skip("first")),
                nullcontext(),
            ),
            "edit_unsupported_references": (
                completed_edit(
                    edited_create(
                        "first",
                        references=["first:a0", "unknown:a9"],
                    )
                ),
                nullcontext(),
            ),
            "edit_validation_failed": (
                completed_edit(edited_create("first")),
                patch(
                    "app.agent.audience_review_workflow.finalize_audience_decisions",
                    return_value=validation_report,
                ),
            ),
            "edit_intent_conformance_failed": (
                completed_edit(make_create("first")),
                nullcontext(),
            ),
            "edit_internal_failure": (
                completed_edit(edited_create("first")),
                patch(
                    "app.agent.audience_review_workflow._restore_preparation",
                    side_effect=RuntimeError("PRIVATE-INTERNAL-API-SENTINEL"),
                ),
            ),
        }
        for expected, (edit_result, boundary) in cases.items():
            with self.subTest(expected=expected):
                provider = EditProvider(("first",))
                provider.edit_result = edit_result
                application, _ = self._application(provider)
                prepare = AsyncMock(
                    return_value=SimpleNamespace(
                        preparation=preparation("first")
                    )
                )
                with (
                    patch(
                        "app.api.audience_reviews.prepare_audience_analysis",
                        new=prepare,
                    ),
                    TestClient(application) as client,
                ):
                    started = client.post(
                        "/api/audience-reviews",
                        json=start_payload(),
                    ).json()
                    command = command_for(started, "edit_recommendation")
                    with boundary:
                        edited = client.post(
                            f"/api/audience-reviews/{started['run_id']}/commands",
                            json=command,
                        )
                    replay = client.post(
                        f"/api/audience-reviews/{started['run_id']}/commands",
                        json=command,
                    )

                self.assertEqual(edited.status_code, 200)
                self.assertEqual(
                    provider.edit_calls,
                    0 if expected == "edit_internal_failure" else 1,
                )
                self.assertEqual(replay.status_code, 200)
                self.assertTrue(
                    replay.json()["receipt"]["idempotent_replay"]
                )
                if expected == "published":
                    self.assertEqual(
                        edited.json()["receipt"]["resulting_status"],
                        "published",
                    )
                    self.assertEqual(
                        edited.json()["run"]["published_audiences"][0][
                            "publication_source"
                        ],
                        "analyst_edit",
                    )
                else:
                    self.assertEqual(
                        edited.json()["receipt"]["resulting_status"],
                        "edit_validation_dropped",
                    )
                    self.assertEqual(
                        edited.json()["run"]["edit_validation_drops"][0][
                            "drop_code"
                        ],
                        expected,
                    )
                self.assertNotIn("PRIVATE-INTERNAL-API-SENTINEL", edited.text)

    def test_get_reports_editing_and_conflicting_action_without_waiting(self) -> None:
        provider = BlockingEditProvider()
        application, _ = self._application(provider)
        prepare = AsyncMock(
            return_value=SimpleNamespace(preparation=preparation("first"))
        )
        with (
            patch(
                "app.api.audience_reviews.prepare_audience_analysis",
                new=prepare,
            ),
            TestClient(application) as client,
            ThreadPoolExecutor(max_workers=1) as executor,
        ):
            started = client.post("/api/audience-reviews", json=start_payload()).json()
            edit = command_for(started, "edit_recommendation")
            future = executor.submit(
                client.post,
                f"/api/audience-reviews/{started['run_id']}/commands",
                json=edit,
            )
            self.assertTrue(provider.thread_started.wait(timeout=1))
            editing = client.get(
                f"/api/audience-reviews/{started['run_id']}"
            )
            conflicting = client.post(
                f"/api/audience-reviews/{started['run_id']}/commands",
                json={
                    **command_for(started, "approve"),
                    "command_id": new_command_id(),
                },
            )
            provider.thread_release.set()
            completed = future.result(timeout=2)

        self.assertEqual(editing.status_code, 200)
        self.assertEqual(editing.json()["status"], "editing")
        self.assertEqual(editing.json()["current_review"]["status"], "editing")
        self.assertNotIn("expected_version", editing.json()["current_review"])
        self.assertEqual(conflicting.status_code, 409)
        self.assertEqual(
            conflicting.json()["error"]["code"],
            "review_currently_editing",
        )
        self.assertEqual(completed.status_code, 200)
        self.assertEqual(provider.edit_calls, 1)

    def test_provider_skip_and_validation_drop_do_not_create_review(self) -> None:
        provider = FakeProvider(
            response(make_skip("first"), make_create("outside"))
        )
        application, _ = self._application(provider)
        prepare = AsyncMock(
            return_value=SimpleNamespace(preparation=preparation("first"))
        )
        with (
            patch(
                "app.api.audience_reviews.prepare_audience_analysis",
                new=prepare,
            ),
            TestClient(application) as client,
        ):
            result = client.post(
                "/api/audience-reviews",
                json=start_payload(),
            )

        self.assertEqual(result.status_code, 201)
        payload = result.json()
        self.assertEqual(payload["status"], "completed")
        self.assertIsNone(payload["current_review"])
        self.assertEqual([item["cluster_id"] for item in payload["provider_skips"]], ["first"])
        self.assertEqual([item["cluster_id"] for item in payload["validation_drops"]], ["outside"])
        self.assertEqual(payload["progress"]["total_reviews"], 0)


if __name__ == "__main__":
    unittest.main()
