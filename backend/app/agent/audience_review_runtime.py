"""Process-local lifecycle, concurrency, expiry, and idempotency for review."""

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from json import dumps
import logging
from pathlib import Path
import sqlite3
from threading import Lock
from typing import Literal, Sequence
from uuid import UUID, uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command, StateSnapshot

from ..models.audience_review import (
    ActiveAnalystEditSnapshot,
    AppliedReviewCommandSnapshot,
    EditRecommendationReviewCommand,
    ExpireReviewRunCommand,
    PendingAudienceReview,
    ReviewCommand,
    ReviewCommandReceipt,
    ReviewConflictCode,
    ReviewRunResult,
    new_run_id,
    parse_review_command,
    review_thread_id,
)
from ..models.audience_generation import CompactClusterContext
from .audience_finalization import AudiencePreparation
from .audience_provider import (
    AnalystEditProviderRequest,
    AnalystEditProviderResult,
    AudienceGenerationProvider,
    AudienceProviderResult,
    AudienceRevisionRequest,
)
from .audience_review_checkpointer import (
    LifecycleFencedCheckpointer,
    ReviewPersistenceFenceRejected,
    ReviewSaverInspection,
)
from .audience_review_store import (
    AudienceReviewDurableStore,
    DurableReceiptRecord,
    DurableRunRecord,
)
from .audience_review_workflow import (
    AudienceReviewConflictError,
    AudienceReviewState,
    AudienceReviewWorkflowContext,
    build_audience_review_graph,
    build_review_initial_state,
    build_review_run_result,
)


DEFAULT_REVIEW_TTL = timedelta(hours=24)
REVIEW_RUNTIME_SHUTDOWN_TIMEOUT_SECONDS = 30.0
REVIEW_RUNTIME_CANCEL_TIMEOUT_SECONDS = 5.0
MAX_STABILIZATION_CONTINUATIONS = 4
_POST_COMMAND_NODES = frozenset(
    {
        "apply_analyst_command",
        "mark_review_pending",
        "await_analyst_command",
        "finalize_review_run",
        "regenerate_and_finalize_edit",
    }
)


class AudienceReviewRuntimeError(RuntimeError):
    """Safe runtime failure without provider or framework details."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(slots=True)
class _CommandReservation:
    payload_digest: str
    receipt: ReviewCommandReceipt | None = None


@dataclass(slots=True)
class _RunRecord:
    run_id: str
    thread_id: str
    provider: AudienceGenerationProvider
    created_at: datetime
    expires_at: datetime
    start_request_digest: str | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    receipts: dict[str, _CommandReservation] = field(default_factory=dict)
    state: AudienceReviewState | None = None
    failed: bool = False


@dataclass(slots=True)
class _EditOperation:
    """Runtime-only single-flight provider operation for one edit command."""

    record: _RunRecord
    command_id: str
    command_digest: str
    generation: int
    lifecycle_lease: "_LifecycleLease"
    registered: asyncio.Event = field(default_factory=asyncio.Event)
    commit_gate: asyncio.Event = field(default_factory=asyncio.Event)
    abandoned: bool = False
    request_digest: str | None = None
    provider_task: asyncio.Task[AnalystEditProviderResult] | None = None
    graph_task: asyncio.Task[object] | None = None


@dataclass(slots=True)
class _StartReservation:
    """One private API-start owner plus waiters for an identical request."""

    request_digest: str
    lifecycle_lease: "_LifecycleLease"
    completed: asyncio.Event = field(default_factory=asyncio.Event)


_LifecycleOperationKind = Literal[
    "start",
    "command",
    "get_reconcile",
    "expiry_scan",
    "edit_provider_commit",
]


@dataclass(frozen=True, slots=True)
class _LifecycleLease:
    """Immutable authority for one runtime mutation operation."""

    generation: int
    operation_id: str
    kind: _LifecycleOperationKind


@dataclass(frozen=True, slots=True)
class _CleanupThreadOwnership:
    """Exact provisional thread ownership held by one start lease."""

    run_id: str
    thread_id: str
    checkpoint_ns: str = ""


@dataclass(slots=True)
class _ActiveLifecycleOperation:
    """Runtime-only ownership for tasks covered by one lifecycle lease."""

    lease: _LifecycleLease
    tasks: set[asyncio.Task[object]] = field(default_factory=set)
    provider_tasks: set[asyncio.Task[object]] = field(default_factory=set)
    cleanup_threads: dict[str, _CleanupThreadOwnership] = field(
        default_factory=dict
    )
    abandoned: bool = False


@dataclass(frozen=True, slots=True)
class _LifecycleFencedProvider:
    """Prevent a late automatic-provider result from reaching LangGraph."""

    runtime: "AudienceReviewRuntime"
    lease: _LifecycleLease
    provider: AudienceGenerationProvider

    async def generate(
        self,
        contexts: Sequence[CompactClusterContext],
    ) -> AudienceProviderResult:
        self.runtime._assert_provider_lease(self.lease)
        try:
            result = await self.provider.generate(contexts)
        except Exception:
            self.runtime._assert_provider_lease(self.lease)
            raise
        self.runtime._assert_provider_lease(self.lease)
        return result

    async def revise(
        self,
        requests: Sequence[AudienceRevisionRequest],
    ) -> AudienceProviderResult:
        self.runtime._assert_provider_lease(self.lease)
        try:
            result = await self.provider.revise(requests)
        except Exception:
            self.runtime._assert_provider_lease(self.lease)
            raise
        self.runtime._assert_provider_lease(self.lease)
        return result

    async def regenerate_from_analyst_edit(
        self,
        request: AnalystEditProviderRequest,
    ) -> AnalystEditProviderResult:
        self.runtime._assert_provider_lease(self.lease)
        try:
            result = await self.provider.regenerate_from_analyst_edit(request)
        except Exception:
            self.runtime._assert_provider_lease(self.lease)
            raise
        self.runtime._assert_provider_lease(self.lease)
        return result


@dataclass(frozen=True, slots=True)
class NormalizedReviewTTL:
    """One clock reading and representable lifetime reused by API and runtime."""

    lifetime: timedelta
    created_at: datetime
    expires_at: datetime
    microseconds: int


logger = logging.getLogger(__name__)


class AudienceReviewRuntime:
    """Own the review graph plus a process-local hydrated registry."""

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] | None = None,
        run_id_factory: Callable[[], str] = new_run_id,
        default_ttl: timedelta = DEFAULT_REVIEW_TTL,
        _after_command_resume_hook: (
            Callable[[], Awaitable[None]] | None
        ) = None,
        provider_call_lock: asyncio.Lock | None = None,
        provider_cleanup: Callable[[], Awaitable[None]] | None = None,
        durable_path: str | None = None,
    ) -> None:
        if default_ttl <= timedelta(0):
            raise ValueError("default_ttl must be positive")
        self._clock = clock or (lambda: datetime.now(UTC))
        self._run_id_factory = run_id_factory
        self._default_ttl = default_ttl
        self._after_command_resume_hook = _after_command_resume_hook
        self._provider_call_lock = provider_call_lock
        self._provider_cleanup = provider_cleanup
        self._persistence_gate = Lock()
        self._durable_store: AudienceReviewDurableStore | None = None
        self._durable_connection: sqlite3.Connection | None = None
        abandoned_start_run_ids: tuple[str, ...] = ()
        if durable_path is None:
            checkpointer: BaseCheckpointSaver = InMemorySaver()
        else:
            Path(durable_path).parent.mkdir(parents=True, exist_ok=True)
            try:
                from langgraph.checkpoint.sqlite import SqliteSaver
            except ImportError:
                raise AudienceReviewRuntimeError(
                    "review_durable_store_unavailable"
                ) from None
            self._durable_connection = sqlite3.connect(
                durable_path,
                check_same_thread=False,
            )
            checkpointer = SqliteSaver(self._durable_connection)
            checkpointer.setup()
            self._durable_store = AudienceReviewDurableStore(durable_path)
            abandoned_start_run_ids = self._durable_store.load_incomplete_starts()
        self._runs: dict[str, _RunRecord] = {}
        self._starting_run_ids: set[str] = set()
        self._start_reservations: dict[str, _StartReservation] = {}
        self._start_reservation_lock = asyncio.Lock()
        self._edit_operations: dict[str, _EditOperation] = {}
        self._active_operations: dict[
            str, _ActiveLifecycleOperation
        ] = {}
        self._lifecycle: Literal["open", "closing", "closed"] = "open"
        self._generation = 1
        self._shutdown_task: asyncio.Task[None] | None = None
        self._provider_cleanup_task: asyncio.Task[None] | None = None
        self._provider_cleanup_waiting_on: set[asyncio.Task[object]] = set()
        self._configure_checkpointer(checkpointer)
        if abandoned_start_run_ids:
            with self._persistence_gate:
                for abandoned_run_id in abandoned_start_run_ids:
                    self._fenced_checkpointer._delete_thread_administratively(
                        review_thread_id(abandoned_run_id)
                    )
                    assert self._durable_store is not None
                    self._durable_store.discard_incomplete_start(
                        abandoned_run_id
                    )

    def _inspect_checkpointer(
        self,
        config: dict | None = None,
    ) -> ReviewSaverInspection:
        """Return a detached structural view without saver mutation access."""
        return self._fenced_checkpointer._inspect(config)

    @property
    def default_ttl(self) -> timedelta:
        """Expose the immutable server default for canonical API requests."""
        return self._default_ttl

    def normalize_start_ttl(
        self,
        *,
        ttl: timedelta | None = None,
        ttl_seconds: int | None = None,
    ) -> NormalizedReviewTTL:
        """Validate one effective expiry before reservation or expensive work."""
        if ttl is not None and ttl_seconds is not None:
            raise AudienceReviewRuntimeError("invalid_review_ttl")
        try:
            lifetime = (
                timedelta(seconds=ttl_seconds)
                if ttl_seconds is not None
                else (self._default_ttl if ttl is None else ttl)
            )
        except (OverflowError, TypeError, ValueError):
            raise AudienceReviewRuntimeError("invalid_review_ttl") from None
        if lifetime <= timedelta(0):
            raise ValueError("ttl must be positive")
        created_at = self._safe_now()
        try:
            expires_at = created_at + lifetime
        except (OverflowError, ValueError):
            raise AudienceReviewRuntimeError("invalid_review_ttl") from None
        if expires_at <= created_at:
            raise AudienceReviewRuntimeError("invalid_review_ttl")
        microseconds = (
            lifetime.days * 86_400_000_000
            + lifetime.seconds * 1_000_000
            + lifetime.microseconds
        )
        return NormalizedReviewTTL(
            lifetime=lifetime,
            created_at=created_at,
            expires_at=expires_at,
            microseconds=microseconds,
        )

    async def aclose(self) -> None:
        """Share one bounded OPEN -> CLOSING -> CLOSED transition."""
        shutdown_task = self._shutdown_task
        if shutdown_task is None:
            with self._persistence_gate:
                if self._lifecycle == "closed":
                    return
                closing_generation = self._generation
                self._lifecycle = "closing"
                self._generation += 1
                active_operations = tuple(self._active_operations.values())
                for active in active_operations:
                    active.abandoned = True
            shutdown_task = asyncio.create_task(
                self._close_once(closing_generation, active_operations)
            )
            self._shutdown_task = shutdown_task

        caller_cancelled = False
        wait_budget = (
            REVIEW_RUNTIME_SHUTDOWN_TIMEOUT_SECONDS
            + REVIEW_RUNTIME_CANCEL_TIMEOUT_SECONDS
            + 0.1
        )
        while not shutdown_task.done():
            try:
                done, _ = await asyncio.wait(
                    {shutdown_task},
                    timeout=wait_budget,
                )
            except asyncio.CancelledError:
                caller_cancelled = True
                continue
            if not done:
                shutdown_task.cancel()
                done, _ = await asyncio.wait(
                    {shutdown_task},
                    timeout=REVIEW_RUNTIME_CANCEL_TIMEOUT_SECONDS,
                )
                if not done:
                    shutdown_task.add_done_callback(_consume_task_result)
                    logger.warning(
                        "Audience review shutdown lifecycle did not terminate"
                    )
                break
        if shutdown_task.done():
            _consume_task_result(shutdown_task)
        if caller_cancelled:
            raise asyncio.CancelledError

    async def _close_once(
        self,
        generation: int,
        active_operations: tuple[_ActiveLifecycleOperation, ...],
    ) -> None:
        """Own the single bounded shutdown decision for this runtime."""
        edit_operations = tuple(
            operation
            for operation in self._edit_operations.values()
            if operation.generation == generation
        )
        operation_tasks = {
            task
            for active in active_operations
            for task in active.tasks
            if not task.done()
        }
        operation_tasks.update(
            task
            for operation in edit_operations
            for task in (operation.provider_task, operation.graph_task)
            if task is not None and not task.done()
        )
        provider_tasks = {
            task
            for active in active_operations
            for task in active.provider_tasks
            if not task.done()
        }
        provider_tasks.update(
            operation.provider_task
            for operation in edit_operations
            if operation.provider_task is not None
            and not operation.provider_task.done()
        )
        cleanup_threads = tuple(
            (active.lease, ownership)
            for active in active_operations
            for ownership in active.cleanup_threads.values()
        )
        pending: set[asyncio.Task[object]] = set(operation_tasks)
        cleanup_tasks: set[asyncio.Task[object]] = set()
        try:
            for task in pending:
                task.cancel()
            if pending:
                done, pending = await asyncio.wait(
                    pending,
                    timeout=REVIEW_RUNTIME_SHUTDOWN_TIMEOUT_SECONDS,
                )
                for task in done:
                    _consume_task_result(task)

            for task in pending:
                task.cancel()
            cleanup_tasks = {
                asyncio.create_task(
                    self._discard_thread(
                        lease=lease,
                        run_id=ownership.run_id,
                        thread_id=ownership.thread_id,
                        checkpoint_ns=ownership.checkpoint_ns,
                    )
                )
                for lease, ownership in cleanup_threads
            }
            pending.update(cleanup_tasks)
            cancel_deadline = (
                asyncio.get_running_loop().time()
                + REVIEW_RUNTIME_CANCEL_TIMEOUT_SECONDS
            )
            if pending:
                done, pending = await asyncio.wait(
                    pending,
                    timeout=REVIEW_RUNTIME_CANCEL_TIMEOUT_SECONDS,
                )
                for task in done:
                    _consume_task_result(task)

            for task in pending:
                task.add_done_callback(_consume_task_result)
            if pending:
                logger.warning(
                    "Audience review shutdown detached %d unresponsive task(s)",
                    len(pending),
                )

            self._defer_provider_cleanup(
                {task for task in provider_tasks if not task.done()}
            )

            cleanup_task = self._provider_cleanup_task
            remaining = max(
                0.0,
                cancel_deadline - asyncio.get_running_loop().time(),
            )
            if cleanup_task is not None and not cleanup_task.done() and remaining:
                done, _ = await asyncio.wait(
                    {cleanup_task},
                    timeout=remaining,
                )
                if done:
                    _consume_task_result(cleanup_task)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                "Audience review shutdown failed safely: context=%s",
                type(exc).__name__,
            )
        finally:
            remaining_operations = tuple(self._edit_operations.values())
            for operation in remaining_operations:
                self._abandon_operation(operation)
            remaining_provider_tasks = {
                task
                for task in provider_tasks
                if not task.done()
            }
            remaining_provider_tasks.update(
                operation.provider_task
                for operation in remaining_operations
                if operation.provider_task is not None
                and not operation.provider_task.done()
            )
            self._defer_provider_cleanup(remaining_provider_tasks)
            for task in pending:
                if not task.done():
                    task.cancel()
                    task.add_done_callback(_consume_task_result)
            self._active_operations.clear()
            self._edit_operations.clear()
            self._starting_run_ids.clear()
            self._clear_incomplete_command_reservations()
            async with self._start_reservation_lock:
                reservations = tuple(self._start_reservations.values())
                self._start_reservations.clear()
            for reservation in reservations:
                reservation.completed.set()
            self._lifecycle = "closed"
            try:
                if self._durable_store is not None:
                    self._durable_store.close()
                    self._durable_store = None
            except Exception as exc:
                logger.error(
                    "Audience review durable index close failed safely: "
                    "context=%s",
                    type(exc).__name__,
                )
            try:
                self._fenced_checkpointer._close_administratively()
            except Exception as exc:
                logger.error(
                    "Audience review checkpointer close failed safely: "
                    "context=%s",
                    type(exc).__name__,
                )

    def _defer_provider_cleanup(
        self,
        pending_provider_tasks: set[asyncio.Task[object]],
    ) -> None:
        """Close the owned provider once no provider coroutine can use it."""
        if (
            self._provider_cleanup is None
            or self._provider_cleanup_task is not None
            or self._provider_cleanup_waiting_on
        ):
            return
        self._provider_cleanup_waiting_on = set(pending_provider_tasks)
        if not self._provider_cleanup_waiting_on:
            self._start_provider_cleanup()
            return
        for task in self._provider_cleanup_waiting_on:
            task.add_done_callback(self._provider_task_finished)

    def _provider_task_finished(self, task: asyncio.Task[object]) -> None:
        _consume_task_result(task)
        self._provider_cleanup_waiting_on.discard(task)
        if not self._provider_cleanup_waiting_on:
            self._start_provider_cleanup()

    def _start_provider_cleanup(self) -> None:
        if self._provider_cleanup is None or self._provider_cleanup_task is not None:
            return
        self._provider_cleanup_task = asyncio.create_task(
            self._run_provider_cleanup()
        )
        self._provider_cleanup_task.add_done_callback(_consume_task_result)

    async def _run_provider_cleanup(self) -> None:
        cleanup = self._provider_cleanup
        if cleanup is None:
            return
        try:
            await cleanup()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                "Audience review provider cleanup failed safely: context=%s",
                type(exc).__name__,
            )

    def _acquire_mutation_lease(
        self,
        kind: _LifecycleOperationKind,
        *,
        provider_capable: bool = False,
    ) -> _LifecycleLease:
        """Atomically register one mutation authority while OPEN."""
        self._require_open()
        task = asyncio.current_task()
        if task is None:
            raise AudienceReviewRuntimeError("review_runtime_closed")
        lease = _LifecycleLease(
            generation=self._generation,
            operation_id=str(uuid4()),
            kind=kind,
        )
        active = _ActiveLifecycleOperation(lease=lease)
        active.tasks.add(task)
        if provider_capable:
            active.provider_tasks.add(task)
        self._active_operations[lease.operation_id] = active
        return lease

    def _assert_mutation_lease(self, lease: _LifecycleLease) -> None:
        """Reject any mutation after this operation lost runtime authority."""
        active = self._active_operations.get(lease.operation_id)
        if (
            self._lifecycle != "open"
            or lease.generation != self._generation
            or active is None
            or active.lease != lease
            or active.abandoned
        ):
            raise AudienceReviewRuntimeError("review_runtime_closed")

    def _mutation_lease_is_valid(self, lease: _LifecycleLease) -> bool:
        try:
            self._assert_mutation_lease(lease)
        except AudienceReviewRuntimeError:
            return False
        return True

    def _persistence_capability_is_valid(self, capability: object) -> bool:
        return isinstance(capability, _LifecycleLease) and (
            self._mutation_lease_is_valid(capability)
        )

    def _configure_checkpointer(self, saver: BaseCheckpointSaver) -> None:
        """Compile the review graph against the lifecycle-fenced saver."""
        self._checkpointer = saver
        self._fenced_checkpointer = LifecycleFencedCheckpointer(
            saver,
            persistence_gate=self._persistence_gate,
            is_authorized=self._persistence_capability_is_valid,
        )
        self._graph = build_audience_review_graph(self._fenced_checkpointer)

    async def hydrate(self, provider: AudienceGenerationProvider) -> int:
        """Load published run indexes and authoritative checkpoints read-only."""
        self._require_open()
        if self._durable_store is None:
            return 0
        hydrated = 0
        for durable in self._durable_store.load_runs():
            if durable.run_id in self._runs:
                continue
            try:
                created_at = datetime.fromisoformat(durable.created_at)
                expires_at = datetime.fromisoformat(durable.expires_at)
                if created_at.tzinfo is None or expires_at.tzinfo is None:
                    raise ValueError
                record = _RunRecord(
                    run_id=durable.run_id,
                    thread_id=durable.thread_id,
                    provider=provider,
                    created_at=created_at.astimezone(UTC),
                    expires_at=expires_at.astimezone(UTC),
                    start_request_digest=durable.start_request_digest,
                    failed=durable.failed,
                )
                snapshot = await self._graph.aget_state(
                    self._config(record.thread_id)
                )
                if not snapshot.values or snapshot.values.get("run_id") != record.run_id:
                    continue
                record.state = dict(snapshot.values)
                result = self._result(record)
                if result.status == "editing":
                    # A provider call cannot be proven complete across process
                    # death. Fail closed without replaying it or advancing graph.
                    record.failed = True
                for saved in self._durable_store.load_receipts(record.run_id):
                    receipt = ReviewCommandReceipt.model_validate_json(
                        saved.receipt_json
                    )
                    record.receipts[saved.command_id] = _CommandReservation(
                        payload_digest=saved.command_digest,
                        receipt=receipt,
                    )
                raw_applied = record.state.get("last_applied_command")
                if raw_applied is not None:
                    applied = AppliedReviewCommandSnapshot.model_validate_json(
                        dumps(raw_applied)
                    )
                    record.receipts.setdefault(
                        applied.command_id,
                        _CommandReservation(
                            payload_digest=applied.command_digest,
                        ),
                    )
                self._runs[record.run_id] = record
                hydrated += 1
            except Exception:
                logger.error(
                    "Audience review durable hydration skipped one invalid run"
                )
        return hydrated

    async def _ainvoke_review_graph(
        self,
        graph_input: object,
        *,
        record: _RunRecord,
        lease: _LifecycleLease,
        provider: AudienceGenerationProvider | None = None,
    ) -> object:
        """Invoke LangGraph with one private persistence capability."""
        self._assert_mutation_lease(lease)
        with self._fenced_checkpointer.bind(lease) as capability:
            try:
                return await self._graph.ainvoke(
                    graph_input,
                    config=self._config(record.thread_id),
                    context=self._context(record, provider=provider),
                    durability="sync",
                )
            except ReviewPersistenceFenceRejected:
                raise AudienceReviewRuntimeError(
                    "review_runtime_closed"
                ) from None
            except asyncio.CancelledError:
                if capability.rejected:
                    raise AudienceReviewRuntimeError(
                        "review_runtime_closed"
                    ) from None
                raise

    def _assert_provider_lease(self, lease: _LifecycleLease) -> None:
        """Use cancellation so the outer graph cannot checkpoint a late result."""
        if not self._mutation_lease_is_valid(lease):
            raise asyncio.CancelledError

    def _register_lease_task(
        self,
        lease: _LifecycleLease,
        task: asyncio.Task[object],
        *,
        provider_capable: bool = False,
    ) -> None:
        self._assert_mutation_lease(lease)
        active = self._active_operations[lease.operation_id]
        active.tasks.add(task)
        if provider_capable:
            active.provider_tasks.add(task)

    def _register_lease_thread(
        self,
        lease: _LifecycleLease,
        run_id: str,
        thread_id: str,
        checkpoint_ns: str = "",
    ) -> None:
        self._assert_mutation_lease(lease)
        ownership = _CleanupThreadOwnership(
            run_id=run_id,
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
        )
        self._active_operations[lease.operation_id].cleanup_threads[
            thread_id
        ] = ownership

    def _release_mutation_lease(self, lease: _LifecycleLease) -> None:
        active = self._active_operations.get(lease.operation_id)
        if active is not None and active.lease == lease:
            self._active_operations.pop(lease.operation_id, None)

    def _start_reservation_lease(
        self,
        run_id: str,
        request_digest: str | None,
    ) -> _LifecycleLease | None:
        if request_digest is None:
            return None
        reservation = self._start_reservations.get(run_id)
        if (
            reservation is None
            or reservation.request_digest != request_digest
        ):
            return None
        task = asyncio.current_task()
        active = self._active_operations.get(
            reservation.lifecycle_lease.operation_id
        )
        if active is None or task not in active.tasks:
            return None
        self._assert_mutation_lease(reservation.lifecycle_lease)
        return reservation.lifecycle_lease

    async def claim_start_request(
        self,
        run_id: str,
        request_digest: str,
    ) -> bool:
        """Return whether this caller owns a new canonical start request."""
        while True:
            self._require_open()
            async with self._start_reservation_lock:
                self._require_open()
                record = self._runs.get(run_id)
                if record is not None:
                    if record.start_request_digest != request_digest:
                        raise AudienceReviewRuntimeError(
                            "review_start_request_conflict"
                        )
                    return False
                reservation = self._start_reservations.get(run_id)
                if reservation is None:
                    lease = self._acquire_mutation_lease(
                        "start",
                        provider_capable=True,
                    )
                    if self._durable_store is not None:
                        try:
                            claimed = self._durable_store.claim_start(
                                run_id,
                                request_digest,
                                self._safe_now().isoformat(),
                            )
                            if not claimed:
                                self._release_mutation_lease(lease)
                                raise AudienceReviewRuntimeError(
                                    "review_state_unavailable"
                                )
                        except ValueError:
                            self._release_mutation_lease(lease)
                            raise AudienceReviewRuntimeError(
                                "review_start_request_conflict"
                            ) from None
                        except Exception:
                            self._release_mutation_lease(lease)
                            raise AudienceReviewRuntimeError(
                                "review_state_unavailable"
                            ) from None
                    self._start_reservations[run_id] = _StartReservation(
                        request_digest=request_digest,
                        lifecycle_lease=lease,
                    )
                    return True
                if reservation.request_digest != request_digest:
                    raise AudienceReviewRuntimeError(
                        "review_start_request_conflict"
                    )
                completed = reservation.completed
            await completed.wait()

    async def release_start_request(
        self,
        run_id: str,
        request_digest: str,
    ) -> None:
        """Release one matching start reservation and wake exact waiters."""
        async with self._start_reservation_lock:
            reservation = self._start_reservations.get(run_id)
            if reservation is None or reservation.request_digest != request_digest:
                return
            self._start_reservations.pop(run_id, None)
            if self._durable_store is not None and run_id not in self._runs:
                self._durable_store.release_incomplete_start(
                    run_id,
                    request_digest,
                )
            reservation.completed.set()
            self._release_mutation_lease(reservation.lifecycle_lease)

    async def start(
        self,
        preparation: AudiencePreparation,
        provider: AudienceGenerationProvider,
        *,
        ttl: timedelta | None = None,
        normalized_ttl: NormalizedReviewTTL | None = None,
        run_id: str | None = None,
        start_request_digest: str | None = None,
    ) -> ReviewRunResult:
        """Run automatic analysis and return at completion or first pause."""
        self._require_open()
        if normalized_ttl is not None and ttl is not None:
            raise AudienceReviewRuntimeError("invalid_review_ttl")
        normalized = normalized_ttl or self.normalize_start_ttl(ttl=ttl)
        now = normalized.created_at
        expires_at = normalized.expires_at
        run_id = self._run_id_factory() if run_id is None else run_id
        try:
            parsed_run_id = UUID(run_id)
        except (TypeError, ValueError, AttributeError):
            raise AudienceReviewRuntimeError("invalid_run_id") from None
        if parsed_run_id.version != 4 or str(parsed_run_id) != run_id:
            raise AudienceReviewRuntimeError("invalid_run_id")
        lease = self._start_reservation_lease(
            run_id,
            start_request_digest,
        )
        release_lease = lease is None
        if lease is None:
            lease = self._acquire_mutation_lease(
                "start",
                provider_capable=True,
            )
        thread_id = review_thread_id(run_id)
        record: _RunRecord | None = None
        try:
            self._assert_mutation_lease(lease)
            if run_id in self._runs or run_id in self._starting_run_ids:
                raise AudienceReviewRuntimeError("duplicate_run_id")
            try:
                initial_state = build_review_initial_state(
                    preparation,
                    run_id=run_id,
                    expires_at=expires_at.isoformat(),
                )
            except Exception:
                raise AudienceReviewRuntimeError(
                    "review_initialization_failed"
                ) from None
            record = _RunRecord(
                run_id=run_id,
                thread_id=thread_id,
                provider=provider,
                created_at=now,
                expires_at=expires_at,
                start_request_digest=start_request_digest,
            )
            self._register_lease_thread(lease, run_id, record.thread_id)
            self._assert_mutation_lease(lease)
            self._starting_run_ids.add(run_id)
            async with record.lock:
                self._assert_mutation_lease(lease)
                fenced_provider = _LifecycleFencedProvider(
                    runtime=self,
                    lease=lease,
                    provider=provider,
                )
                await self._ainvoke_review_graph(
                    initial_state,
                    record=record,
                    lease=lease,
                    provider=fenced_provider,
                )
                self._assert_mutation_lease(lease)
                await self._stabilize(record, lease)
                self._assert_mutation_lease(lease)
            result = self._result(record)
            if result.failure_code == "review_projection_failed":
                await self._discard_thread(
                    lease=lease,
                    run_id=run_id,
                    thread_id=record.thread_id,
                )
                raise AudienceReviewRuntimeError("review_analysis_failed")
            self._assert_mutation_lease(lease)
            self._persist_record(record)
            self._assert_mutation_lease(lease)
            self._runs[run_id] = record
            if result.status == "failed":
                self._assert_mutation_lease(lease)
                record.failed = True
                raise AudienceReviewRuntimeError("review_analysis_failed")
            return result
        except asyncio.CancelledError:
            lease_was_valid = self._mutation_lease_is_valid(lease)
            await self._discard_thread(
                lease=lease,
                run_id=run_id,
                thread_id=thread_id,
            )
            if not lease_was_valid:
                raise AudienceReviewRuntimeError(
                    "review_runtime_closed"
                ) from None
            raise
        except AudienceReviewRuntimeError:
            if record is not None and run_id not in self._runs:
                await self._discard_thread(
                    lease=lease,
                    run_id=run_id,
                    thread_id=record.thread_id,
                )
            raise
        except Exception:
            await self._discard_thread(
                lease=lease,
                run_id=run_id,
                thread_id=thread_id,
            )
            raise AudienceReviewRuntimeError("review_analysis_failed") from None
        finally:
            self._starting_run_ids.discard(run_id)
            if release_lease:
                self._release_mutation_lease(lease)

    async def get_run(self, run_id: str) -> ReviewRunResult:
        """Look up a run, externally expiring it first when due."""
        lease = self._acquire_mutation_lease("get_reconcile")
        try:
            record = self._require_run(run_id)
            async with record.lock:
                self._assert_mutation_lease(lease)
                if record.failed:
                    return self._result(record)
                await self._stabilize_and_reconcile(record, lease)
                self._assert_mutation_lease(lease)
                await self._progress_edit_locked(
                    record,
                    lease=lease,
                    wait_for_provider=False,
                )
                self._assert_mutation_lease(lease)
                await self._expire_if_due(record, lease=lease)
                self._assert_mutation_lease(lease)
                return self._result(record)
        finally:
            self._release_mutation_lease(lease)

    async def peek_run(self, run_id: str) -> ReviewRunResult:
        """Return a read-only cached snapshot without graph reconciliation."""
        self._require_open()
        record = self._require_run(run_id)
        async with record.lock:
            self._require_open()
            return self._result(record)

    async def submit_command(
        self,
        command: ReviewCommand | Mapping[str, object],
        *,
        thread_id: str | None = None,
    ) -> ReviewCommandReceipt:
        """Serialize one command and enforce replay and identity semantics."""
        lease = self._acquire_mutation_lease("command")
        try:
            validated = parse_review_command(command)
            return await self._submit_validated_command(
                validated,
                thread_id=thread_id,
                lease=lease,
            )
        finally:
            self._release_mutation_lease(lease)

    async def _submit_validated_command(
        self,
        command: ReviewCommand,
        *,
        thread_id: str | None,
        lease: _LifecycleLease,
    ) -> ReviewCommandReceipt:
        """Apply a command that crossed the strict parsing boundary."""
        if isinstance(command, EditRecommendationReviewCommand):
            return await self._submit_edit_command(
                command,
                thread_id=thread_id,
                lease=lease,
            )
        record = self._require_run(command.run_id)
        supplied_thread = thread_id or record.thread_id
        if supplied_thread != record.thread_id:
            raise AudienceReviewConflictError(ReviewConflictCode.WRONG_THREAD)
        payload = command.model_dump(mode="json")
        payload_digest = _payload_digest(payload)
        async with record.lock:
            self._assert_mutation_lease(lease)
            await self._stabilize_and_reconcile(record, lease)
            self._assert_mutation_lease(lease)
            prior = record.receipts.get(command.command_id)
            if prior is not None:
                if prior.payload_digest != payload_digest:
                    raise AudienceReviewConflictError(
                        ReviewConflictCode.COMMAND_ID_REUSED
                    )

            await self._expire_if_due(
                record,
                lease=lease,
                stabilize=False,
            )
            self._assert_mutation_lease(lease)
            if prior is not None and prior.receipt is not None:
                return prior.receipt.model_copy(
                    update={"idempotent_replay": True}
                )
            current_result = self._result(record)
            if current_result.status == "expired":
                raise AudienceReviewConflictError(ReviewConflictCode.RUN_EXPIRED)
            if current_result.status == "editing":
                raise AudienceReviewConflictError(
                    ReviewConflictCode.REVIEW_CURRENTLY_EDITING
                )
            if current_result.status != "pending_review":
                raise AudienceReviewConflictError(ReviewConflictCode.RUN_TERMINAL)
            pending = current_result.pending_review
            if pending is None:
                raise AudienceReviewConflictError(
                    ReviewConflictCode.REVIEW_NOT_PENDING
                )
            _require_match(
                command.review_id,
                pending.review_id,
                ReviewConflictCode.REVIEW_ID_MISMATCH,
            )
            _require_match(
                command.cluster_id,
                pending.cluster_id,
                ReviewConflictCode.CLUSTER_ID_MISMATCH,
            )
            _require_match(
                command.expected_version,
                pending.version,
                ReviewConflictCode.STALE_VERSION,
            )
            if prior is None:
                self._assert_mutation_lease(lease)
                prior = _CommandReservation(payload_digest=payload_digest)
                record.receipts[command.command_id] = prior
            graph_payload = dict(payload)
            graph_payload.pop("private_note", None)
            graph_payload["command_digest"] = payload_digest
            try:
                self._assert_mutation_lease(lease)
                await self._ainvoke_review_graph(
                    Command(resume=graph_payload),
                    record=record,
                    lease=lease,
                )
                self._assert_mutation_lease(lease)
                if self._after_command_resume_hook is not None:
                    await self._after_command_resume_hook()
                    self._assert_mutation_lease(lease)
                await self._stabilize_and_reconcile(record, lease)
                self._assert_mutation_lease(lease)
            except BaseException:
                if self._mutation_lease_is_valid(lease):
                    await self._best_effort_stabilize_and_reconcile(
                        record,
                        lease,
                    )
                raise
            self._assert_mutation_lease(lease)
            reservation = record.receipts[command.command_id]
            if reservation.receipt is None:
                raise AudienceReviewRuntimeError(
                    "review_command_outcome_unavailable"
                )
            return reservation.receipt

    async def _submit_edit_command(
        self,
        command: EditRecommendationReviewCommand,
        *,
        thread_id: str | None,
        lease: _LifecycleLease,
    ) -> ReviewCommandReceipt:
        """Run one edit with provider I/O outside the per-run lock."""
        record = self._require_run(command.run_id)
        supplied_thread = thread_id or record.thread_id
        if supplied_thread != record.thread_id:
            raise AudienceReviewConflictError(ReviewConflictCode.WRONG_THREAD)
        payload = command.model_dump(mode="json")
        payload_digest = _payload_digest(payload)
        operation: _EditOperation | None = None

        async with record.lock:
            self._assert_mutation_lease(lease)
            await self._stabilize_and_reconcile(record, lease)
            self._assert_mutation_lease(lease)
            prior = record.receipts.get(command.command_id)
            if prior is not None and prior.payload_digest != payload_digest:
                raise AudienceReviewConflictError(
                    ReviewConflictCode.COMMAND_ID_REUSED
                )
            await self._expire_if_due(
                record,
                lease=lease,
                stabilize=False,
            )
            self._assert_mutation_lease(lease)
            if prior is not None and prior.receipt is not None:
                return prior.receipt.model_copy(
                    update={"idempotent_replay": True}
                )

            current_result = self._result(record)
            if current_result.status == "expired":
                raise AudienceReviewConflictError(
                    ReviewConflictCode.RUN_EXPIRED
                )
            if current_result.status == "editing":
                active = self._active_edit(record)
                if active.command_id != command.command_id:
                    raise AudienceReviewConflictError(
                        ReviewConflictCode.REVIEW_CURRENTLY_EDITING
                    )
                operation = self._operation_for_active_edit(
                    record,
                    command.command_id,
                    payload_digest,
                    lease=lease,
                )
                await self._ensure_edit_invocation_locked(record, operation)
                self._assert_mutation_lease(lease)
            else:
                if any(
                    candidate.review_id == command.review_id
                    and candidate.cluster_id == command.cluster_id
                    and candidate.edit_attempted
                    for candidate in current_result.review_candidates
                ):
                    raise AudienceReviewConflictError(
                        ReviewConflictCode.EDIT_ALREADY_ATTEMPTED
                    )
                if current_result.status != "pending_review":
                    raise AudienceReviewConflictError(
                        ReviewConflictCode.RUN_TERMINAL
                    )
                pending = current_result.pending_review
                if pending is None:
                    raise AudienceReviewConflictError(
                        ReviewConflictCode.REVIEW_NOT_PENDING
                    )
                _require_match(
                    command.review_id,
                    pending.review_id,
                    ReviewConflictCode.REVIEW_ID_MISMATCH,
                )
                _require_match(
                    command.cluster_id,
                    pending.cluster_id,
                    ReviewConflictCode.CLUSTER_ID_MISMATCH,
                )
                _require_match(
                    command.expected_version,
                    pending.version,
                    ReviewConflictCode.STALE_VERSION,
                )
                if prior is None:
                    self._assert_mutation_lease(lease)
                    prior = _CommandReservation(payload_digest=payload_digest)
                    record.receipts[command.command_id] = prior
                operation = self._edit_operations.get(command.command_id)
                if operation is None:
                    self._assert_mutation_lease(lease)
                    operation_lease = self._acquire_mutation_lease(
                        "edit_provider_commit"
                    )
                    operation = _EditOperation(
                        record=record,
                        command_id=command.command_id,
                        command_digest=payload_digest,
                        generation=operation_lease.generation,
                        lifecycle_lease=operation_lease,
                    )
                    self._edit_operations[command.command_id] = operation
                elif operation.command_digest != payload_digest:
                    raise AudienceReviewConflictError(
                        ReviewConflictCode.COMMAND_ID_REUSED
                    )
                graph_payload = dict(payload)
                graph_payload["command_digest"] = payload_digest
                await self._ensure_edit_invocation_locked(
                    record,
                    operation,
                    resume_payload=graph_payload,
                )
                self._assert_mutation_lease(lease)
            await self._stabilize_and_reconcile(record, lease)
            self._assert_mutation_lease(lease)
            reservation = record.receipts.get(command.command_id)
            if reservation is not None and reservation.receipt is not None:
                receipt = reservation.receipt
                self._cleanup_edit_operation(command.command_id)
                return receipt

        if operation is None or operation.provider_task is None:
            raise AudienceReviewRuntimeError(
                "review_edit_operation_unavailable"
            )
        await asyncio.shield(operation.provider_task)
        self._assert_mutation_lease(lease)

        async with record.lock:
            self._assert_mutation_lease(lease)
            reservation = record.receipts.get(command.command_id)
            if reservation is not None and reservation.receipt is not None:
                return reservation.receipt
            if not self._can_commit(operation):
                raise AudienceReviewRuntimeError("review_runtime_closed")
            await self._commit_edit_operation_locked(record, operation)
            self._assert_mutation_lease(lease)
            await self._expire_if_due(
                record,
                lease=lease,
                stabilize=False,
            )
            self._assert_mutation_lease(lease)
            reservation = record.receipts.get(command.command_id)
            if reservation is None or reservation.receipt is None:
                raise AudienceReviewRuntimeError(
                    "review_command_outcome_unavailable"
                )
            receipt = reservation.receipt
            self._cleanup_edit_operation(command.command_id)
            return receipt

    async def execute(
        self,
        request: AnalystEditProviderRequest,
        *,
        command_id: str,
        command_digest: str,
    ) -> AnalystEditProviderResult:
        """Register and await one runtime-only single-flight provider call."""
        operation = self._edit_operations.get(command_id)
        if operation is None or operation.command_digest != command_digest:
            return _safe_edit_provider_failure()
        if not self._can_commit(operation):
            operation.registered.set()
            raise asyncio.CancelledError
        self._assert_mutation_lease(operation.lifecycle_lease)
        request_digest = _payload_digest(request.model_dump(mode="json"))
        if operation.request_digest is None:
            operation.request_digest = request_digest
        elif operation.request_digest != request_digest:
            operation.registered.set()
            return _safe_edit_provider_failure()
        if operation.provider_task is None:
            self._assert_provider_lease(operation.lifecycle_lease)
            operation.provider_task = asyncio.create_task(
                self._call_edit_provider(operation, request)
            )
            self._register_lease_task(
                operation.lifecycle_lease,
                operation.provider_task,
                provider_capable=True,
            )
        operation.registered.set()
        result = await asyncio.shield(operation.provider_task)
        self._assert_provider_lease(operation.lifecycle_lease)
        await operation.commit_gate.wait()
        if not self._can_commit(operation):
            raise asyncio.CancelledError
        return result

    async def _call_edit_provider(
        self,
        operation: _EditOperation,
        request: AnalystEditProviderRequest,
    ) -> AnalystEditProviderResult:
        record = operation.record
        try:
            if self._provider_call_lock is None:
                self._assert_provider_lease(operation.lifecycle_lease)
                result = await record.provider.regenerate_from_analyst_edit(
                    request
                )
            else:
                async with self._provider_call_lock:
                    self._assert_provider_lease(operation.lifecycle_lease)
                    result = await record.provider.regenerate_from_analyst_edit(
                        request
                    )
        except asyncio.CancelledError:
            raise
        except Exception:
            return _safe_edit_provider_failure()
        if not isinstance(result, AnalystEditProviderResult):
            return _safe_edit_provider_failure()
        return result

    async def _progress_edit_locked(
        self,
        record: _RunRecord,
        *,
        lease: _LifecycleLease,
        wait_for_provider: bool,
    ) -> None:
        self._assert_mutation_lease(lease)
        result = self._result(record)
        if result.status != "editing":
            return
        active = self._active_edit(record)
        operation = self._operation_for_active_edit(
            record,
            active.command_id,
            active.command_digest,
            lease=lease,
        )
        await self._ensure_edit_invocation_locked(record, operation)
        self._assert_mutation_lease(lease)
        provider_task = operation.provider_task
        if provider_task is None:
            raise AudienceReviewRuntimeError(
                "review_edit_operation_unavailable"
            )
        if not provider_task.done() and not wait_for_provider:
            return
        if wait_for_provider:
            await asyncio.shield(provider_task)
            self._assert_mutation_lease(lease)
        if provider_task.done():
            await self._commit_edit_operation_locked(record, operation)
            self._assert_mutation_lease(lease)
            reservation = record.receipts.get(active.command_id)
            if reservation is not None and reservation.receipt is not None:
                self._cleanup_edit_operation(active.command_id)

    def _operation_for_active_edit(
        self,
        record: _RunRecord,
        command_id: str,
        command_digest: str,
        *,
        lease: _LifecycleLease,
    ) -> _EditOperation:
        self._assert_mutation_lease(lease)
        active = self._active_edit(record)
        if (
            active.command_id != command_id
            or active.command_digest != command_digest
        ):
            raise AudienceReviewConflictError(
                ReviewConflictCode.COMMAND_ID_REUSED
            )
        operation = self._edit_operations.get(command_id)
        if operation is None:
            operation_lease = self._acquire_mutation_lease(
                "edit_provider_commit"
            )
            operation = _EditOperation(
                record=record,
                command_id=command_id,
                command_digest=command_digest,
                generation=operation_lease.generation,
                lifecycle_lease=operation_lease,
            )
            self._edit_operations[command_id] = operation
        elif (
            operation.record is not record
            or operation.command_digest != command_digest
        ):
            raise AudienceReviewConflictError(
                ReviewConflictCode.COMMAND_ID_REUSED
            )
        return operation

    async def _ensure_edit_invocation_locked(
        self,
        record: _RunRecord,
        operation: _EditOperation,
        *,
        resume_payload: dict[str, object] | None = None,
    ) -> None:
        self._assert_mutation_lease(operation.lifecycle_lease)
        graph_task = operation.graph_task
        if graph_task is None or graph_task.done():
            graph_input: object = (
                Command(resume=resume_payload)
                if resume_payload is not None
                else None
            )
            graph_task = asyncio.create_task(
                self._ainvoke_review_graph(
                    graph_input,
                    record=record,
                    lease=operation.lifecycle_lease,
                )
            )
            operation.graph_task = graph_task
            self._register_lease_task(
                operation.lifecycle_lease,
                graph_task,
            )
        if operation.registered.is_set():
            return
        registration_waiter = asyncio.create_task(operation.registered.wait())
        try:
            done, _ = await asyncio.wait(
                {graph_task, registration_waiter},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if graph_task in done and not operation.registered.is_set():
                await graph_task
                self._assert_mutation_lease(operation.lifecycle_lease)
                return
        finally:
            if not registration_waiter.done():
                registration_waiter.cancel()

    async def _commit_edit_operation_locked(
        self,
        record: _RunRecord,
        operation: _EditOperation,
    ) -> None:
        if not self._can_commit(operation):
            raise AudienceReviewRuntimeError("review_runtime_closed")
        await self._stabilize_and_reconcile(
            record,
            operation.lifecycle_lease,
        )
        self._assert_mutation_lease(operation.lifecycle_lease)
        if self._result(record).status != "editing":
            return
        await self._ensure_edit_invocation_locked(record, operation)
        if operation.provider_task is None or not operation.provider_task.done():
            return
        if operation.graph_task is None:
            raise AudienceReviewRuntimeError(
                "review_edit_operation_unavailable"
            )
        if not self._can_commit(operation):
            raise AudienceReviewRuntimeError("review_runtime_closed")
        operation.commit_gate.set()
        graph_task = operation.graph_task
        try:
            await asyncio.shield(graph_task)
        except asyncio.CancelledError:
            if self._mutation_lease_is_valid(operation.lifecycle_lease):
                try:
                    await graph_task
                finally:
                    raise
            raise
        except Exception:
            if self._mutation_lease_is_valid(operation.lifecycle_lease):
                await self._best_effort_stabilize_and_reconcile(
                    record,
                    operation.lifecycle_lease,
                )
            raise AudienceReviewRuntimeError(
                "review_edit_completion_failed"
            ) from None
        self._assert_mutation_lease(operation.lifecycle_lease)
        if self._after_command_resume_hook is not None:
            await self._after_command_resume_hook()
            self._assert_mutation_lease(operation.lifecycle_lease)
        await self._stabilize_and_reconcile(
            record,
            operation.lifecycle_lease,
        )

    def _active_edit(self, record: _RunRecord) -> ActiveAnalystEditSnapshot:
        if record.state is None:
            raise AudienceReviewRuntimeError("review_state_unavailable")
        try:
            return ActiveAnalystEditSnapshot.model_validate_json(
                dumps(record.state.get("active_edit"))
            )
        except Exception:
            raise AudienceReviewRuntimeError(
                "review_edit_operation_unavailable"
            ) from None

    def _cleanup_edit_operation(self, command_id: str) -> None:
        operation = self._edit_operations.get(command_id)
        if operation is None:
            return
        if (
            operation.provider_task is not None
            and operation.provider_task.done()
            and operation.graph_task is not None
            and operation.graph_task.done()
        ):
            self._edit_operations.pop(command_id, None)
            self._release_mutation_lease(operation.lifecycle_lease)

    async def expire_due_runs(self) -> int:
        """Explicitly resume every due paused run with the internal command."""
        lease = self._acquire_mutation_lease("expiry_scan")
        try:
            expired_count = 0
            for run_id in tuple(self._runs):
                self._assert_mutation_lease(lease)
                record = self._runs[run_id]
                async with record.lock:
                    self._assert_mutation_lease(lease)
                    if await self._expire_if_due(record, lease=lease):
                        self._assert_mutation_lease(lease)
                        expired_count += 1
            return expired_count
        finally:
            self._release_mutation_lease(lease)

    def _require_run(self, run_id: str) -> _RunRecord:
        record = self._runs.get(run_id)
        if record is None:
            raise AudienceReviewConflictError(ReviewConflictCode.RUN_NOT_FOUND)
        return record

    def _require_open(self) -> None:
        if self._lifecycle != "open":
            raise AudienceReviewRuntimeError("review_runtime_closed")

    def _can_commit(
        self,
        operation: _EditOperation,
    ) -> bool:
        if operation.abandoned:
            return False
        return self._mutation_lease_is_valid(operation.lifecycle_lease)

    def _abandon_operation(self, operation: _EditOperation) -> None:
        operation.abandoned = True
        reservation = operation.record.receipts.get(operation.command_id)
        if (
            reservation is not None
            and reservation.payload_digest == operation.command_digest
            and reservation.receipt is None
        ):
            operation.record.receipts.pop(operation.command_id, None)
        self._release_mutation_lease(operation.lifecycle_lease)

    def _clear_incomplete_command_reservations(self) -> None:
        for record in self._runs.values():
            incomplete = tuple(
                command_id
                for command_id, reservation in record.receipts.items()
                if reservation.receipt is None
            )
            for command_id in incomplete:
                record.receipts.pop(command_id, None)

    async def _expire_if_due(
        self,
        record: _RunRecord,
        *,
        lease: _LifecycleLease,
        stabilize: bool = True,
    ) -> bool:
        if stabilize:
            await self._stabilize_and_reconcile(record, lease)
            self._assert_mutation_lease(lease)
        if record.failed:
            return False
        result = self._result(record)
        now = self._safe_now()
        if result.status != "pending_review" or now < record.expires_at:
            return False
        internal_command = ExpireReviewRunCommand(
            type="expire_run",
            run_id=record.run_id,
            expired_at=now.isoformat(),
        )
        try:
            self._assert_mutation_lease(lease)
            await self._ainvoke_review_graph(
                Command(resume=internal_command.model_dump(mode="json")),
                record=record,
                lease=lease,
            )
            self._assert_mutation_lease(lease)
            await self._stabilize_and_reconcile(record, lease)
            self._assert_mutation_lease(lease)
        except BaseException:
            if self._mutation_lease_is_valid(lease):
                await self._best_effort_stabilize_and_reconcile(
                    record,
                    lease,
                )
            raise
        return True

    async def _stabilize_and_reconcile(
        self,
        record: _RunRecord,
        lease: _LifecycleLease,
    ) -> None:
        await self._stabilize(record, lease)
        self._assert_mutation_lease(lease)
        self._reconcile_receipt(record)
        self._assert_mutation_lease(lease)
        self._persist_record(record)
        self._cleanup_reconciled_edit_operation(record)

    async def _stabilize(
        self,
        record: _RunRecord,
        lease: _LifecycleLease,
    ) -> None:
        """Advance committed internal work to the next interrupt or END."""
        for continuation_count in range(MAX_STABILIZATION_CONTINUATIONS + 1):
            snapshot = await self._read_snapshot(record)
            self._assert_mutation_lease(lease)
            record.state = dict(snapshot.values)
            try:
                is_stable = self._is_stable_boundary(record, snapshot)
            except AudienceReviewRuntimeError:
                raise
            except Exception:
                raise AudienceReviewRuntimeError(
                    "review_stabilization_failed"
                ) from None
            if is_stable:
                return
            if (
                not snapshot.next
                or not set(snapshot.next).issubset(_POST_COMMAND_NODES)
                or len(snapshot.tasks) != len(snapshot.next)
                or tuple(task.name for task in snapshot.tasks)
                != snapshot.next
                or any(task.error is not None for task in snapshot.tasks)
            ):
                raise AudienceReviewRuntimeError(
                    "review_stabilization_failed"
                )
            if continuation_count == MAX_STABILIZATION_CONTINUATIONS:
                break
            try:
                self._assert_mutation_lease(lease)
                await self._ainvoke_review_graph(
                    None,
                    record=record,
                    lease=lease,
                )
                self._assert_mutation_lease(lease)
            except asyncio.CancelledError:
                raise
            except Exception:
                raise AudienceReviewRuntimeError(
                    "review_stabilization_failed"
                ) from None
        raise AudienceReviewRuntimeError("review_stabilization_failed")

    def _is_stable_boundary(
        self,
        record: _RunRecord,
        snapshot: StateSnapshot,
    ) -> bool:
        result = self._result(record)
        if not snapshot.next:
            if (
                snapshot.interrupts
                or snapshot.tasks
                or not result.is_complete
                or any(
                    candidate.status in {"queued", "pending_review", "editing"}
                    for candidate in result.review_candidates
                )
            ):
                raise AudienceReviewRuntimeError(
                    "review_stabilization_failed"
                )
            return True
        if snapshot.next == ("regenerate_and_finalize_edit",):
            if (
                snapshot.interrupts
                or len(snapshot.tasks) != 1
                or snapshot.tasks[0].name
                != "regenerate_and_finalize_edit"
                or snapshot.tasks[0].error is not None
                or snapshot.tasks[0].interrupts
                or result.status != "editing"
                or sum(
                    candidate.status == "editing"
                    for candidate in result.review_candidates
                )
                != 1
                or result.pending_review is not None
            ):
                raise AudienceReviewRuntimeError(
                    "review_stabilization_failed"
                )
            self._active_edit(record)
            return True
        if not snapshot.interrupts:
            return False
        if (
            snapshot.next != ("await_analyst_command",)
            or len(snapshot.interrupts) != 1
            or len(snapshot.tasks) != 1
            or snapshot.tasks[0].name != "await_analyst_command"
            or snapshot.tasks[0].error is not None
            or len(snapshot.tasks[0].interrupts) != 1
            or result.status != "pending_review"
            or result.pending_review is None
            or sum(
                candidate.status == "pending_review"
                for candidate in result.review_candidates
            )
            != 1
        ):
            raise AudienceReviewRuntimeError(
                "review_stabilization_failed"
            )
        try:
            interrupt_payload = PendingAudienceReview.model_validate_json(
                dumps(snapshot.interrupts[0].value)
            )
            task_interrupt_payload = PendingAudienceReview.model_validate_json(
                dumps(snapshot.tasks[0].interrupts[0].value)
            )
        except Exception:
            raise AudienceReviewRuntimeError(
                "review_stabilization_failed"
            ) from None
        if (
            interrupt_payload != result.pending_review
            or task_interrupt_payload != result.pending_review
            or snapshot.tasks[0].interrupts[0] != snapshot.interrupts[0]
        ):
            raise AudienceReviewRuntimeError("review_stabilization_failed")
        return True

    def _reconcile_receipt(self, record: _RunRecord) -> None:
        if record.state is None:
            raise AudienceReviewRuntimeError("review_state_unavailable")
        raw_applied = record.state.get("last_applied_command")
        if raw_applied is None:
            return
        applied = AppliedReviewCommandSnapshot.model_validate_json(
            dumps(raw_applied)
        )
        reservation = record.receipts.get(applied.command_id)
        if reservation is None:
            return
        if reservation.payload_digest != applied.command_digest:
            raise AudienceReviewConflictError(
                ReviewConflictCode.COMMAND_ID_REUSED
            )
        if reservation.receipt is not None:
            return
        result = self._result(record)
        reservation.receipt = ReviewCommandReceipt(
            run_id=record.run_id,
            review_id=applied.review_id,
            cluster_id=applied.cluster_id,
            command_id=applied.command_id,
            command_type=applied.command_type,
            resulting_status=applied.resulting_status,
            run_status=result.status,
        )

    def _durable_run_record(self, record: _RunRecord) -> DurableRunRecord:
        result = self._result(record)
        current_version = max(
            (candidate.version for candidate in result.review_candidates),
            default=0,
        )
        return DurableRunRecord(
            run_id=record.run_id,
            thread_id=record.thread_id,
            created_at=record.created_at.isoformat(),
            expires_at=record.expires_at.isoformat(),
            start_request_digest=record.start_request_digest,
            status=result.status,
            current_version=current_version,
            terminal=result.is_complete,
            failed=record.failed,
        )

    def _persist_record(self, record: _RunRecord) -> None:
        """Publish only safe index/receipt data after authoritative graph state."""
        store = self._durable_store
        if store is None:
            return
        try:
            durable = self._durable_run_record(record)
            completed = tuple(
                (command_id, reservation)
                for command_id, reservation in record.receipts.items()
                if reservation.receipt is not None
            )
            if not completed:
                store.save_run(durable)
                return
            for command_id, reservation in completed:
                assert reservation.receipt is not None
                store.save_run_and_receipt(
                    durable,
                    DurableReceiptRecord(
                        run_id=record.run_id,
                        command_id=command_id,
                        command_digest=reservation.payload_digest,
                        receipt_json=reservation.receipt.model_dump_json(),
                        completed_at=self._safe_now().isoformat(),
                    ),
                )
        except AudienceReviewRuntimeError:
            raise
        except Exception:
            raise AudienceReviewRuntimeError(
                "review_state_unavailable"
            ) from None

    def _cleanup_reconciled_edit_operation(self, record: _RunRecord) -> None:
        """Release only a checkpoint-proven terminal edit operation."""
        if record.state is None or record.state.get("active_edit") is not None:
            return
        raw_applied = record.state.get("last_applied_command")
        if raw_applied is None:
            return
        try:
            applied = AppliedReviewCommandSnapshot.model_validate_json(
                dumps(raw_applied)
            )
        except Exception:
            return
        if applied.command_type != "edit_recommendation":
            return
        operation = self._edit_operations.get(applied.command_id)
        reservation = record.receipts.get(applied.command_id)
        if (
            operation is None
            or operation.record is not record
            or operation.command_digest != applied.command_digest
            or reservation is None
            or reservation.payload_digest != applied.command_digest
            or reservation.receipt is None
        ):
            return
        result = self._result(record)
        matching_candidates = tuple(
            candidate
            for candidate in result.review_candidates
            if candidate.review_id == applied.review_id
            and candidate.cluster_id == applied.cluster_id
            and candidate.terminal_command_id == applied.command_id
            and candidate.status == applied.resulting_status
        )
        if len(matching_candidates) != 1:
            return
        self._cleanup_edit_operation(applied.command_id)

    async def _best_effort_stabilize_and_reconcile(
        self,
        record: _RunRecord,
        lease: _LifecycleLease,
    ) -> None:
        if not self._mutation_lease_is_valid(lease):
            return
        try:
            await self._stabilize_and_reconcile(record, lease)
        except BaseException:
            if not self._mutation_lease_is_valid(lease):
                return
            try:
                snapshot = await self._read_snapshot(record)
                self._assert_mutation_lease(lease)
                record.state = dict(snapshot.values)
            except BaseException:
                return

    async def _read_state(self, record: _RunRecord) -> AudienceReviewState:
        return dict((await self._read_snapshot(record)).values)

    async def _read_snapshot(self, record: _RunRecord) -> StateSnapshot:
        try:
            snapshot = await self._graph.aget_state(
                self._config(record.thread_id)
            )
        except Exception:
            raise AudienceReviewConflictError(
                ReviewConflictCode.CHECKPOINT_NOT_FOUND
            ) from None
        if not snapshot.values or "run_id" not in snapshot.values:
            raise AudienceReviewConflictError(
                ReviewConflictCode.CHECKPOINT_NOT_FOUND
            )
        return snapshot

    def _result(self, record: _RunRecord) -> ReviewRunResult:
        if record.state is None:
            raise AudienceReviewRuntimeError("review_state_unavailable")
        result = build_review_run_result(
            record.state,
            thread_id=record.thread_id,
            failed=record.failed,
        )
        if record.failed and result.failure_code is None:
            result = result.model_copy(
                update={"failure_code": "review_projection_failed"}
            )
        return result.model_copy(
            update={
                "created_at": record.created_at.isoformat(),
                "expires_at": record.expires_at.isoformat(),
            }
        )

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return value.astimezone(UTC)

    def _safe_now(self) -> datetime:
        try:
            return self._now()
        except Exception:
            raise AudienceReviewRuntimeError("invalid_review_clock") from None

    async def _discard_thread(
        self,
        *,
        lease: _LifecycleLease,
        run_id: str,
        thread_id: str,
        checkpoint_ns: str = "",
    ) -> bool:
        """Revoke exact provisional ownership and atomically delete its thread."""
        with self._persistence_gate:
            active = self._active_operations.get(lease.operation_id)
            ownership = (
                active.cleanup_threads.get(thread_id)
                if active is not None
                else None
            )
            expected = _CleanupThreadOwnership(
                run_id=run_id,
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
            )
            if (
                active is None
                or active.lease != lease
                or ownership != expected
                or run_id in self._runs
            ):
                return False
            active.abandoned = True
            self._active_operations.pop(lease.operation_id, None)
            try:
                self._fenced_checkpointer._delete_thread_administratively(
                    thread_id
                )
            except Exception:
                # Revocation still prevents orphan writes after failed cleanup.
                return False
            return True

    def _context(
        self,
        record: _RunRecord,
        *,
        provider: AudienceGenerationProvider | None = None,
    ) -> AudienceReviewWorkflowContext:
        return AudienceReviewWorkflowContext(
            provider=record.provider if provider is None else provider,
            edit_executor=self,
        )

    @staticmethod
    def _config(thread_id: str) -> dict[str, dict[str, str]]:
        return {"configurable": {"thread_id": thread_id}}


def _payload_digest(payload: dict[str, object]) -> str:
    canonical = dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def _safe_edit_provider_failure() -> AnalystEditProviderResult:
    return AnalystEditProviderResult(
        status="provider_failed",
        response=None,
        elapsed_seconds=0,
        usage=None,
    )


def _consume_task_result(task: asyncio.Task[object]) -> None:
    """Retrieve a detached task result without logging private failures."""
    try:
        task.exception()
    except (asyncio.CancelledError, Exception):
        return


def _require_match(
    actual: object,
    expected: object,
    code: ReviewConflictCode,
) -> None:
    if actual != expected:
        raise AudienceReviewConflictError(code)
