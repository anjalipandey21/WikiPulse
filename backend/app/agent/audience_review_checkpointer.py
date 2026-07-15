"""Private lifecycle fencing for review checkpoint persistence."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import Lock
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
)
from langgraph.checkpoint.memory import InMemorySaver


class ReviewPersistenceFenceRejected(asyncio.CancelledError):
    """Private cancellation signal for a stale graph invocation."""


@dataclass(slots=True)
class _PersistenceInvocationCapability:
    authority: object
    rejected: bool = False


@dataclass(frozen=True, slots=True)
class ReviewSaverInspection:
    """Detached structural data for tests without saver mutation authority."""

    storage: object
    writes: object
    blobs: object
    checkpoints: tuple[CheckpointTuple, ...]
    decoded_blobs: tuple[object, ...]


def _freeze_structure(value: object) -> object:
    """Copy saver containers into deterministic immutable tuple structures."""
    if isinstance(value, dict):
        return tuple(
            (_freeze_structure(key), _freeze_structure(item))
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_structure(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_structure(item) for item in value)
    return deepcopy(value)


class LifecycleFencedCheckpointer(BaseCheckpointSaver):
    """Fence asynchronous LangGraph writes with a process-local capability."""

    def __init__(
        self,
        delegate: BaseCheckpointSaver,
        *,
        persistence_gate: Lock,
        is_authorized: Callable[[object], bool],
    ) -> None:
        super().__init__(serde=delegate.serde)
        self._delegate = delegate
        self._persistence_gate = persistence_gate
        self._is_authorized = is_authorized
        self._capability: ContextVar[
            _PersistenceInvocationCapability | None
        ] = ContextVar(
            "wikipulse_review_persistence_capability",
            default=None,
        )

    @contextmanager
    def bind(
        self,
        authority: object,
    ) -> Iterator[_PersistenceInvocationCapability]:
        """Bind one non-serializable invocation capability to child tasks."""
        capability = _PersistenceInvocationCapability(authority=authority)
        token = self._capability.set(capability)
        try:
            yield capability
        finally:
            self._capability.reset(token)

    async def _before_persist(self, _kind: str) -> None:
        """Private no-op seam used by deterministic persistence race tests."""

    def _persist(self, operation: Callable[[], Any]) -> Any:
        capability = self._capability.get()
        # Supported review savers expose synchronous mutation methods. There
        # is deliberately no await while this gate is held, so an abandoned
        # coroutine cannot retain persistence authority during shutdown.
        with self._persistence_gate:
            if capability is None or not self._is_authorized(
                capability.authority
            ):
                if capability is not None:
                    capability.rejected = True
                raise ReviewPersistenceFenceRejected
            return operation()

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Fail closed if synchronous graph persistence is attempted."""
        raise ReviewPersistenceFenceRejected

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Fail closed if synchronous pending-write persistence is attempted."""
        raise ReviewPersistenceFenceRejected

    def delete_thread(self, thread_id: str) -> None:
        """Normal graph execution cannot use administrative deletion."""
        raise ReviewPersistenceFenceRejected

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        await self._before_persist("checkpoint")
        return self._persist(
            lambda: self._delegate.put(
                config,
                checkpoint,
                metadata,
                new_versions,
            )
        )

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        await self._before_persist("pending_writes")
        self._persist(
            lambda: self._delegate.put_writes(
                config,
                writes,
                task_id,
                task_path,
            )
        )

    async def adelete_thread(self, thread_id: str) -> None:
        """Normal graph execution cannot use administrative deletion."""
        raise ReviewPersistenceFenceRejected

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        if not isinstance(self._delegate, InMemorySaver):
            return self._delegate.get_tuple(config)
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        thread_storage = self._delegate.storage.get(thread_id)
        if thread_storage is None:
            return None
        namespace_storage = thread_storage.get(checkpoint_ns)
        if not namespace_storage:
            return None
        checkpoint_id = get_checkpoint_id(config)
        if checkpoint_id:
            saved = namespace_storage.get(checkpoint_id)
            if saved is None:
                return None
        else:
            checkpoint_id = max(namespace_storage)
            saved = namespace_storage[checkpoint_id]
        checkpoint, metadata, parent_checkpoint_id = saved
        writes = self._delegate.writes.get(
            (thread_id, checkpoint_ns, checkpoint_id),
            {},
        ).values()
        checkpoint_value: Checkpoint = self._delegate.serde.loads_typed(
            checkpoint
        )
        return CheckpointTuple(
            config=(
                config
                if get_checkpoint_id(config)
                else {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                    }
                }
            ),
            checkpoint={
                **checkpoint_value,
                "channel_values": self._delegate._load_blobs(
                    thread_id,
                    checkpoint_ns,
                    checkpoint_value["channel_versions"],
                ),
            },
            metadata=self._delegate.serde.loads_typed(metadata),
            pending_writes=[
                (task_id, channel, self._delegate.serde.loads_typed(value))
                for task_id, channel, value, _ in writes
            ],
            parent_config=(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": parent_checkpoint_id,
                    }
                }
                if parent_checkpoint_id
                else None
            ),
        )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        return iter(
            self._list_checkpoints(
                config,
                filter=filter,
                before=before,
                limit=limit,
            )
        )

    def _list_checkpoints(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> tuple[CheckpointTuple, ...]:
        if not isinstance(self._delegate, InMemorySaver):
            return tuple(
                self._delegate.list(
                    config,
                    filter=filter,
                    before=before,
                    limit=limit,
                )
            )
        if config is None:
            thread_ids = tuple(self._delegate.storage)
            config_checkpoint_ns = None
        else:
            thread_id = config["configurable"]["thread_id"]
            if thread_id not in self._delegate.storage:
                return ()
            thread_ids = (thread_id,)
            config_checkpoint_ns = config["configurable"].get(
                "checkpoint_ns"
            )
        config_checkpoint_id = get_checkpoint_id(config) if config else None
        before_checkpoint_id = get_checkpoint_id(before) if before else None
        result: list[CheckpointTuple] = []
        remaining = limit
        for thread_id in thread_ids:
            thread_storage = self._delegate.storage.get(thread_id)
            if thread_storage is None:
                continue
            for checkpoint_ns, namespace_storage in thread_storage.items():
                if (
                    config_checkpoint_ns is not None
                    and checkpoint_ns != config_checkpoint_ns
                ):
                    continue
                for checkpoint_id, (
                    checkpoint,
                    metadata_value,
                    parent_checkpoint_id,
                ) in sorted(
                    namespace_storage.items(),
                    key=lambda item: item[0],
                    reverse=True,
                ):
                    if (
                        config_checkpoint_id
                        and checkpoint_id != config_checkpoint_id
                    ):
                        continue
                    if (
                        before_checkpoint_id
                        and checkpoint_id >= before_checkpoint_id
                    ):
                        continue
                    metadata = self._delegate.serde.loads_typed(
                        metadata_value
                    )
                    if filter and not all(
                        value == metadata.get(key)
                        for key, value in filter.items()
                    ):
                        continue
                    if remaining is not None and remaining <= 0:
                        return tuple(result)
                    if remaining is not None:
                        remaining -= 1
                    writes = self._delegate.writes.get(
                        (thread_id, checkpoint_ns, checkpoint_id),
                        {},
                    ).values()
                    checkpoint_value: Checkpoint = (
                        self._delegate.serde.loads_typed(checkpoint)
                    )
                    result.append(
                        CheckpointTuple(
                            config={
                                "configurable": {
                                    "thread_id": thread_id,
                                    "checkpoint_ns": checkpoint_ns,
                                    "checkpoint_id": checkpoint_id,
                                }
                            },
                            checkpoint={
                                **checkpoint_value,
                                "channel_values": self._delegate._load_blobs(
                                    thread_id,
                                    checkpoint_ns,
                                    checkpoint_value["channel_versions"],
                                ),
                            },
                            metadata=metadata,
                            parent_config=(
                                {
                                    "configurable": {
                                        "thread_id": thread_id,
                                        "checkpoint_ns": checkpoint_ns,
                                        "checkpoint_id": parent_checkpoint_id,
                                    }
                                }
                                if parent_checkpoint_id
                                else None
                            ),
                            pending_writes=[
                                (
                                    task_id,
                                    channel,
                                    self._delegate.serde.loads_typed(value),
                                )
                                for task_id, channel, value, _ in writes
                            ],
                        )
                    )
        return tuple(result)

    async def aget_tuple(
        self,
        config: RunnableConfig,
    ) -> CheckpointTuple | None:
        return self.get_tuple(config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        for item in self.list(
            config,
            filter=filter,
            before=before,
            limit=limit,
        ):
            yield item

    def _inspect(
        self,
        config: RunnableConfig | None = None,
    ) -> ReviewSaverInspection:
        """Return detached saver contents without exposing the delegate."""
        with self._persistence_gate:
            checkpoints = deepcopy(self._list_checkpoints(config))
            if not isinstance(self._delegate, InMemorySaver):
                return ReviewSaverInspection(
                    storage=(),
                    writes=(),
                    blobs=(),
                    checkpoints=checkpoints,
                    decoded_blobs=(),
                )
            decoded_blobs = tuple(
                deepcopy(self._delegate.serde.loads_typed(value))
                for value in self._delegate.blobs.values()
                if value[0] != "empty"
            )
            return ReviewSaverInspection(
                storage=_freeze_structure(self._delegate.storage),
                writes=_freeze_structure(self._delegate.writes),
                blobs=_freeze_structure(self._delegate.blobs),
                checkpoints=checkpoints,
                decoded_blobs=decoded_blobs,
            )

    def _delete_thread_administratively(self, thread_id: str) -> None:
        """Delete one revoked thread; caller must hold persistence gate."""
        self._delegate.delete_thread(thread_id)

    def _close_administratively(self) -> None:
        """Close an owned durable saver connection when one exists."""
        connection = getattr(self._delegate, "conn", None)
        if connection is not None:
            connection.close()

    @property
    def config_specs(self) -> list:
        return self._delegate.config_specs

    def get_next_version(self, current: Any, channel: Any) -> Any:
        return self._delegate.get_next_version(current, channel)
