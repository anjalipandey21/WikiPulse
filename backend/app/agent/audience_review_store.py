"""Small durable index for process-restart review recovery."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import sqlite3
from threading import Lock


@dataclass(frozen=True, slots=True)
class DurableRunRecord:
    run_id: str
    thread_id: str
    created_at: str
    expires_at: str
    start_request_digest: str | None
    status: str
    current_version: int
    terminal: bool
    failed: bool


@dataclass(frozen=True, slots=True)
class DurableReceiptRecord:
    run_id: str
    command_id: str
    command_digest: str
    receipt_json: str
    completed_at: str


class AudienceReviewDurableStore:
    """Transactional SQLite metadata beside LangGraph checkpoints."""

    def __init__(self, path: str) -> None:
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._lock = Lock()
        self.setup()

    def setup(self) -> None:
        with self._lock, self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS review_starts (
                    run_id TEXT PRIMARY KEY,
                    request_digest TEXT NOT NULL,
                    published INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS review_runs (
                    run_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    start_request_digest TEXT,
                    status TEXT NOT NULL,
                    current_version INTEGER NOT NULL,
                    terminal INTEGER NOT NULL,
                    failed INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS review_receipts (
                    run_id TEXT NOT NULL,
                    command_id TEXT NOT NULL,
                    command_digest TEXT NOT NULL,
                    receipt_json TEXT NOT NULL,
                    completed_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, command_id),
                    FOREIGN KEY (run_id) REFERENCES review_runs(run_id)
                        ON DELETE CASCADE
                );
                """
            )

    def claim_start(
        self,
        run_id: str,
        request_digest: str,
        created_at: str,
    ) -> bool:
        """Persist a provisional identity, returning False for exact replay."""
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT request_digest, published FROM review_starts "
                "WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if row is not None:
                if row[0] != request_digest:
                    raise ValueError("start_digest_conflict")
                return False
            self._connection.execute(
                "INSERT INTO review_starts "
                "(run_id, request_digest, published, created_at) "
                "VALUES (?, ?, 0, ?)",
                (run_id, request_digest, created_at),
            )
            return True

    def release_incomplete_start(self, run_id: str, request_digest: str) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "DELETE FROM review_starts WHERE run_id = ? "
                "AND request_digest = ? AND published = 0",
                (run_id, request_digest),
            )

    def discard_incomplete_starts(self) -> tuple[str, ...]:
        """Remove crash-left provisional starts and return their run IDs."""
        with self._lock, self._connection:
            rows = self._connection.execute(
                "SELECT run_id FROM review_starts WHERE published = 0"
            ).fetchall()
            self._connection.execute(
                "DELETE FROM review_starts WHERE published = 0"
            )
        return tuple(row[0] for row in rows)

    def load_incomplete_starts(self) -> tuple[str, ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT run_id FROM review_starts WHERE published = 0 "
                "ORDER BY created_at, run_id"
            ).fetchall()
        return tuple(row[0] for row in rows)

    def discard_incomplete_start(self, run_id: str) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "DELETE FROM review_starts WHERE run_id = ? AND published = 0",
                (run_id,),
            )

    def save_run(self, record: DurableRunRecord) -> None:
        with self._lock, self._connection:
            self._upsert_run(record)
            if record.start_request_digest is not None:
                self._connection.execute(
                    "INSERT INTO review_starts "
                    "(run_id, request_digest, published, created_at) "
                    "VALUES (?, ?, 1, ?) "
                    "ON CONFLICT(run_id) DO UPDATE SET "
                    "request_digest=excluded.request_digest, published=1",
                    (
                        record.run_id,
                        record.start_request_digest,
                        record.created_at,
                    ),
                )

    def save_run_and_receipt(
        self,
        record: DurableRunRecord,
        receipt: DurableReceiptRecord,
    ) -> None:
        with self._lock, self._connection:
            self._upsert_run(record)
            self._connection.execute(
                "INSERT INTO review_receipts "
                "(run_id, command_id, command_digest, receipt_json, completed_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(run_id, command_id) DO NOTHING",
                (
                    receipt.run_id,
                    receipt.command_id,
                    receipt.command_digest,
                    receipt.receipt_json,
                    receipt.completed_at,
                ),
            )

    def _upsert_run(self, record: DurableRunRecord) -> None:
        self._connection.execute(
            "INSERT INTO review_runs "
            "(run_id, thread_id, created_at, expires_at, "
            "start_request_digest, status, current_version, terminal, failed) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(run_id) DO UPDATE SET "
            "thread_id=excluded.thread_id, expires_at=excluded.expires_at, "
            "status=excluded.status, current_version=excluded.current_version, "
            "terminal=excluded.terminal, failed=excluded.failed",
            (
                record.run_id,
                record.thread_id,
                record.created_at,
                record.expires_at,
                record.start_request_digest,
                record.status,
                record.current_version,
                int(record.terminal),
                int(record.failed),
            ),
        )

    def load_runs(self) -> tuple[DurableRunRecord, ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT run_id, thread_id, created_at, expires_at, "
                "start_request_digest, status, current_version, terminal, failed "
                "FROM review_runs ORDER BY created_at, run_id"
            ).fetchall()
        return tuple(
            DurableRunRecord(
                run_id=row[0],
                thread_id=row[1],
                created_at=row[2],
                expires_at=row[3],
                start_request_digest=row[4],
                status=row[5],
                current_version=row[6],
                terminal=bool(row[7]),
                failed=bool(row[8]),
            )
            for row in rows
        )

    def load_receipts(self, run_id: str) -> tuple[DurableReceiptRecord, ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT run_id, command_id, command_digest, receipt_json, "
                "completed_at FROM review_receipts WHERE run_id = ? "
                "ORDER BY completed_at, command_id",
                (run_id,),
            ).fetchall()
        return tuple(DurableReceiptRecord(*row) for row in rows)

    def close(self) -> None:
        with self._lock:
            self._connection.close()


def utc_text(value: datetime) -> str:
    return value.isoformat()
