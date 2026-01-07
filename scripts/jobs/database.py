"""SQLite database for persistent job queue management."""

import json
import sqlite3
import threading
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class JobStatus(str, Enum):
    """Job status enumeration."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobDatabase:
    """SQLite database for managing training job queue."""

    def __init__(self, db_path: Path):
        """Initialize the job database.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                status TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                params TEXT NOT NULL,
                progress TEXT,
                error_message TEXT,
                logs TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                process_id INTEGER,
                output_dir TEXT,
                checkpoint_path TEXT,
                validation_sample TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_status ON jobs(status)
        """
        )

        conn.commit()

    def create_job(
        self,
        dataset_name: str,
        params: dict[str, Any],
    ) -> int:
        """Create a new training job.

        Args:
            dataset_name: Name of the dataset to train on
            params: Training parameters dictionary

        Returns:
            Job ID
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO jobs (
                status, dataset_name, params, created_at
            ) VALUES (?, ?, ?, ?)
        """,
            (JobStatus.PENDING, dataset_name, json.dumps(params), datetime.now(timezone.utc).isoformat()),
        )

        conn.commit()
        return cursor.lastrowid

    def get_job(self, job_id: int) -> Optional[dict[str, Any]]:
        """Get job by ID.

        Args:
            job_id: Job ID

        Returns:
            Job dictionary or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_dict(row)

    def get_all_jobs(self, status: Optional[JobStatus] = None) -> list[dict[str, Any]]:
        """Get all jobs, optionally filtered by status.

        Args:
            status: Filter by status (optional)

        Returns:
            List of job dictionaries
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        if status:
            cursor.execute("SELECT * FROM jobs WHERE status = ? ORDER BY created_at DESC", (status,))
        else:
            cursor.execute("SELECT * FROM jobs ORDER BY created_at DESC")

        rows = cursor.fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_next_pending_job(self) -> Optional[dict[str, Any]]:
        """Get the next pending job (oldest first).

        Returns:
            Job dictionary or None if no pending jobs
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM jobs
            WHERE status = ?
            ORDER BY created_at ASC
            LIMIT 1
        """,
            (JobStatus.PENDING,),
        )

        row = cursor.fetchone()
        if row is None:
            return None

        return self._row_to_dict(row)

    def update_job_status(
        self,
        job_id: int,
        status: JobStatus,
        progress: Optional[str] = None,
        error_message: Optional[str] = None,
        logs: Optional[str] = None,
        process_id: Optional[int] = None,
        output_dir: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        validation_sample: Optional[str] = None,
    ) -> None:
        """Update job status and related fields.

        Args:
            job_id: Job ID
            status: New status
            progress: Progress message (optional)
            error_message: Error message if failed (optional)
            logs: Training logs (optional)
            process_id: Process ID if running (optional)
            output_dir: Output directory path (optional)
            checkpoint_path: Path to final checkpoint (optional)
            validation_sample: Path to latest validation sample (optional)
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        updates = ["status = ?"]
        values = [status]

        if progress is not None:
            updates.append("progress = ?")
            values.append(progress)

        if error_message is not None:
            updates.append("error_message = ?")
            values.append(error_message)

        if logs is not None:
            updates.append("logs = ?")
            values.append(logs)

        if process_id is not None:
            updates.append("process_id = ?")
            values.append(process_id)

        if output_dir is not None:
            updates.append("output_dir = ?")
            values.append(output_dir)

        if checkpoint_path is not None:
            updates.append("checkpoint_path = ?")
            values.append(checkpoint_path)

        if validation_sample is not None:
            updates.append("validation_sample = ?")
            values.append(validation_sample)

        if status == JobStatus.RUNNING and not self.get_job(job_id).get("started_at"):
            updates.append("started_at = ?")
            values.append(datetime.now(timezone.utc).isoformat())
        elif status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            updates.append("completed_at = ?")
            values.append(datetime.now(timezone.utc).isoformat())

        values.append(job_id)

        cursor.execute(
            f"""
            UPDATE jobs
            SET {", ".join(updates)}
            WHERE id = ?
        """,
            values,
        )

        conn.commit()

    def delete_job(self, job_id: int) -> None:
        """Delete a job from the database.

        Args:
            job_id: Job ID
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        conn.commit()

    def get_job_count(self, status: Optional[JobStatus] = None) -> int:
        """Get count of jobs, optionally filtered by status.

        Args:
            status: Filter by status (optional)

        Returns:
            Job count
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        if status:
            cursor.execute("SELECT COUNT(*) FROM jobs WHERE status = ?", (status,))
        else:
            cursor.execute("SELECT COUNT(*) FROM jobs")

        return cursor.fetchone()[0]

    def clear_completed_jobs(self) -> None:
        """Clear all completed, failed, and cancelled jobs."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            DELETE FROM jobs
            WHERE status IN (?, ?, ?)
        """,
            (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED),
        )

        conn.commit()

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        """Convert SQLite row to dictionary.

        Args:
            row: SQLite row

        Returns:
            Dictionary representation
        """
        data = dict(row)
        if data.get("params"):
            data["params"] = json.loads(data["params"])
        return data

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None
