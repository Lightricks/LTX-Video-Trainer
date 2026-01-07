"""Queue worker that processes training jobs."""

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from scripts.jobs.database import JobDatabase, JobStatus

logger = logging.getLogger(__name__)


class QueueWorker:
    """Worker that processes training jobs from the queue."""

    def __init__(self, db_path: Path, check_interval: int = 5):
        """Initialize the queue worker.

        Args:
            db_path: Path to the job database
            check_interval: Seconds between checking for new jobs
        """
        self.db = JobDatabase(db_path)
        self.check_interval = check_interval
        self.running = False
        self.current_process: Optional[subprocess.Popen] = None
        self.current_job_id: Optional[int] = None

        self.shutdown_signal_file = db_path.parent / ".worker_shutdown_signal"
        if self.shutdown_signal_file.exists():
            self.shutdown_signal_file.unlink()

        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum: int, _frame: object) -> None:
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop()

    def start(self) -> None:
        """Start the worker loop."""
        self.running = True
        logger.info("Queue worker started")

        while self.running:
            try:
                if self.shutdown_signal_file.exists():
                    logger.info("Shutdown signal detected, stopping worker...")
                    self.stop()
                    break

                job = self.db.get_next_pending_job()

                if job:
                    self._process_job(job)
                else:
                    time.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Error in worker loop: {e}", exc_info=True)
                time.sleep(self.check_interval)

        logger.info("Queue worker stopped")

        if self.shutdown_signal_file.exists():
            self.shutdown_signal_file.unlink()

    def stop(self) -> None:
        """Stop the worker loop."""
        self.running = False

        if self.current_process and self.current_process.poll() is None:
            logger.info(f"Cancelling current job {self.current_job_id}")
            self.current_process.terminate()
            try:
                self.current_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.current_process.kill()

            if self.current_job_id:
                self.db.update_job_status(self.current_job_id, JobStatus.CANCELLED, error_message="Worker shutdown")

    def _process_job(self, job: dict) -> None:
        """Process a single training job.

        Args:
            job: Job dictionary from database
        """
        job_id = job["id"]
        self.current_job_id = job_id

        logger.info(f"Starting job {job_id} for dataset: {job['dataset_name']}")

        self.db.update_job_status(job_id, JobStatus.RUNNING)

        try:
            cmd = self._build_training_command(job)

            self.current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            self.db.update_job_status(job_id, JobStatus.RUNNING, process_id=self.current_process.pid)

            log_lines = []
            for line in self.current_process.stdout:
                if not self.running:
                    break

                current_job = self.db.get_job(job_id)
                if current_job and current_job["status"] == JobStatus.CANCELLED:
                    logger.info(f"Job {job_id} was cancelled, terminating process")
                    self.current_process.terminate()
                    try:
                        self.current_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.current_process.kill()
                    break

                log_lines.append(line)

                if len(log_lines) > 500:
                    log_lines = log_lines[-500:]

                if "Step" in line or "steps" in line.lower():
                    self.db.update_job_status(job_id, JobStatus.RUNNING, progress=line.strip(), logs="".join(log_lines))

            returncode = self.current_process.wait()

            final_logs = "".join(log_lines)

            current_job = self.db.get_job(job_id)
            if current_job and current_job["status"] == JobStatus.CANCELLED:
                logger.info(f"Job {job_id} was cancelled")
                self.db.update_job_status(
                    job_id,
                    JobStatus.CANCELLED,
                    logs=final_logs,
                )
            elif returncode == 0:
                logger.info(f"Job {job_id} completed successfully")

                output_dir = self._get_output_dir(job)
                checkpoint_path = self._find_latest_checkpoint(output_dir)

                self.db.update_job_status(
                    job_id,
                    JobStatus.COMPLETED,
                    progress="Training completed",
                    logs=final_logs,
                    output_dir=str(output_dir) if output_dir else None,
                    checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
                )
            else:
                logger.error(f"Job {job_id} failed with return code {returncode}")
                self.db.update_job_status(
                    job_id,
                    JobStatus.FAILED,
                    error_message=f"Training process exited with code {returncode}",
                    logs=final_logs,
                )

        except Exception as e:
            logger.error(f"Error processing job {job_id}: {e}", exc_info=True)
            current_job = self.db.get_job(job_id)
            if current_job and current_job["status"] != JobStatus.CANCELLED:
                self.db.update_job_status(job_id, JobStatus.FAILED, error_message=str(e))

        finally:
            self.current_process = None
            self.current_job_id = None

    def _build_training_command(self, job: dict) -> list[str]:
        """Build the training command from job parameters.

        Args:
            job: Job dictionary

        Returns:
            Command as list of strings
        """
        params = job["params"]

        python_exe = sys.executable

        scripts_dir = Path(__file__).parent.parent
        train_script = scripts_dir / "train_cli.py"

        cmd = [
            python_exe,
            str(train_script),
            "--job-id",
            str(job["id"]),
            "--dataset",
            job["dataset_name"],
            "--model-version",
            params["model_source"],
            "--learning-rate",
            str(params["learning_rate"]),
            "--steps",
            str(params["steps"]),
            "--lora-rank",
            str(params["lora_rank"]),
            "--batch-size",
            str(params["batch_size"]),
            "--width",
            str(params["width"]),
            "--height",
            str(params["height"]),
            "--num-frames",
            str(params["num_frames"]),
            "--validation-prompt",
            params["validation_prompt"],
            "--validation-interval",
            str(params["validation_interval"]),
        ]

        if params.get("id_token"):
            cmd.extend(["--id-token", params["id_token"]])

        if params.get("push_to_hub"):
            cmd.append("--push-to-hub")
            if params.get("hf_model_id"):
                cmd.extend(["--hf-model-id", params["hf_model_id"]])
            if params.get("hf_token"):
                os.environ["HF_TOKEN"] = params["hf_token"]

        return cmd

    def _get_output_dir(self, job: dict) -> Optional[Path]:
        """Get the output directory for a job.

        Args:
            job: Job dictionary

        Returns:
            Output directory path or None
        """
        params = job["params"]
        project_root = Path(__file__).parent.parent.parent
        output_dir = project_root / "outputs" / f"lora_r{params['lora_rank']}_job{job['id']}"

        if output_dir.exists():
            return output_dir
        return None

    def _find_latest_checkpoint(self, output_dir: Optional[Path]) -> Optional[Path]:
        """Find the latest checkpoint in the output directory.

        Args:
            output_dir: Output directory path

        Returns:
            Path to latest checkpoint or None
        """
        if not output_dir or not output_dir.exists():
            return None

        checkpoints_dir = output_dir / "checkpoints"
        if not checkpoints_dir.exists():
            return None

        checkpoints = list(checkpoints_dir.glob("*.safetensors"))
        if not checkpoints:
            return None

        return max(checkpoints, key=lambda p: p.stat().st_mtime)


def main() -> None:
    """Main entry point for the queue worker."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    project_root = Path(__file__).parent.parent.parent
    db_path = project_root / "jobs.db"

    worker = QueueWorker(db_path)

    try:
        worker.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
        worker.stop()


if __name__ == "__main__":
    main()
