#!/usr/bin/env python3
"""
Start both the queue worker and Gradio UI together.

This script launches:
1. Queue worker in the background to process training jobs
2. Gradio UI for user interaction

Both processes will be managed together, and stopping this script will stop both.
"""

import logging
import signal
import subprocess
import sys
import time
from pathlib import Path

# Configure logging for terminal output (alternative to print statements)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",  # Simple format for terminal output
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))


class ServiceManager:
    """Manage worker and UI processes."""

    def __init__(self):
        self.worker_process = None
        self.ui_process = None
        self.running = True

        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, _signum: int, _frame: object) -> None:
        """Handle shutdown signals."""
        self.stop()
        sys.exit(0)

    def start(self) -> bool:
        """Start both worker and UI processes."""
        python_exe = sys.executable
        scripts_dir = Path(__file__).parent

        logger.info("🚀 Starting LTX-Video Trainer with Queue System")
        logger.info("=" * 60)
        logger.info("\n📋 Starting job worker...")

        worker_script = scripts_dir / "jobs" / "run_worker.py"
        self.worker_process = subprocess.Popen(
            [python_exe, str(worker_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        time.sleep(2)

        if self.worker_process.poll() is not None:
            logger.error("❌ Worker failed to start")
            if self.worker_process.stdout:
                logger.error("\n📋 Worker Error Output:")
                logger.error("-" * 60)
                for line in self.worker_process.stdout:
                    logger.error(line.rstrip())
                logger.error("-" * 60)
            return False

        logger.info("✅ Queue worker started")
        logger.info("\n🎨 Starting Gradio UI...")

        ui_script = scripts_dir / "app_gradio_v2.py"
        self.ui_process = subprocess.Popen(
            [python_exe, str(ui_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        time.sleep(3)

        if self.ui_process.poll() is not None:
            logger.error("❌ UI failed to start")
            if self.ui_process.stdout:
                logger.error("\n📋 UI Error Output:")
                logger.error("-" * 60)
                for line in self.ui_process.stdout:
                    logger.error(line.rstrip())
                logger.error("-" * 60)
            self.stop()
            return False

        logger.info("✅ Gradio UI started")
        logger.info("\n" + "=" * 60)
        logger.info("🎉 All services running!")
        logger.info("📍 Open your browser to: http://localhost:7860")
        logger.info("💡 Press Ctrl+C to stop all services")
        logger.info("=" * 60 + "\n")

        return True

    def monitor(self) -> None:
        """Monitor both processes and restart if needed."""
        while self.running:
            if self.worker_process and self.worker_process.poll() is not None:
                logger.error("⚠️  Queue worker stopped unexpectedly")
                if self.worker_process.stdout:
                    remaining = self.worker_process.stdout.read()
                    if remaining:
                        logger.error("\n📋 Worker Error Output:")
                        logger.error("-" * 60)
                        logger.error(remaining)
                        logger.error("-" * 60)
                self.running = False
                break

            if self.ui_process and self.ui_process.poll() is not None:
                logger.error("⚠️  UI stopped unexpectedly")
                if self.ui_process.stdout:
                    remaining = self.ui_process.stdout.read()
                    if remaining:
                        logger.error("\n📋 UI Error Output:")
                        logger.error("-" * 60)
                        logger.error(remaining)
                        logger.error("-" * 60)
                self.running = False
                break

            if self.ui_process and self.ui_process.stdout:
                line = self.ui_process.stdout.readline()
                if line:
                    pass

            if self.worker_process and self.worker_process.stdout:
                line = self.worker_process.stdout.readline()
                if line:
                    pass

            time.sleep(0.1)

    def stop(self) -> None:
        """Stop both processes."""
        logger.info("\n🛑 Shutting down services...")
        self.running = False

        if self.ui_process:
            self.ui_process.terminate()
            try:
                self.ui_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.ui_process.kill()

        if self.worker_process:
            self.worker_process.terminate()
            try:
                self.worker_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.worker_process.kill()


def main() -> None:
    """Main entry point."""
    manager = ServiceManager()

    if not manager.start():
        sys.exit(1)

    try:
        manager.monitor()
    except KeyboardInterrupt:
        pass
    finally:
        manager.stop()


if __name__ == "__main__":
    main()
