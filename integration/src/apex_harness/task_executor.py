"""
Task execution engine.

Handles running individual tasks and collecting results.
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from apex_harness.docker_utils import cleanup_docker
from apex_harness.models import ModelConfig
from apex_harness.status_tracker import CSVStatusTracker


def run_task_command(
    task_name: str,
    timestamp: str,
    model_config: ModelConfig,
    tracker: Optional[CSVStatusTracker] = None,
    dry_run: bool = False,
) -> Tuple[str, int, str, str]:
    """
    Run the apx reports command for a single task.

    Args:
        task_name: Name of the task to run
        timestamp: Timestamp string to include in the report name
        model_config: Model configuration
        tracker: CSV status tracker instance
        dry_run: If True, only print the command without executing

    Returns:
        Tuple of (task_name, return_code, stdout, stderr)
    """
    cmd = [
        "apx",
        "reports",
        "run",
        f"{model_config.report_prefix}-{task_name}-{timestamp}",
        "--tasks",
        task_name,
        "--models",
        model_config.identifier,
        "--n-trials",
        str(model_config.n_trials),
        "--max-workers",
        str(model_config.max_workers),
        "--timeout",
        str(model_config.timeout),
    ]

    cmd_str = " ".join(cmd)
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Running: {cmd_str}")

    if dry_run:
        return task_name, 0, "", ""

    # Track start time
    start_time = datetime.now()
    start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")

    if tracker:
        tracker.update_status(task_name, "in_progress", start_time=start_time_str)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=model_config.timeout + 200,  # Buffer for cleanup
        )

        # Track end time and calculate duration
        end_time = datetime.now()
        end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
        duration = (end_time - start_time).total_seconds()

        # Update status based on result
        status = "completed" if result.returncode == 0 else "failed"
        error_msg = result.stderr[:500] if result.stderr else ""  # Truncate

        if tracker:
            tracker.update_status(
                task_name,
                status,
                start_time=start_time_str,
                end_time=end_time_str,
                duration_seconds=str(round(duration, 2)),
                return_code=str(result.returncode),
                error_message=error_msg,
            )

        # Clean up Docker resources after each task
        cleanup_docker()

        return task_name, result.returncode, result.stdout, result.stderr

    except subprocess.TimeoutExpired:
        error_msg = f"Command timed out after {model_config.timeout} seconds"
        print(f"ERROR [{task_name}]: {error_msg}", file=sys.stderr)

        end_time = datetime.now()
        end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
        duration = (end_time - start_time).total_seconds()

        if tracker:
            tracker.update_status(
                task_name,
                "failed",
                start_time=start_time_str,
                end_time=end_time_str,
                duration_seconds=str(round(duration, 2)),
                return_code="-1",
                error_message=error_msg,
            )

        cleanup_docker()
        return task_name, -1, "", error_msg

    except Exception as e:
        error_msg = f"Exception occurred: {str(e)}"
        print(f"ERROR [{task_name}]: {error_msg}", file=sys.stderr)

        end_time = datetime.now()
        end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
        duration = (end_time - start_time).total_seconds()

        if tracker:
            tracker.update_status(
                task_name,
                "failed",
                start_time=start_time_str,
                end_time=end_time_str,
                duration_seconds=str(round(duration, 2)),
                return_code="-1",
                error_message=error_msg,
            )

        cleanup_docker()
        return task_name, -1, "", error_msg
