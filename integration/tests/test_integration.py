"""
Integration tests for backward compatibility and regression validation.

These tests verify that the refactored code produces identical outputs to the original.
"""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from apex_harness.models import get_model_config
from apex_harness.task_executor import run_task_command


@pytest.mark.integration
def test_command_generation_matches_legacy():
    """Verify generated commands match original scripts."""
    model_config = get_model_config("claude")
    task_name = "test-task"
    timestamp = "20260114-120000"

    # We'll capture the command that would be run
    with patch("apex_harness.task_executor.subprocess.run") as mock_run:
        with patch("apex_harness.task_executor.cleanup_docker"):
            mock_run.return_value = MagicMock(
                returncode=0, stdout="", stderr=""
            )

            run_task_command(task_name, timestamp, model_config, dry_run=False)

            # Verify the command matches legacy format
            call_args = mock_run.call_args[0][0]
        expected = [
            "apx",
            "reports",
            "run",
            f"claude-{task_name}-{timestamp}",
            "--tasks",
            task_name,
            "--models",
            "claude-sonnet-4-5-20250929",
            "--n-trials",
            "3",
            "--max-workers",
            "3",
            "--timeout",
            "3600",
        ]

        assert call_args == expected


@pytest.mark.integration
def test_xai_command_generation():
    """Verify XAI/Grok command generation matches legacy."""
    model_config = get_model_config("xai")
    task_name = "test-task"
    timestamp = "20260114-120000"

    with patch("apex_harness.task_executor.subprocess.run") as mock_run:
        with patch("apex_harness.task_executor.cleanup_docker"):
            mock_run.return_value = MagicMock(
                returncode=0, stdout="", stderr=""
            )

            run_task_command(task_name, timestamp, model_config, dry_run=False)

            call_args = mock_run.call_args[0][0]

        # Verify report prefix is "grok" not "xai" (matches legacy)
        assert call_args[3] == f"grok-{task_name}-{timestamp}"
        assert call_args[7] == "xai/grok-4"


@pytest.mark.integration
def test_qwen_command_generation():
    """Verify Qwen/Fireworks command generation matches legacy."""
    model_config = get_model_config("qwen")
    task_name = "test-task"
    timestamp = "20260114-120000"

    with patch("apex_harness.task_executor.subprocess.run") as mock_run:
        with patch("apex_harness.task_executor.cleanup_docker"):
            mock_run.return_value = MagicMock(
                returncode=0, stdout="", stderr=""
            )

            run_task_command(task_name, timestamp, model_config, dry_run=False)

            call_args = mock_run.call_args[0][0]

        # Verify report prefix is "fireworks" (matches legacy qwen script)
        assert call_args[3] == f"fireworks-{task_name}-{timestamp}"
        # Verify max_workers is 2 for qwen (different from default 3)
        assert call_args[11] == "2"


@pytest.mark.integration
def test_csv_output_format_matches_legacy(tmp_path):
    """Verify CSV status file format matches legacy."""
    from apex_harness.status_tracker import CSVStatusTracker

    csv_path = tmp_path / "test_status.csv"
    tracker = CSVStatusTracker(csv_path)

    tracker.update_status(
        "task1",
        "completed",
        start_time="2026-01-14 10:00:00",
        end_time="2026-01-14 10:05:00",
        duration_seconds="300.5",
        return_code="0",
        error_message="",
    )

    # Read and verify CSV structure matches legacy
    import csv

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    row = rows[0]

    # Verify all legacy fields are present
    assert "task_name" in row
    assert "status" in row
    assert "start_time" in row
    assert "end_time" in row
    assert "duration_seconds" in row
    assert "return_code" in row
    assert "error_message" in row

    # Verify values
    assert row["task_name"] == "task1"
    assert row["status"] == "completed"
    assert row["duration_seconds"] == "300.5"


@pytest.mark.integration
def test_docker_cleanup_is_called():
    """Verify Docker cleanup is invoked after task execution."""
    model_config = get_model_config("claude")

    with patch("apex_harness.task_executor.subprocess.run") as mock_run:
        with patch("apex_harness.task_executor.cleanup_docker") as mock_cleanup:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="", stderr=""
            )

            run_task_command("task1", "timestamp", model_config, dry_run=False)

            # Verify cleanup was called
            mock_cleanup.assert_called_once()


@pytest.mark.integration
def test_timeout_handling_matches_legacy():
    """Verify timeout handling behavior matches legacy."""
    model_config = get_model_config("claude")

    with patch("apex_harness.task_executor.subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired("apx", 3600)

        task_name, return_code, stdout, stderr = run_task_command(
            "task1", "timestamp", model_config, dry_run=False
        )

        assert return_code == -1
        assert "timed out" in stderr.lower()
