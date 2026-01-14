"""Tests for CSV status tracker."""

import csv
from pathlib import Path

import pytest

from apex_harness.status_tracker import CSVStatusTracker


def test_status_tracker_initialization(tmp_path):
    """Test CSV tracker initialization."""
    csv_path = tmp_path / "status.csv"
    tracker = CSVStatusTracker(csv_path)

    assert csv_path.exists()

    # Verify headers
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == [
            "task_name",
            "status",
            "start_time",
            "end_time",
            "duration_seconds",
            "return_code",
            "error_message",
        ]


def test_status_tracker_update_new_task(tmp_path):
    """Test adding a new task status."""
    csv_path = tmp_path / "status.csv"
    tracker = CSVStatusTracker(csv_path)

    tracker.update_status("task1", "pending")

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    assert rows[0]["task_name"] == "task1"
    assert rows[0]["status"] == "pending"


def test_status_tracker_update_existing_task(tmp_path):
    """Test updating an existing task."""
    csv_path = tmp_path / "status.csv"
    tracker = CSVStatusTracker(csv_path)

    tracker.update_status("task1", "pending")
    tracker.update_status(
        "task1",
        "completed",
        start_time="2026-01-14 10:00:00",
        end_time="2026-01-14 10:05:00",
        duration_seconds="300",
        return_code="0",
    )

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    assert rows[0]["task_name"] == "task1"
    assert rows[0]["status"] == "completed"
    assert rows[0]["duration_seconds"] == "300"
    assert rows[0]["return_code"] == "0"


def test_status_tracker_multiple_tasks(tmp_path):
    """Test tracking multiple tasks."""
    csv_path = tmp_path / "status.csv"
    tracker = CSVStatusTracker(csv_path)

    tracker.update_status("task1", "pending")
    tracker.update_status("task2", "in_progress")
    tracker.update_status("task3", "completed")

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 3
    assert {r["task_name"] for r in rows} == {"task1", "task2", "task3"}


def test_status_tracker_thread_safety(tmp_path):
    """Test thread-safe operations (basic smoke test)."""
    from threading import Thread

    csv_path = tmp_path / "status.csv"
    tracker = CSVStatusTracker(csv_path)

    def update_task(task_name):
        tracker.update_status(task_name, "completed")

    threads = [Thread(target=update_task, args=(f"task{i}",)) for i in range(10)]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 10
