"""
Thread-safe status tracking for task execution.

Provides CSV-based progress tracking with atomic updates.
"""

import csv
import threading
from pathlib import Path
from typing import Any, Dict


class CSVStatusTracker:
    """Thread-safe CSV status tracker for task execution."""

    def __init__(self, csv_path: Path):
        """
        Initialize status tracker.

        Args:
            csv_path: Path to CSV file for status tracking
        """
        self.csv_path = csv_path
        self.lock = threading.Lock()
        self.fieldnames = [
            "task_name",
            "status",
            "start_time",
            "end_time",
            "duration_seconds",
            "return_code",
            "error_message",
        ]
        self._initialize_csv()

    def _initialize_csv(self) -> None:
        """Initialize the CSV file with headers."""
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def _update_task_row(self, task_data: Dict[str, str]) -> None:
        """
        Update or append a task row in the CSV.

        Args:
            task_data: Dictionary containing task information
        """
        with self.lock:
            # Read all existing rows
            rows = []
            try:
                with open(self.csv_path, newline="") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
            except FileNotFoundError:
                pass

            # Find and update the task row if it exists
            found = False
            for row in rows:
                if row["task_name"] == task_data["task_name"]:
                    row.update(task_data)
                    found = True
                    break

            # If not found, append new row
            if not found:
                rows.append(task_data)

            # Write back all rows
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
                writer.writerows(rows)

    def update_status(
        self, task_name: str, status: str, **kwargs: Any
    ) -> None:
        """
        Update task status in the CSV.

        Args:
            task_name: Name of the task
            status: Status value (pending, in_progress, completed, failed)
            **kwargs: Additional fields (start_time, end_time, return_code, error_message)
        """
        task_data = {
            "task_name": task_name,
            "status": status,
            "start_time": kwargs.get("start_time", ""),
            "end_time": kwargs.get("end_time", ""),
            "duration_seconds": kwargs.get("duration_seconds", ""),
            "return_code": kwargs.get("return_code", ""),
            "error_message": kwargs.get("error_message", ""),
        }
        self._update_task_row(task_data)
