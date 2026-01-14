"""
Task discovery utilities.

Scans task directories and identifies valid tasks for execution.
"""

import sys
from pathlib import Path


def get_all_tasks(tasks_dir: Path) -> list[str]:
    """
    Get all task directories from the tasks folder.

    Excludes files and special directories like 'shared'.

    Args:
        tasks_dir: Path to the tasks directory

    Returns:
        Sorted list of task names

    Raises:
        SystemExit: If tasks directory doesn't exist
    """
    if not tasks_dir.exists():
        print(f"Error: Tasks directory not found: {tasks_dir}", file=sys.stderr)
        sys.exit(1)

    tasks = []
    for item in sorted(tasks_dir.iterdir()):
        if item.is_dir() and item.name != "shared" and not item.name.startswith("."):
            tasks.append(item.name)

    return tasks
