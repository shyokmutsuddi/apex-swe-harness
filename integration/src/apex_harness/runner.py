"""
Task runner orchestration.

Provides sequential and parallel execution strategies.
"""

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

from apex_harness.models import ModelConfig
from apex_harness.status_tracker import CSVStatusTracker
from apex_harness.task_executor import run_task_command


def run_sequential(
    tasks: List[str],
    timestamp: str,
    model_config: ModelConfig,
    tracker: Optional[CSVStatusTracker] = None,
    dry_run: bool = False,
) -> List[Tuple[str, int, str, str]]:
    """
    Run tasks sequentially.

    Args:
        tasks: List of task names to execute
        timestamp: Timestamp for this execution run
        model_config: Model configuration
        tracker: CSV status tracker
        dry_run: If True, only print commands without executing

    Returns:
        List of (task_name, return_code, stdout, stderr) tuples
    """
    print(f"\nRunning {len(tasks)} tasks sequentially...\n")

    # Initialize all tasks as pending
    if tracker and not dry_run:
        for task in tasks:
            tracker.update_status(task, "pending")

    results = []
    for i, task in enumerate(tasks, 1):
        print(f"\n{'=' * 80}")
        print(f"Task {i}/{len(tasks)}: {task}")
        print(f"{'=' * 80}")

        task_name, return_code, stdout, stderr = run_task_command(
            task, timestamp, model_config, tracker, dry_run
        )
        results.append((task_name, return_code, stdout, stderr))

        if not dry_run:
            if stdout:
                print(stdout)
            if stderr:
                print(stderr, file=sys.stderr)

            if return_code == 0:
                print(f"✓ Task {task} completed successfully")
            else:
                print(
                    f"✗ Task {task} failed with return code {return_code}",
                    file=sys.stderr,
                )

    return results


def run_parallel(
    tasks: List[str],
    timestamp: str,
    model_config: ModelConfig,
    max_workers: int,
    tracker: Optional[CSVStatusTracker] = None,
    dry_run: bool = False,
) -> List[Tuple[str, int, str, str]]:
    """
    Run tasks in parallel using ThreadPoolExecutor.

    Args:
        tasks: List of task names to execute
        timestamp: Timestamp for this execution run
        model_config: Model configuration
        max_workers: Maximum number of parallel workers
        tracker: CSV status tracker
        dry_run: If True, only print commands without executing

    Returns:
        List of (task_name, return_code, stdout, stderr) tuples
    """
    print(
        f"\nRunning {len(tasks)} tasks in parallel (max workers: {max_workers})...\n"
    )

    # Initialize all tasks as pending
    if tracker and not dry_run:
        for task in tasks:
            tracker.update_status(task, "pending")

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(
                run_task_command, task, timestamp, model_config, tracker, dry_run
            ): task
            for task in tasks
        }

        # Process completed tasks
        for i, future in enumerate(as_completed(future_to_task), 1):
            task = future_to_task[future]
            try:
                task_name, return_code, stdout, stderr = future.result()
                results.append((task_name, return_code, stdout, stderr))

                if not dry_run:
                    print(f"\n[{i}/{len(tasks)}] Completed: {task_name}")
                    if stdout:
                        preview = (
                            stdout[:200] + "..." if len(stdout) > 200 else stdout
                        )
                        print(f"  Output: {preview}")

                    if return_code == 0:
                        print("  ✓ Success")
                    else:
                        print(
                            f"  ✗ Failed with return code {return_code}",
                            file=sys.stderr,
                        )
                        if stderr:
                            error_preview = (
                                stderr[:200] + "..." if len(stderr) > 200 else stderr
                            )
                            print(f"  Error: {error_preview}", file=sys.stderr)
            except Exception as e:
                print(f"ERROR processing {task}: {str(e)}", file=sys.stderr)
                results.append((task, -1, "", str(e)))

    return results
