"""
Result reporting and summary generation.
"""

import sys
from typing import List, Tuple


def print_summary(results: List[Tuple[str, int, str, str]]) -> None:
    """
    Print summary of all task executions.

    Args:
        results: List of (task_name, return_code, stdout, stderr) tuples
    """
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}\n")

    successful = [r for r in results if r[1] == 0]
    failed = [r for r in results if r[1] != 0]

    print(f"Total tasks: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailed tasks:")
        for task_name, return_code, _, stderr in failed:
            error_preview = stderr[:100] + "..." if len(stderr) > 100 else stderr
            print(f"  - {task_name} (return code: {return_code})")
            if error_preview:
                print(f"    Error: {error_preview}")

    if successful:
        print("\nSuccessful tasks:")
        for task_name, _, _, _ in successful:
            print(f"  - {task_name}")


def get_exit_code(results: List[Tuple[str, int, str, str]]) -> int:
    """
    Determine exit code based on task results.

    Args:
        results: List of (task_name, return_code, stdout, stderr) tuples

    Returns:
        0 if all tasks succeeded, 1 otherwise
    """
    failed = [r for r in results if r[1] != 0]
    return 0 if len(failed) == 0 else 1
