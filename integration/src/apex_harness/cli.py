"""
Command-line interface for APEX SWE Harness.

Unified entry point replacing model-specific runner scripts.
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from apex_harness.models import ModelConfig, get_model_config, list_available_models
from apex_harness.reporting import get_exit_code, print_summary
from apex_harness.runner import run_parallel, run_sequential
from apex_harness.status_tracker import CSVStatusTracker
from apex_harness.task_discovery import get_all_tasks


def validate_environment(model_config: ModelConfig) -> None:
    """
    Validate required environment variables for the model.

    Args:
        model_config: Model configuration to validate

    Exits:
        If required environment variable is missing (non-dry-run mode only)
    """
    if model_config.requires_env_var:
        if not os.environ.get(model_config.requires_env_var):
            print(
                f"WARNING: {model_config.requires_env_var} environment variable is not set!",
                file=sys.stderr,
            )
            print(
                f"Set it with: export {model_config.requires_env_var}='your-api-key'",
                file=sys.stderr,
            )


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        prog="apex-runner",
        description="Run apx reports for tasks with specified AI model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tasks sequentially with Claude
  apex-runner --model claude

  # Run specific tasks in parallel with Gemini
  apex-runner --model gemini --tasks task1 task2 --parallel --max-workers 5

  # Dry run to see what would be executed
  apex-runner --model xai --dry-run

  # Custom tasks directory and output location
  apex-runner --model opus --tasks-dir ./my-tasks --artifacts-dir ./my-outputs

Available models: %(models)s
        """
        % {"models": ", ".join(list_available_models())},
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list_available_models(),
        help="AI model to use for evaluation",
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tasks in parallel instead of sequentially",
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of parallel workers (defaults to model-specific value)",
    )

    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Specific tasks to run (by default runs all tasks)",
    )

    parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=None,
        help="Path to tasks directory (default: ./tasks)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )

    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for generated artifacts (default: ./artifacts)",
    )

    parser.add_argument(
        "--status-csv",
        type=Path,
        default=None,
        help="Path to CSV file for tracking task status (auto-generated if not specified)",
    )

    return parser


def main() -> None:
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Get model configuration
    try:
        model_config = get_model_config(args.model)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate environment
    if not args.dry_run:
        validate_environment(model_config)

    # Determine tasks directory
    if args.tasks_dir:
        tasks_dir = args.tasks_dir
    else:
        # Default: look for tasks/ in current directory or parent of script
        tasks_dir = Path.cwd() / "tasks"
        if not tasks_dir.exists():
            # Fallback to temp/tasks for backward compatibility during transition
            tasks_dir = Path.cwd() / "temp" / "tasks"

    # Get tasks to run
    if args.tasks:
        tasks = args.tasks
        print(f"Running specified tasks: {', '.join(tasks)}")
    else:
        tasks = get_all_tasks(tasks_dir)
        print(f"Found {len(tasks)} tasks in {tasks_dir}")

    if not tasks:
        print("No tasks found to run!", file=sys.stderr)
        sys.exit(1)

    # Generate timestamp for this execution
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    print(f"\nModel: {model_config.name}")
    print(f"Execution timestamp: {timestamp}")
    print(
        f"Report names will be: {model_config.report_prefix}-{{task_name}}-{timestamp}"
    )

    # Determine max workers
    max_workers = args.max_workers if args.max_workers else model_config.max_workers

    # Setup artifacts directory
    artifacts_dir = args.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Initialize CSV status tracker
    tracker = None
    if not args.dry_run:
        if args.status_csv:
            csv_path = args.status_csv
        else:
            csv_path = (
                artifacts_dir
                / f"{model_config.status_csv_prefix}_tasks_status_{timestamp}.csv"
            )
        tracker = CSVStatusTracker(csv_path)
        print(f"Status tracking CSV: {csv_path}")

    # Run tasks
    if args.parallel:
        results = run_parallel(
            tasks, timestamp, model_config, max_workers, tracker, args.dry_run
        )
    else:
        results = run_sequential(
            tasks, timestamp, model_config, tracker, args.dry_run
        )

    # Print summary
    print_summary(results)

    # Exit with appropriate code
    sys.exit(get_exit_code(results))


if __name__ == "__main__":
    main()
