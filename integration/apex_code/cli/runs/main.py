"""Runs CLI - Evaluation harness management."""

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated

from typer import Argument, Option, Typer, confirm

from ...harness import EvaluationConfig, EvaluationExecutor, ModelType, create_llm
from ..datasets import RegistryClient
from ..utils import (
    DEFAULT_DIRS,
    create_dir_option,
    create_table,
    format_timestamp,
    load_json_file,
    rich_print,
)


def _apply_task_filters(
    tasks_dir,
    dataset_name,
    dataset_version,
    dataset_info,
    task_subset,
    category,
    difficulty,
    tags,
):
    """Apply all task filtering logic and return filtered task list."""
    # Parse tags if provided
    tag_list = None
    if tags:
        tag_list = [tag.strip() for tag in tags.split(",")]
        rich_print(f"Tags filter: {', '.join(tag_list)}")

    # Show other filters
    if category:
        rich_print(f"Category filter: {category}")
    if difficulty:
        rich_print(f"Difficulty filter: {difficulty}")

    # Get available tasks from the dataset
    available_tasks = []
    if tasks_dir.exists():
        for item in tasks_dir.iterdir():
            if item.is_dir() and (item / "task.yaml").exists():
                available_tasks.append(item.name)

    filtered_tasks = available_tasks.copy()

    # Apply metadata filtering first (category, difficulty, tags)
    if category or difficulty or tag_list:
        from ..datasets.models import DatasetInfo

        temp_dataset = DatasetInfo(
            name=dataset_name,
            version=dataset_version,
            github_url=dataset_info.github_url,
            dataset_path=dataset_info.dataset_path,
        )

        metadata_filtered = temp_dataset.filter_tasks_by_metadata(
            tasks_dir, category=category, difficulty=difficulty, tags=tag_list
        )
        filtered_tasks = metadata_filtered
        rich_print(f"Metadata filtered tasks: {', '.join(filtered_tasks)}")

    # Apply task subset filtering (glob patterns)
    if task_subset:
        subset_patterns = [pattern.strip() for pattern in task_subset.split(",")]
        rich_print(f"Task subset: {', '.join(subset_patterns)}")

        subset_filtered = []
        for task_id in filtered_tasks:
            for pattern in subset_patterns:
                import fnmatch

                if fnmatch.fnmatch(task_id, pattern):
                    subset_filtered.append(task_id)
                    break

        filtered_tasks = subset_filtered
        rich_print(f"Subset filtered tasks: {', '.join(filtered_tasks)}")

    rich_print(f"Final filtered tasks: {', '.join(filtered_tasks)}")
    return filtered_tasks


def _print_no_tasks_found(task_subset, category, difficulty, tags):
    """Print error message when no tasks match filters."""
    rich_print("[red]No tasks found matching filters[/red]")
    if task_subset:
        rich_print(f"[red]Task subset: {task_subset}[/red]")
    if category:
        rich_print(f"[red]Category: {category}[/red]")
    if difficulty:
        rich_print(f"[red]Difficulty: {difficulty}[/red]")
    if tags:
        tag_list = [tag.strip() for tag in tags.split(",")]
        rich_print(f"[red]Tags: {', '.join(tag_list)}[/red]")


def _handle_multiple_tasks(filtered_tasks):
    """Handle case when multiple tasks are found."""
    if len(filtered_tasks) > 1:
        rich_print(
            f"[yellow]Warning: Multiple tasks found ({len(filtered_tasks)}). Running first task: {filtered_tasks[0]}[/yellow]"
        )
        rich_print("[yellow]To run all tasks, use separate evaluation runs.[/yellow]")


runs_app = Typer(no_args_is_help=True)

# Default runs directory
DEFAULT_RUNS_DIR = DEFAULT_DIRS["runs"]


# Common option decorator
def runs_dir_option() -> Annotated[Path, Option]:
    return create_dir_option("runs", "runs-dir", "Runs directory")


# Helper functions
def _load_metadata(run_dir: Path) -> dict | None:
    """Load metadata from run directory."""
    return load_json_file(run_dir / "metadata.json")


def _get_run_dirs(runs_dir: Path) -> list[Path]:
    """Get all valid run directories (direct children only, not experiment subdirs)."""
    # Only get direct children of runs_dir that have metadata.json
    # and don't start with "experiment_"
    run_dirs = []
    for item in runs_dir.iterdir():
        if (
            item.is_dir()
            and not item.name.startswith("experiment_")
            and (item / "metadata.json").exists()
        ):
            run_dirs.append(item)
    return run_dirs


@runs_app.command()
def run(
    task_id: Annotated[str, Argument(help="Task ID to run")],
    model: Annotated[
        str,
        Option("--model", "-m", help="Model to use for evaluation"),
    ] = "claude-sonnet-4-20250514",
    dataset: Annotated[
        str | None,
        Option("--dataset", help="Dataset to evaluate (format: name==version)"),
    ] = None,
    registry_url: Annotated[
        str | None,
        Option("--registry-url", help="Custom registry URL"),
    ] = None,
    local_registry_path: Annotated[
        str | None,
        Option("--local-registry-path", help="Path to local registry file"),
    ] = None,
    task_subset: Annotated[
        str | None,
        Option(
            "--task-subset", help="Comma-separated task IDs or glob patterns to run"
        ),
    ] = None,
    category: Annotated[
        str | None,
        Option("--category", help="Filter tasks by category"),
    ] = None,
    difficulty: Annotated[
        str | None,
        Option("--difficulty", help="Filter tasks by difficulty"),
    ] = None,
    tags: Annotated[
        str | None,
        Option("--tags", help="Comma-separated tags to filter by"),
    ] = None,
    exclude: Annotated[
        str | None,
        Option(
            "--exclude", help="Comma-separated task IDs or glob patterns to exclude"
        ),
    ] = None,
    n_tasks: Annotated[
        int | None,
        Option("--n-tasks", help="Limit number of tasks to run"),
    ] = None,
    direct_evaluation: Annotated[
        bool,
        Option(
            "--direct",
            help="Evaluate directly from remote dataset without local download",
        ),
    ] = False,
    runs_dir: runs_dir_option() = DEFAULT_RUNS_DIR,
    max_trials: Annotated[
        int,
        Option("--max-trials", "-n", help="Maximum number of trials"),
    ] = 3,
    timeout: Annotated[
        int,
        Option("--timeout", "-t", help="Timeout per trial in seconds"),
    ] = 1800,
    max_workers: Annotated[
        int,
        Option("--max-workers", "-w", help="Maximum number of parallel workers"),
    ] = 4,
    max_steps: Annotated[
        int | None,
        Option("--max-steps", help="Maximum steps per trial (default: unlimited)"),
    ] = None,
    todo_tool_enabled: Annotated[
        bool,
        Option("--todo-tool-enabled", help="Enable the todo tool for task management"),
    ] = False,
    resume_from: Annotated[
        str | None,
        Option("--resume-from", "-r", help="Resume from a previous run ID"),
    ] = None,
):
    """Run evaluation on a task or dataset."""
    # Use the same timestamp format as ApexLogger to ensure directory consistency
    import os

    base_timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    process_id = os.getpid()
    timestamp = f"{base_timestamp}-{process_id}"
    run_id = timestamp

    if resume_from:
        rich_print(f"[blue]Resuming evaluation run: {run_id}[/blue]")
        rich_print(f"Resuming from: {resume_from}")
    else:
        rich_print(f"[blue]Starting evaluation run: {run_id}[/blue]")

    rich_print(f"Task: {task_id}")
    rich_print(f"Model: {model}")
    rich_print(f"Max trials: {max_trials}")
    rich_print(f"Max workers: {max_workers}")
    rich_print(f"Timeout: {timeout}s")

    # Handle dataset evaluation
    tasks_dir = DEFAULT_DIRS["tasks"]
    if dataset:
        rich_print(f"Dataset: {dataset}")
        try:
            # Parse dataset specification (format: name==version)
            if "==" in dataset:
                dataset_name, dataset_version = dataset.split("==", 1)
            else:
                dataset_name = dataset
                dataset_version = "head"

            # Determine registry source
            if local_registry_path:
                # Use local registry file
                local_path = Path(local_registry_path)
                if not local_path.exists():
                    rich_print(
                        f"[red]Error: Local registry file not found: {local_path}[/red]"
                    )
                    return
                registry_source = f"file://{local_path.absolute()}"
                rich_print(f"Using local registry: {local_path}")
            elif registry_url:
                # Use custom registry URL
                registry_source = registry_url
                rich_print(f"Using custom registry: {registry_url}")
            else:
                # Use default registry
                from ..datasets.constants import DEFAULT_REGISTRY_URL

                registry_source = DEFAULT_REGISTRY_URL
                rich_print(f"Using default registry: {DEFAULT_REGISTRY_URL}")

            # Initialize registry client
            registry_client = RegistryClient(registry_source)

            # Get dataset info
            dataset_info = registry_client.get_dataset(dataset_name, dataset_version)
            if not dataset_info:
                rich_print(
                    f"[red]Error: Dataset '{dataset_name}=={dataset_version}' not found in registry[/red]"
                )
                return

            rich_print(f"Found dataset: {dataset_info.name} v{dataset_info.version}")
            rich_print(f"Description: {dataset_info.description}")

            # Handle dataset download
            if dataset_info.github_url.startswith("file://"):
                # Handle local file URLs - just use the local path
                local_path = Path(
                    dataset_info.github_url[7:]
                )  # Remove "file://" prefix
                dataset_source_path = local_path / dataset_info.dataset_path
                if dataset_source_path.exists():
                    # Copy to temporary directory
                    import tempfile

                    temp_dir = Path(tempfile.mkdtemp(prefix="apex_dataset_"))
                    tasks_dir = temp_dir / "tasks"
                    shutil.copytree(dataset_source_path, tasks_dir)
                    rich_print(f"Using local dataset from: {dataset_source_path}")
                else:
                    rich_print(
                        f"[red]Local dataset path not found: {dataset_source_path}[/red]"
                    )
                    return
            else:
                # Download dataset to temporary directory
                import tempfile

                temp_dir = Path(tempfile.mkdtemp(prefix="apex_dataset_"))
                tasks_dir = registry_client.download_dataset(
                    dataset_name, dataset_version, output_dir=temp_dir, overwrite=True
                )
                rich_print(f"Downloaded dataset to: {tasks_dir}")

        except Exception as e:
            rich_print(f"[red]Error handling dataset: {e}[/red]")
            return

    # Handle direct evaluation from remote dataset
    if direct_evaluation and dataset:
        try:
            rich_print(
                f"[blue]Running direct evaluation on remote dataset: {dataset_name}=={dataset_version}[/blue]"
            )

            # Parse filters
            task_subset_list = None
            if task_subset:
                task_subset_list = [
                    pattern.strip() for pattern in task_subset.split(",")
                ]

            tag_list = None
            if tags:
                tag_list = [tag.strip() for tag in tags.split(",")]

            # Show filters
            if task_subset_list:
                rich_print(f"[blue]Task subset: {', '.join(task_subset_list)}[/blue]")
            if category:
                rich_print(f"[blue]Category filter: {category}[/blue]")
            if difficulty:
                rich_print(f"[blue]Difficulty filter: {difficulty}[/blue]")
            if tag_list:
                rich_print(f"[blue]Tags filter: {', '.join(tag_list)}[/blue]")

            # Progress callback
            def progress_callback(current: int, total: int, task_id: str):
                rich_print(
                    f"[yellow]Evaluating task {current}/{total}: {task_id}[/yellow]"
                )

            # Parse exclusions if provided
            exclude_list = None
            if exclude:
                exclude_list = [pattern.strip() for pattern in exclude.split(",")]

            # Run direct evaluation
            result = registry_client.evaluate_dataset_directly(
                name=dataset_name,
                version=dataset_version,
                task_subset=task_subset_list,
                category=category,
                difficulty=difficulty,
                tags=tag_list,
                exclude=exclude_list,
                n_tasks=n_tasks,
                agent_type=model,
                max_trials=max_trials,
                evaluation_callback=progress_callback,
            )

            # Display results
            if "error" in result:
                rich_print(f"[red]Direct evaluation failed: {result['error']}[/red]")
                return

            # Show summary
            rich_print("\n[green]Direct evaluation completed![/green]")
            rich_print(f"[green]Dataset: {result['dataset']}[/green]")
            rich_print(f"[green]Total tasks: {result['total_tasks']}[/green]")
            rich_print(f"[green]Successful: {result['successful']}[/green]")
            rich_print(f"[green]Failed: {result['failed']}[/green]")

            # Save results to run directory (use timestamp-based directory)
            run_dir = runs_dir / timestamp
            run_dir.mkdir(parents=True, exist_ok=True)

            # Save evaluation results
            results_file = run_dir / "direct_evaluation_results.json"
            with open(results_file, "w") as f:
                json.dump(result, f, indent=2)

            rich_print(f"[green]Results saved to: {results_file}[/green]")
            return

        except Exception as e:
            rich_print(f"[red]Error in direct evaluation: {e}[/red]")
            return

    # Handle task filtering (subset, category, difficulty, tags)
    if (task_subset or category or difficulty or tags) and dataset:
        try:
            filtered_tasks = _apply_task_filters(
                tasks_dir,
                dataset_name,
                dataset_version,
                dataset_info,
                task_subset,
                category,
                difficulty,
                tags,
            )

            if not filtered_tasks:
                _print_no_tasks_found(task_subset, category, difficulty, tags)
                return

            _handle_multiple_tasks(filtered_tasks)
            task_id = filtered_tasks[0]
            rich_print(f"Running task: {task_id}")

        except Exception as e:
            rich_print(f"[red]Error filtering tasks: {e}[/red]")
            return

    run_dir = runs_dir / timestamp  # Use timestamp-based directory
    run_dir.mkdir(parents=True, exist_ok=True)

    # Check if task is marked as not runnable
    import yaml

    task_yaml_path = tasks_dir / task_id / "task.yaml"
    if task_yaml_path.exists():
        try:
            with open(task_yaml_path) as f:
                task_data = yaml.safe_load(f)
                if task_data.get("not_runnable", False):
                    reason = task_data.get(
                        "not_runnable_reason",
                        "This task cannot be run with the evaluation harness.",
                    )
                    rich_print(f"[red]❌ Error: Task '{task_id}' cannot be run.[/red]")
                    rich_print(f"[red]Reason: {reason}[/red]")
                    rich_print("")
                    rich_print(
                        "[yellow]This appears to be a data generation task for human talent.[/yellow]"
                    )
                    rich_print(
                        "[yellow]If you want to evaluate a model on this issue, use the corresponding[/yellow]"
                    )
                    rich_print(
                        "[yellow]-observability variant (not -genobservability).[/yellow]"
                    )
                    return
        except Exception:
            pass  # If YAML parsing fails, assume runnable

    try:
        # Convert model string to ModelType
        try:
            model_type = ModelType(model)
        except ValueError:
            rich_print(
                f"[red]Error: Unknown model '{model}'. Available: {[m.value for m in ModelType]}[/red]"
            )
            return

        # Create evaluation config
        config = EvaluationConfig(
            run_id=run_id,
            task_id=task_id,
            model=model_type,
            max_trials=max_trials,
            timeout=timeout,
            tasks_dir=tasks_dir,
            runs_dir=runs_dir,
            max_steps=max_steps,
            todo_tool_enabled=todo_tool_enabled,
            created_at=datetime.now(),
        )

        # Create executor with LLM factory
        # Note: Oracle model is created later by executor when task context is available
        model_type = ModelType(model)
        if model == "oracle":
            llm_factory = {}
        else:
            llm_factory = {model_type: create_llm(model)}
        executor = EvaluationExecutor(llm_factory, max_workers=max_workers)

        # Run evaluation
        if resume_from:
            rich_print("[yellow]Resuming evaluation...[/yellow]")
        else:
            rich_print(
                "[yellow]Running evaluation...(first build might take a few minutes)[/yellow]"
            )
        result = executor.execute_run(config, resume_from=resume_from)

        # Save results
        result_file = run_dir / "results.json"
        with open(result_file, "w") as f:
            json.dump(result.model_dump(), f, indent=2, default=str)

        # Update metadata with completion
        metadata = {
            "run_id": timestamp,  # Use timestamp as run_id
            "task_id": task_id,
            "model": model,
            "max_trials": max_trials,
            "timeout": timeout,
            "created_at": config.created_at.isoformat(),
            "completed_at": datetime.now().isoformat(),
            "status": "completed",
            "success_rate": result.success_rate,
            "total_trials": len(result.trials),
        }

        # Add dataset information if used
        if dataset:
            metadata["dataset"] = dataset
            if "dataset_info" in locals():
                metadata["dataset_info"] = {
                    "name": dataset_info.name,
                    "version": dataset_info.version,
                    "description": dataset_info.description,
                    "github_url": dataset_info.github_url,
                }

        with open(run_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Display results
        if result.success_rate == 1.0:
            rich_print("[green]✅ Evaluation completed successfully![/green]")
        else:
            rich_print("[red]❌ Evaluation completed with failures[/red]")

        rich_print(f"Success rate: {result.success_rate:.2%}")
        rich_print(f"Total trials: {len(result.trials)}")
        rich_print(f"Average time: {result.average_time:.2f}s")
        rich_print(f"Results saved: {result_file}")

        # Exit with appropriate code - only exit 0 if 100% success
        if result.success_rate < 1.0:
            sys.exit(1)

    except Exception as e:
        rich_print(f"[red]❌ Evaluation failed: {e}[/red]")

        # Save error metadata
        metadata = {
            "run_id": timestamp,
            "task_id": task_id,
            "model": model,
            "max_trials": max_trials,
            "timeout": timeout,
            "created_at": datetime.now().isoformat(),
            "status": "failed",
            "error": str(e),
        }

        # Add dataset information if used
        if dataset:
            metadata["dataset"] = dataset
            if "dataset_info" in locals():
                metadata["dataset_info"] = {
                    "name": dataset_info.name,
                    "version": dataset_info.version,
                    "description": dataset_info.description,
                    "github_url": dataset_info.github_url,
                }

        with open(run_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        raise

    finally:
        # Clean up temporary dataset directory if used
        if dataset and "temp_dir" in locals():
            try:
                shutil.rmtree(temp_dir)
                rich_print("[blue]Cleaned up temporary dataset directory[/blue]")
            except Exception as e:
                rich_print(
                    f"[yellow]Warning: Failed to clean up temporary directory: {e}[/yellow]"
                )


@runs_app.command()
def list(
    runs_dir: runs_dir_option() = DEFAULT_RUNS_DIR,
):
    """List all runs and experiments."""
    # Get regular runs
    run_dirs = _get_run_dirs(runs_dir)

    # Get experiment runs
    experiment_dirs = [
        d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("experiment_")
    ]

    all_runs = []

    # Process regular runs
    for run_dir in run_dirs:
        metadata = _load_metadata(run_dir)
        if metadata:
            # Check if processes are actually running
            if metadata.get("status") == "running":
                # Check if directory has results.json
                results_file = run_dir / "results.json"
                if results_file.exists():
                    metadata["status"] = "completed"
                else:
                    # Check if it was interrupted (no results but metadata shows running)
                    metadata["status"] = "exited"

            metadata["type"] = "run"
            metadata["dir_name"] = run_dir.name
            all_runs.append(metadata)

    # Process experiment runs
    for exp_dir in experiment_dirs:
        metadata_file = exp_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
                metadata["type"] = "experiment"
                metadata["run_id"] = metadata.get(
                    "experiment_id", exp_dir.name.replace("experiment_", "")
                )
                metadata["dir_name"] = exp_dir.name

                # Check actual status
                if metadata.get("status") == "running":
                    results_file = exp_dir / "results.json"
                    if results_file.exists():
                        metadata["status"] = "completed"
                    else:
                        metadata["status"] = "exited"

                all_runs.append(metadata)
            except:
                pass

    if not all_runs:
        rich_print("[yellow]No runs or experiments found[/yellow]")
        return

    all_runs.sort(key=lambda x: x.get("created_at", ""), reverse=True)

    # Create table
    table = create_table(
        "Runs and Experiments",
        [
            ("Run ID", "cyan"),
            ("Type", "magenta"),
            ("Task/Tasks", "green"),
            ("Agent/Models", "blue"),
            ("Status", "yellow"),
            ("Created", "white"),
        ],
    )

    for run in all_runs:
        if run["type"] == "experiment":
            # For experiments
            tasks = run.get("tasks", [])
            tasks_str = (
                ", ".join(tasks[:2]) + ("..." if len(tasks) > 2 else "")
                if tasks
                else "unknown"
            )
            models = run.get("models", [])
            models_str = (
                ", ".join(models[:1]) + ("..." if len(models) > 1 else "")
                if models
                else "unknown"
            )
            run_id = run.get("run_id", "unknown")
        else:
            # For regular runs
            tasks_str = run.get("task_id", "unknown")
            models_str = run.get("agent", run.get("model", "unknown"))
            run_id = run.get("run_id", "unknown")

        # Color status appropriately
        status = run.get("status", "unknown")
        if status == "completed":
            status_display = "[green]completed[/green]"
        elif status == "exited":
            status_display = "[red]exited[/red]"
        elif status == "failed":
            status_display = "[red]failed[/red]"
        elif status == "running":
            status_display = "[yellow]running[/yellow]"
        else:
            status_display = status

        table.add_row(
            run_id,
            run["type"],
            tasks_str,
            models_str,
            status_display,
            format_timestamp(run.get("created_at", "")),
        )

    rich_print(table)


@runs_app.command()
def status(
    run_id: Annotated[
        str, Argument(help="Run ID (timestamp format: YYYY-MM-DD__HH-MM-SS)")
    ],
    runs_dir: runs_dir_option() = DEFAULT_RUNS_DIR,
):
    """Check status of a specific run."""
    run_dir = runs_dir / run_id

    if not run_dir.exists():
        rich_print(f"[red]Run '{run_id}' not found[/red]")
        return

    metadata = _load_metadata(run_dir)
    if not metadata:
        rich_print(f"[red]No metadata found for run '{run_id}'[/red]")
        return

    rich_print(f"[bold]Run Status: {run_id}[/bold]")
    rich_print(f"Task: {metadata.get('task_id', 'unknown')}")
    rich_print(f"Agent: {metadata.get('agent', 'unknown')}")
    rich_print(f"Status: {metadata.get('status', 'unknown')}")
    rich_print(f"Max Trials: {metadata.get('max_trials', 'unknown')}")
    rich_print(f"Timeout: {metadata.get('timeout', 'unknown')}s")
    rich_print(f"Created: {format_timestamp(metadata.get('created_at', ''))}")

    # Check for results
    results_file = run_dir / "results.json"
    if results_file.exists():
        rich_print(f"[green]Results available: {results_file}[/green]")
    else:
        rich_print("[yellow]No results yet[/yellow]")


@runs_app.command()
def results(
    run_id: Annotated[str, Argument(help="Run ID or Experiment ID")],
    runs_dir: runs_dir_option() = DEFAULT_RUNS_DIR,
):
    """Show results for a specific run or experiment."""
    # Try regular run directory first
    run_dir = runs_dir / run_id

    # If not found, try experiment directory with both formats
    if not run_dir.exists():
        # Handle both experiment_xxx and xxx formats
        clean_id = (
            run_id.replace("experiment_", "")
            if run_id.startswith("experiment_")
            else run_id
        )
        run_dir = runs_dir / f"experiment_{clean_id}"

    if not run_dir.exists():
        rich_print(f"[red]Run or experiment '{run_id}' not found[/red]")
        return

    results_file = run_dir / "results.json"
    if not results_file.exists():
        rich_print(f"[yellow]No results found for '{run_id}'[/yellow]")
        return

    try:
        with open(results_file) as f:
            results = json.load(f)

        # Check if this is an experiment or regular run
        if run_dir.name.startswith("experiment_"):
            rich_print(f"[bold]Results for Experiment: {run_id}[/bold]")
        else:
            rich_print(f"[bold]Results for Run: {run_id}[/bold]")

        # Print JSON without markdown code blocks
        rich_print(json.dumps(results, indent=2))

    except (json.JSONDecodeError, KeyError) as e:
        rich_print(f"[red]Error reading results: {e}[/red]")


@runs_app.command()
def clean(
    runs_dir: runs_dir_option() = DEFAULT_RUNS_DIR,
    force: Annotated[
        bool,
        Option("--force", "-f", help="Force cleanup without confirmation"),
    ] = False,
):
    """Clean up old runs."""
    run_dirs = _get_run_dirs(runs_dir)
    if not run_dirs:
        rich_print("[yellow]No runs to clean[/yellow]")
        return

    rich_print(f"[yellow]Found {len(run_dirs)} runs to clean[/yellow]")

    if not force:
        confirm = (
            input("Are you sure you want to clean all runs? (y/N): ").lower().strip()
        )
        if confirm != "y":
            rich_print("[yellow]Cleanup cancelled[/yellow]")
            return

    for run_dir in run_dirs:
        shutil.rmtree(run_dir)

    rich_print(f"[green]Cleaned {len(run_dirs)} runs[/green]")


@runs_app.command()
def cache_cleanup():
    """Clean up Docker cache images."""
    try:
        from ...harness.docker_manager import get_docker_pool

        # Get Docker client from pool
        pool = get_docker_pool()
        client = pool.get_connection()

        try:
            # Get all apex-cache images
            cache_images = client.images.list(filters={"reference": "apex-cache:*"})

            if not cache_images:
                rich_print("[green]No cache images found[/green]")
                return

            rich_print(f"[yellow]Found {len(cache_images)} cache images:[/yellow]")
            for image in cache_images:
                tag = image.tags[0] if image.tags else image.id[:12]
                size = image.attrs.get("Size", 0) / (1024**2)  # Convert to MB
                rich_print(f"  • {tag} ({size:.1f} MB)")

            # Confirm deletion
            if not confirm(f"Delete {len(cache_images)} cache images?"):
                rich_print("[yellow]Cache cleanup cancelled[/yellow]")
                return

            # Delete cache images
            deleted_count = 0
            for image in cache_images:
                try:
                    client.images.remove(image.id, force=True)
                    deleted_count += 1
                    rich_print(
                        f"[green]Removed: {image.tags[0] if image.tags else image.id[:12]}[/green]"
                    )
                except Exception as e:
                    rich_print(f"[red]Failed to remove {image.id}: {e}[/red]")

            rich_print(f"[green]Cleaned up {deleted_count} cache images[/green]")

        finally:
            pool.return_connection(client)

    except Exception as e:
        rich_print(f"[red]Cache cleanup failed: {e}[/red]")


@runs_app.command()
def cache_status():
    """Show Docker cache status and metrics."""
    try:
        from ...harness.docker_manager import get_docker_pool

        # Get Docker client from pool
        pool = get_docker_pool()
        client = pool.get_connection()

        try:
            # Get cache images
            cache_images = client.images.list(filters={"reference": "apex-cache:*"})

            rich_print("[bold]Docker Cache Status[/bold]")
            rich_print(f"Cache images: {len(cache_images)}")

            if cache_images:
                total_size = 0
                rich_print("\n[bold]Cached Images:[/bold]")
                for image in cache_images:
                    tag = image.tags[0] if image.tags else image.id[:12]
                    size = image.attrs.get("Size", 0) / (1024**2)  # Convert to MB
                    total_size += size
                    created = image.attrs.get("Created", "Unknown")
                    rich_print(f"  • {tag} ({size:.1f} MB) - {created}")

                rich_print(f"\n[bold]Total cache size: {total_size:.1f} MB[/bold]")

            # Show connection pool metrics
            metrics = pool.get_metrics()
            rich_print("\n[bold]Connection Pool Metrics:[/bold]")
            rich_print(
                f"  • Pool size: {metrics['pool_size']}/{metrics['max_connections']}"
            )
            rich_print(f"  • Hit rate: {metrics['hit_rate']:.1%}")
            rich_print(f"  • Connections created: {metrics['connections_created']}")
            rich_print(
                f"  • Health checks performed: {metrics['health_checks_performed']}"
            )

        finally:
            pool.return_connection(client)

    except Exception as e:
        rich_print(f"[red]Failed to get cache status: {e}[/red]")
