"""Reports CLI - Advanced experiment management and parallel execution."""

import json
import queue
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from typer import Argument, Option, Typer

from ...harness import EvaluationConfig, EvaluationExecutor, ModelType, create_llm
from ..utils import (
    DEFAULT_DIRS,
    create_dir_option,
    create_table,
    rich_print,
)

reports_app = Typer(no_args_is_help=True)
console = Console()

# Default directories
DEFAULT_RUNS_DIR = DEFAULT_DIRS["runs"]
DEFAULT_TASKS_DIR = DEFAULT_DIRS["tasks"]


# Common options
def runs_dir_option() -> Annotated[Path, Option]:
    return create_dir_option("runs", "runs-dir", "Directory to store run results", short_flag="-r")


def tasks_dir_option() -> Annotated[Path, Option]:
    return create_dir_option("tasks", "tasks-dir", "Path to the tasks directory", short_flag="-d")


class ExperimentConfig:
    """Configuration for experiment runs."""

    def __init__(
        self,
        experiment_id: str,
        tasks: list[str],
        models: list[str],
        n_trials: int = 3,
        timeout: int = 1800,
        max_workers: int = 4,
        max_steps: int | None = None,
        todo_tool_enabled: bool = False,
        runs_dir: Path = DEFAULT_RUNS_DIR,
        tasks_dir: Path = DEFAULT_TASKS_DIR,
    ):
        self.experiment_id = experiment_id
        self.tasks = tasks
        self.models = models
        self.n_trials = n_trials
        self.timeout = timeout
        self.max_workers = max_workers
        self.max_steps = max_steps
        self.todo_tool_enabled = todo_tool_enabled
        self.runs_dir = runs_dir
        self.tasks_dir = tasks_dir
        self.created_at = datetime.now()


def _execute_single_run_process(
    config: dict[str, Any], experiment_config: ExperimentConfig
) -> dict[str, Any]:
    """Execute a single task-model combination in a separate process."""
    try:
        # Convert model string to ModelType
        model_type = ModelType(config["model"])

        # Create evaluation config - each run executes exactly 1 trial
        eval_config = EvaluationConfig(
            run_id=config["run_id"],
            task_id=config["task_id"],
            model=model_type,
            max_trials=1,  # Each individual run executes exactly 1 trial
            timeout=experiment_config.timeout,
            tasks_dir=experiment_config.tasks_dir,
            runs_dir=config["experiment_dir"],
            max_steps=experiment_config.max_steps,
            todo_tool_enabled=experiment_config.todo_tool_enabled,
            created_at=datetime.now(),
        )

        # Create executor and run
        # Note: Use max_workers=1 to avoid threading issues with signal-based timeouts
        # Create LLM factory with the specified model
        # Note: Oracle model is created later by executor when task context is available
        model_type = ModelType(config["model"])
        if config["model"] == "oracle":
            llm_factory = {}
        else:
            llm_factory = {model_type: create_llm(config["model"])}
        executor = EvaluationExecutor(llm_factory, max_workers=1)
        result = executor.execute_run(eval_config)

        return {
            "run_id": config["run_id"],
            "task_id": config["task_id"],
            "model": config["model"],
            "trial_number": config["trial_number"],
            "status": "completed",
            "success_rate": result.success_rate,
            "total_trials": len(result.trials),
            "average_time": result.average_time,
            "trials": [trial.model_dump() for trial in result.trials],
        }

    except Exception as e:
        return {
            "run_id": config["run_id"],
            "task_id": config["task_id"],
            "model": config["model"],
            "trial_number": config["trial_number"],
            "status": "failed",
            "error": str(e),
        }


class ExperimentRunner:
    """Manages parallel execution of experiments across multiple tasks and models."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results: dict[str, Any] = {}
        self.progress_queue = queue.Queue()
        self.completed_runs = 0
        self.total_runs = len(config.tasks) * len(config.models) * config.n_trials

    def run_experiment(self) -> dict[str, Any]:
        """Execute the full experiment with parallel execution."""
        experiment_start = time.time()

        # Create experiment directory
        experiment_dir = (
            self.config.runs_dir / f"experiment_{self.config.experiment_id}"
        )
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # Save experiment metadata
        metadata = {
            "experiment_id": self.config.experiment_id,
            "tasks": self.config.tasks,
            "models": self.config.models,
            "n_trials": self.config.n_trials,
            "timeout": self.config.timeout,
            "max_workers": self.config.max_workers,
            "max_steps": self.config.max_steps,
            "created_at": self.config.created_at.isoformat(),
            "status": "running",
            "total_runs": self.total_runs,
        }

        with open(experiment_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Initialize progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            main_task = progress.add_task(
                f"Running experiment {self.config.experiment_id}",
                total=self.total_runs,
                completed=0,
            )

            # Create all run configurations
            run_configs = []
            for task_id in self.config.tasks:
                for model in self.config.models:
                    for trial_num in range(1, self.config.n_trials + 1):
                        # Create descriptive run ID with task, model, and trial number
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        run_id = f"{self.config.experiment_id}_{task_id}_{model}_trial{trial_num:02d}_{timestamp}"
                        run_configs.append(
                            {
                                "run_id": run_id,
                                "task_id": task_id,
                                "model": model,
                                "trial_number": trial_num,
                                "experiment_dir": experiment_dir,
                            }
                        )

            # Execute runs in parallel using processes to avoid signal/threading issues
            with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Submit all tasks
                future_to_config = {
                    executor.submit(
                        _execute_single_run_process, config, self.config
                    ): config
                    for config in run_configs
                }

                # Process completed tasks
                for future in as_completed(future_to_config):
                    config = future_to_config[future]
                    try:
                        result = future.result()
                        # Store result with unique key including trial number
                        result_key = f"{config['task_id']}_{config['model']}_trial{config['trial_number']:02d}"
                        self.results[result_key] = result
                        self.completed_runs += 1

                        # Update progress
                        progress.update(
                            main_task,
                            advance=1,
                            description=f"Completed {config['task_id']} with {config['model']} "
                            f"trial {config['trial_number']} ({self.completed_runs}/{self.total_runs})",
                        )

                    except Exception as e:
                        rich_print(
                            f"[red]Error in {config['task_id']} with {config['model']} trial {config['trial_number']}: {e}[/red]"
                        )
                        result_key = f"{config['task_id']}_{config['model']}_trial{config['trial_number']:02d}"
                        self.results[result_key] = {
                            "error": str(e),
                            "status": "failed",
                            "task_id": config["task_id"],
                            "model": config["model"],
                            "trial_number": config["trial_number"],
                        }
                        self.completed_runs += 1
                        progress.update(main_task, advance=1)

        # Calculate experiment results
        experiment_end = time.time()
        experiment_duration = experiment_end - experiment_start

        # Aggregate results - count runs as successful only if they actually passed tests
        successful_runs = sum(
            1
            for r in self.results.values()
            if r.get("status") != "failed" and r.get("success_rate", 0.0) > 0.0
        )
        failed_runs = self.total_runs - successful_runs

        final_results = {
            "experiment_id": self.config.experiment_id,
            "total_runs": self.total_runs,
            "successful_runs": successful_runs,
            "failed_runs": failed_runs,
            "success_rate": successful_runs / self.total_runs
            if self.total_runs > 0
            else 0,
            "experiment_duration": experiment_duration,
            "results": self.results,
            "completed_at": datetime.now().isoformat(),
        }

        # Save final results
        with open(experiment_dir / "results.json", "w") as f:
            json.dump(final_results, f, indent=2, default=str)

        # Update metadata
        metadata.update(
            {
                "status": "completed",
                "successful_runs": successful_runs,
                "failed_runs": failed_runs,
                "success_rate": successful_runs / self.total_runs,
                "experiment_duration": experiment_duration,
                "completed_at": datetime.now().isoformat(),
            }
        )

        with open(experiment_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return final_results


@reports_app.command()
def run(
    experiment_id: Annotated[str, Argument(help="Unique experiment identifier")],
    tasks: Annotated[
        str, Option("--tasks", "-t", help="Comma-separated list of task IDs")
    ],
    models: Annotated[
        str, Option("--models", "-m", help="Comma-separated list of models")
    ] = "claude-sonnet-4-20250514",
    n_trials: Annotated[
        int,
        Option(
            "--n-trials",
            "-n",
            help="Number of trials to run per task-model combination",
        ),
    ] = 3,
    timeout: Annotated[
        int, Option("--timeout", help="Timeout per trial in seconds")
    ] = 15 * 60,
    max_workers: Annotated[
        int, Option("--max-workers", "-w", help="Maximum parallel workers")
    ] = 4,
    max_steps: Annotated[
        int | None, Option("--max-steps", help="Maximum steps per trial")
    ] = None,
    todo_tool_enabled: Annotated[
        bool,
        Option("--todo-tool-enabled", help="Enable the todo tool for task management"),
    ] = False,
    runs_dir: runs_dir_option() = DEFAULT_RUNS_DIR,
    tasks_dir: tasks_dir_option() = DEFAULT_TASKS_DIR,
):
    """Run parallel experiments across multiple tasks and models."""

    # Parse input lists
    task_list = [t.strip() for t in tasks.split(",")]
    model_list = [m.strip() for m in models.split(",")]

    rich_print(f"[blue]Starting experiment: {experiment_id}[/blue]")

    # Create execution plan table
    from rich.console import Console

    console = Console()
    plan_table = Table(
        title="Execution Plan", show_header=True, header_style="bold magenta"
    )
    plan_table.add_column("Task", style="cyan", no_wrap=True, min_width=15)
    plan_table.add_column("Models", style="green", no_wrap=True, min_width=20)
    plan_table.add_column("Trials Per Model", justify="center", style="yellow", width=8)
    plan_table.add_column("Total Runs", justify="center", style="red", width=10)

    for task in task_list:
        # Truncate models if too long
        models_str = ", ".join(model_list)
        if len(models_str) > 35:
            models_str = models_str[:32] + "..."

        plan_table.add_row(
            task, models_str, str(n_trials), str(len(model_list) * n_trials)
        )

    # Add separator row if multiple tasks
    if len(task_list) > 1:
        plan_table.add_section()

    plan_table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{len(model_list)} models[/bold]",
        f"[bold]{n_trials}[/bold]",
        f"[bold]{len(task_list) * len(model_list) * n_trials}[/bold]",
    )

    console.print(plan_table)
    rich_print(f"Max parallel workers: {max_workers}")
    rich_print("")

    # Validate tasks exist and are runnable
    import yaml

    missing_tasks = []
    not_runnable_tasks = []

    for task_id in task_list:
        task_path = tasks_dir / task_id
        task_yaml_path = task_path / "task.yaml"

        if not task_path.exists() or not task_yaml_path.exists():
            missing_tasks.append(task_id)
            continue

        # Check if task is marked as not runnable
        try:
            with open(task_yaml_path) as f:
                task_data = yaml.safe_load(f)
                if task_data.get("not_runnable", False):
                    reason = task_data.get(
                        "not_runnable_reason",
                        "This task cannot be run with the evaluation harness.",
                    )
                    not_runnable_tasks.append((task_id, reason))
        except Exception:
            pass  # If YAML parsing fails, assume runnable

    if missing_tasks:
        rich_print(f"[red]Error: Tasks not found: {', '.join(missing_tasks)}[/red]")
        return

    if not_runnable_tasks:
        rich_print("[red]❌ Error: The following tasks cannot be run:[/red]")
        for task_id, reason in not_runnable_tasks:
            rich_print(f"  [red]• {task_id}:[/red] {reason}")
        rich_print("")
        rich_print("[yellow]These are data generation tasks for human talent.[/yellow]")
        rich_print(
            "[yellow]Use the corresponding -observability variant (not -genobservability).[/yellow]"
        )
        return

    # Validate models
    available_models = [model.value for model in ModelType]
    print(available_models)
    invalid_models = [m for m in model_list if m not in available_models]
    print(invalid_models)
    if invalid_models:
        rich_print(f"[red]Error: Invalid models: {', '.join(invalid_models)}[/red]")
        rich_print(f"Available models: {', '.join(available_models)}")
        return

    # Create experiment config
    config = ExperimentConfig(
        experiment_id=experiment_id,
        tasks=task_list,
        models=model_list,
        n_trials=n_trials,
        timeout=timeout,
        max_workers=max_workers,
        max_steps=max_steps,
        todo_tool_enabled=todo_tool_enabled,
        runs_dir=runs_dir,
        tasks_dir=tasks_dir,
    )

    # Run experiment
    try:
        runner = ExperimentRunner(config)
        results = runner.run_experiment()

        # Display summary
        rich_print(f"\n[green]Experiment completed: {experiment_id}[/green]")
        rich_print(f"Total runs: {results['total_runs']}")
        rich_print(f"Successful: {results['successful_runs']}")
        rich_print(f"Failed: {results['failed_runs']}")
        rich_print(f"Success rate: {results['success_rate']:.2%}")
        rich_print(f"Duration: {results['experiment_duration']:.2f}s")

        experiment_dir = runs_dir / f"experiment_{experiment_id}"
        rich_print(f"Results saved to: {experiment_dir}")

    except Exception as e:
        rich_print(f"[red]Experiment failed: {e}[/red]")
        raise


@reports_app.command(name="list")
def list_experiments(
    runs_dir: runs_dir_option() = DEFAULT_RUNS_DIR,
):
    """List all experiments."""
    experiment_dirs = [
        d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("experiment_")
    ]

    if not experiment_dirs:
        rich_print("[yellow]No experiments found[/yellow]")
        return

    table = create_table(
        "Experiments",
        [
            ("Experiment ID", "cyan"),
            ("Tasks", "green"),
            ("Models", "blue"),
            ("Status", "yellow"),
            ("Success Rate", "magenta"),
            ("Created", "white"),
        ],
    )

    for exp_dir in sorted(
        experiment_dirs, key=lambda x: x.stat().st_mtime, reverse=True
    ):
        metadata_file = exp_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)

                experiment_id = metadata.get("experiment_id", exp_dir.name)
                tasks = (
                    ", ".join(metadata.get("tasks", []))[:30] + "..."
                    if len(", ".join(metadata.get("tasks", []))) > 30
                    else ", ".join(metadata.get("tasks", []))
                )
                models = (
                    ", ".join(metadata.get("models", []))[:20] + "..."
                    if len(", ".join(metadata.get("models", []))) > 20
                    else ", ".join(metadata.get("models", []))
                )
                status = metadata.get("status", "unknown")
                success_rate = (
                    f"{metadata.get('success_rate', 0):.1%}"
                    if metadata.get("success_rate") is not None
                    else "N/A"
                )
                created = (
                    metadata.get("created_at", "unknown")[:16]
                    if metadata.get("created_at")
                    else "unknown"
                )

                table.add_row(
                    experiment_id, tasks, models, status, success_rate, created
                )

            except (json.JSONDecodeError, KeyError):
                table.add_row(
                    exp_dir.name, "unknown", "unknown", "error", "N/A", "unknown"
                )

    rich_print(table)


@reports_app.command()
def show(
    experiment_id: Annotated[str, Argument(help="Experiment ID to show")],
    runs_dir: runs_dir_option() = DEFAULT_RUNS_DIR,
):
    """Show detailed results for an experiment."""
    # Handle both experiment_xxx and xxx formats
    clean_id = (
        experiment_id.replace("experiment_", "")
        if experiment_id.startswith("experiment_")
        else experiment_id
    )
    experiment_dir = runs_dir / f"experiment_{clean_id}"

    if not experiment_dir.exists():
        rich_print(f"[red]Experiment '{experiment_id}' not found[/red]")
        return

    # Load metadata
    metadata_file = experiment_dir / "metadata.json"
    results_file = experiment_dir / "results.json"

    if not metadata_file.exists():
        rich_print("[red]Experiment metadata not found[/red]")
        return

    try:
        with open(metadata_file) as f:
            metadata = json.load(f)

        rich_print(f"[bold]Experiment: {experiment_id}[/bold]")
        rich_print(f"Status: {metadata.get('status', 'unknown')}")
        rich_print(f"Tasks: {', '.join(metadata.get('tasks', []))}")
        rich_print(f"Models: {', '.join(metadata.get('models', []))}")
        rich_print(f"Created: {metadata.get('created_at', 'unknown')}")

        if metadata.get("status") == "completed":
            rich_print(f"Completed: {metadata.get('completed_at', 'unknown')}")
            rich_print(f"Duration: {metadata.get('experiment_duration', 0):.2f}s")
            rich_print(f"Success Rate: {metadata.get('success_rate', 0):.2%}")

        # Show detailed results if available
        if results_file.exists():
            with open(results_file) as f:
                results = json.load(f)

            rich_print("\n[bold]Detailed Results:[/bold]")

            # Create results table
            table = create_table(
                "Individual Trial Results",
                [
                    ("Task", "cyan"),
                    ("Model", "blue"),
                    ("Trial", "magenta"),
                    ("Success", "green"),
                    ("Time", "yellow"),
                    ("Status", "white"),
                ],
            )

            for key, result in results.get("results", {}).items():
                if result.get("status") != "failed":
                    task_id = result.get("task_id", "unknown")
                    model = result.get("model", "unknown")
                    trial_num = result.get("trial_number", "?")
                    success_rate = f"{result.get('success_rate', 0):.1%}"
                    avg_time = f"{result.get('average_time', 0):.2f}s"
                    status = "COMPLETED"

                    table.add_row(
                        task_id, model, str(trial_num), success_rate, avg_time, status
                    )
                else:
                    task_id = result.get("task_id", "unknown")
                    model = result.get("model", "unknown")
                    trial_num = result.get("trial_number", "?")
                    table.add_row(
                        task_id,
                        model,
                        str(trial_num),
                        "N/A",
                        "N/A",
                        "[red]FAILED[/red]",
                    )

            rich_print(table)

    except (json.JSONDecodeError, KeyError) as e:
        rich_print(f"[red]Error reading experiment data: {e}[/red]")


@reports_app.command()
def compare(
    experiment_ids: Annotated[
        str, Argument(help="Comma-separated list of experiment IDs to compare")
    ],
    runs_dir: runs_dir_option() = DEFAULT_RUNS_DIR,
):
    """Compare results across multiple experiments."""
    exp_list = [exp_id.strip() for exp_id in experiment_ids.split(",")]

    rich_print(f"[blue]Comparing experiments: {', '.join(exp_list)}[/blue]")

    # Load all experiment data
    experiments = {}
    for exp_id in exp_list:
        exp_dir = runs_dir / f"experiment_{exp_id}"
        if not exp_dir.exists():
            rich_print(f"[yellow]Warning: Experiment '{exp_id}' not found[/yellow]")
            continue

        metadata_file = exp_dir / "metadata.json"
        results_file = exp_dir / "results.json"

        if metadata_file.exists() and results_file.exists():
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
                with open(results_file) as f:
                    results = json.load(f)

                experiments[exp_id] = {"metadata": metadata, "results": results}
            except (json.JSONDecodeError, KeyError):
                rich_print(
                    f"[yellow]Warning: Could not load data for experiment '{exp_id}'[/yellow]"
                )

    if not experiments:
        rich_print("[red]No valid experiments found for comparison[/red]")
        return

    # Create comparison table
    table = create_table(
        "Experiment Comparison",
        [
            ("Experiment", "cyan"),
            ("Total Runs", "blue"),
            ("Success Rate", "green"),
            ("Avg Duration", "yellow"),
            ("Status", "white"),
        ],
    )

    for exp_id, data in experiments.items():
        metadata = data["metadata"]
        results = data["results"]

        total_runs = results.get("total_runs", 0)
        success_rate = f"{results.get('success_rate', 0):.1%}"
        duration = f"{results.get('experiment_duration', 0):.2f}s"
        status = metadata.get("status", "unknown")

        table.add_row(exp_id, str(total_runs), success_rate, duration, status)

    rich_print(table)


@reports_app.command()
def clean(
    experiment_id: Annotated[
        str | None, Argument(help="Specific experiment ID to clean (optional)")
    ] = None,
    runs_dir: runs_dir_option() = DEFAULT_RUNS_DIR,
    force: Annotated[
        bool, Option("--force", "-f", help="Force cleanup without confirmation")
    ] = False,
):
    """Clean up experiment directories."""
    if experiment_id:
        # Clean specific experiment - handle both experiment_xxx and xxx formats
        clean_id = (
            experiment_id.replace("experiment_", "")
            if experiment_id.startswith("experiment_")
            else experiment_id
        )
        exp_dir = runs_dir / f"experiment_{clean_id}"
        if not exp_dir.exists():
            rich_print(f"[red]Experiment '{experiment_id}' not found[/red]")
            return

        if not force:
            confirm = (
                input(f"Delete experiment '{experiment_id}'? (y/N): ").lower().strip()
            )
            if confirm != "y":
                rich_print("[yellow]Cleanup cancelled[/yellow]")
                return

        import shutil

        shutil.rmtree(exp_dir)
        rich_print(f"[green]Deleted experiment: {experiment_id}[/green]")

    else:
        # Clean all experiments
        experiment_dirs = [
            d
            for d in runs_dir.iterdir()
            if d.is_dir() and d.name.startswith("experiment_")
        ]

        if not experiment_dirs:
            rich_print("[yellow]No experiments to clean[/yellow]")
            return

        rich_print(
            f"[yellow]Found {len(experiment_dirs)} experiments to clean[/yellow]"
        )

        if not force:
            confirm = input("Delete all experiments? (y/N): ").lower().strip()
            if confirm != "y":
                rich_print("[yellow]Cleanup cancelled[/yellow]")
                return

        import shutil

        for exp_dir in experiment_dirs:
            shutil.rmtree(exp_dir)

        rich_print(f"[green]Cleaned {len(experiment_dirs)} experiments[/green]")
