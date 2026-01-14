import subprocess
from pathlib import Path
from typing import Annotated

from typer import Argument, Option, Typer

from ..utils import (
    DEFAULT_DIRS,
    create_dir_option,
    create_table,
    get_directories_with_file,
    rich_print,
)

tasks_app = Typer(no_args_is_help=True)

# Constants
DEFAULT_TASKS_DIR = DEFAULT_DIRS["tasks"]


# Common option for all commands
def tasks_dir_option() -> Annotated[Path, Option]:
    return create_dir_option("tasks", "tasks-dir", "Path to the tasks directory")


def _list_tasks(tasks_dir: Path) -> list[Path]:
    """List task directories that have task.yaml files."""
    return get_directories_with_file(tasks_dir, "task.yaml")


def _task_exists(task_id: str, tasks_dir: Path) -> Path | None:
    """Check if task exists and return its path, or None if not found."""
    task_dir = tasks_dir / task_id
    return task_dir if task_dir.exists() else None


@tasks_app.command()
def list(
    tasks_dir: tasks_dir_option() = DEFAULT_TASKS_DIR,
    show_details: Annotated[
        bool,
        Option("--details", help="Show detailed information about each task"),
    ] = False,
):
    """List all available tasks."""
    task_dirs = _list_tasks(tasks_dir)

    if not task_dirs:
        rich_print(f"[yellow]No tasks found in {tasks_dir}[/yellow]")
        return

    if show_details:
        table = create_table(
            "Available Tasks",
            [
                ("Task ID", "cyan"),
                ("Status", "green"),
                ("Files", "blue"),
            ],
        )

        for task_dir in sorted(task_dirs):
            # Check what files exist
            files = []
            if (task_dir / "Dockerfile").exists():
                files.append("Dockerfile")
            if (task_dir / "docker-compose.yaml").exists():
                files.append("docker-compose.yaml")
            if (task_dir / "tests").exists():
                files.append("tests/")
            if (task_dir / "solution.py").exists():
                files.append("solution.py")

            status = "Complete" if len(files) >= 3 else "In Progress"
            files_str = ", ".join(files) if files else "None"
            table.add_row(task_dir.name, status, files_str)

        rich_print(table)
    else:
        rich_print(f"[bold]Available tasks in {tasks_dir}:[/bold]")
        for task_dir in sorted(task_dirs):
            rich_print(f"  • {task_dir.name}")


@tasks_app.command()
def show(
    task_id: Annotated[str, Argument(help="The ID of the task to show")],
    tasks_dir: tasks_dir_option() = DEFAULT_TASKS_DIR,
):
    """Show detailed information about a specific task."""
    task_dir = _task_exists(task_id, tasks_dir)
    if not task_dir:
        rich_print(f"[red]Task '{task_id}' not found in {tasks_dir}[/red]")
        return

    rich_print(f"[bold]Task: {task_id}[/bold]")
    rich_print(f"Path: {task_dir.absolute()}")

    # Show task.yaml content
    task_yaml_path = task_dir / "task.yaml"
    if task_yaml_path.exists():
        rich_print("\n[bold]Task Configuration:[/bold]")
        try:
            content = task_yaml_path.read_text()
            rich_print(f"```yaml\n{content}\n```")
        except Exception as e:
            rich_print(f"[red]Error reading task.yaml: {e}[/red]")

    # Show file structure
    rich_print("\n[bold]File Structure:[/bold]")
    for item in sorted(task_dir.iterdir()):
        if item.is_dir():
            rich_print(f"  📁 {item.name}/")
        else:
            rich_print(f"  📄 {item.name}")


@tasks_app.command()
def info(
    tasks_dir: tasks_dir_option() = DEFAULT_TASKS_DIR,
):
    """Show information about available tasks."""
    import yaml

    task_dirs = _list_tasks(tasks_dir)

    if not task_dirs:
        rich_print(f"[yellow]No tasks found in {tasks_dir}[/yellow]")
        return

    # Gather statistics
    total_tasks = len(task_dirs)
    categories = {}
    difficulties = {}
    tags_count = {}
    complete_tasks = 0

    for task_dir in task_dirs:
        # Check if task is complete (has all necessary files)
        has_dockerfile = (task_dir / "Dockerfile").exists()
        has_compose = (task_dir / "docker-compose.yaml").exists()
        has_tests = (task_dir / "tests").exists()

        if has_dockerfile and has_compose:
            complete_tasks += 1

        # Read task.yaml for metadata
        task_yaml_path = task_dir / "task.yaml"
        if task_yaml_path.exists():
            try:
                with open(task_yaml_path) as f:
                    task_data = yaml.safe_load(f)

                # Extract metadata
                metadata = task_data.get("metadata", {})

                # Count categories
                category = metadata.get("category", "uncategorized")
                categories[category] = categories.get(category, 0) + 1

                # Count difficulties
                difficulty = metadata.get("difficulty", "unknown")
                difficulties[difficulty] = difficulties.get(difficulty, 0) + 1

                # Count tags
                tags = metadata.get("tags", [])
                for tag in tags:
                    tags_count[tag] = tags_count.get(tag, 0) + 1

            except Exception:
                pass

    # Display information
    rich_print("[bold]Tasks Information[/bold]")
    rich_print(f"Directory: {tasks_dir.absolute()}")
    rich_print("\n[bold]Summary:[/bold]")
    rich_print(f"  • Total tasks: {total_tasks}")
    rich_print(f"  • Complete tasks: {complete_tasks}")
    rich_print(f"  • Incomplete tasks: {total_tasks - complete_tasks}")

    # Show categories breakdown
    if categories:
        rich_print("\n[bold]Categories:[/bold]")
        for category, count in sorted(
            categories.items(), key=lambda x: x[1], reverse=True
        ):
            rich_print(f"  • {category}: {count}")

    # Show difficulties breakdown
    if difficulties:
        rich_print("\n[bold]Difficulties:[/bold]")
        difficulty_order = ["easy", "medium", "hard", "unknown"]
        for difficulty in difficulty_order:
            if difficulty in difficulties:
                rich_print(f"  • {difficulty}: {difficulties[difficulty]}")

    # Show top tags
    if tags_count:
        rich_print("\n[bold]Top Tags:[/bold]")
        sorted_tags = sorted(tags_count.items(), key=lambda x: x[1], reverse=True)[:10]
        for tag, count in sorted_tags:
            rich_print(f"  • {tag}: {count}")

    rich_print(
        "\n[yellow]Use 'apx tasks list --details' to see individual task details[/yellow]"
    )


@tasks_app.command()
def clean(
    force: Annotated[
        bool,
        Option("--force", "-f", help="Force cleanup without confirmation"),
    ] = False,
):
    """Clean up Docker images and containers."""
    _clean_docker_cache(force)


def _clean_docker_cache(force: bool = False) -> None:
    """Clean up Docker images and containers."""
    try:
        # List Docker images with apex- prefix
        result = subprocess.run(
            [
                "docker",
                "images",
                "--filter",
                "reference=apex-*",
                "--format",
                "table {{.Repository}}:{{.Tag}}\t{{.Size}}",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        if result.stdout.strip():
            rich_print("[yellow]Docker images to remove:[/yellow]")
            rich_print(result.stdout)
        else:
            rich_print("[green]No apex-* Docker images found[/green]")
            return

        if not force:
            confirm = input("\nRemove these Docker images? (y/N): ").lower().strip()
            if confirm != "y":
                rich_print("[yellow]Cleanup cancelled.[/yellow]")
                return

        # Remove Docker images
        subprocess.run(
            ["docker", "rmi", "-f", "$(docker images -q apex-*)"],
            shell=True,
            capture_output=True,
            check=False,
        )

        # Remove stopped containers
        subprocess.run(
            ["docker", "container", "prune", "-f"], capture_output=True, check=False
        )

        # Remove unused networks
        subprocess.run(
            ["docker", "network", "prune", "-f"], capture_output=True, check=False
        )

        rich_print("[green]Docker cache cleanup completed[/green]")

    except subprocess.CalledProcessError as e:
        rich_print(f"[red]Error during cleanup: {e}[/red]")
    except FileNotFoundError:
        rich_print(
            "[red]Docker not found. Make sure Docker is installed and running.[/red]"
        )


@tasks_app.command()
def export(
    task_id: Annotated[str, Argument(help="The ID of the task to export")],
    output_path: Annotated[
        Path,
        Option("--output", "-o", help="Output archive path"),
    ] = None,
    tasks_dir: tasks_dir_option() = DEFAULT_TASKS_DIR,
):
    """Export a task to a tar archive."""
    task_dir = _task_exists(task_id, tasks_dir)
    if not task_dir:
        rich_print(f"[red]Task '{task_id}' not found in {tasks_dir}[/red]")
        return

    if output_path is None:
        output_path = Path(f"{task_id}.tar.gz")

    try:
        rich_print(f"[blue]Exporting task '{task_id}' to {output_path}...[/blue]")

        # Create tar archive
        subprocess.run(
            [
                "tar",
                "-czf",
                str(output_path),
                "-C",
                str(task_dir.parent),
                task_dir.name,
            ],
            check=True,
            capture_output=True,
        )

        rich_print(f"[green]Successfully exported task to {output_path}[/green]")

    except subprocess.CalledProcessError as e:
        rich_print(f"[red]Error exporting task: {e}[/red]")
    except FileNotFoundError:
        rich_print("[red]tar command not found. Please install tar.[/red]")
