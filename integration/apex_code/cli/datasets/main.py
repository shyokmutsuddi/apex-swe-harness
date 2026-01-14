import builtins
import shutil
from pathlib import Path
from typing import Annotated

from typer import Argument, Option, Typer

from ..utils import create_table, rich_print
from .constants import CACHE_DIR
from .models import Dataset, DatasetConfig
from .registry_client import RegistryClient

datasets_app = Typer(no_args_is_help=True)


@datasets_app.command()
def list(
    name: Annotated[
        str | None,
        Option("--name", "-n", help="Filter datasets by name"),
    ] = None,
    registry_url: Annotated[
        str | None,
        Option("--registry-url", help="Custom registry URL"),
    ] = None,
    local_registry_path: Annotated[
        str | None,
        Option("--local-registry-path", help="Path to local registry file"),
    ] = None,
):
    """List available datasets."""
    try:
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
            from .constants import DEFAULT_REGISTRY_URL

            registry_source = DEFAULT_REGISTRY_URL

        client = RegistryClient(registry_source)
        datasets = client.get_datasets()

        # Filter by name if specified
        if name:
            datasets = [d for d in datasets if name.lower() in d.name.lower()]

        if not datasets:
            rich_print(
                f"[yellow]No datasets found{' for name: ' + name if name else ''}[/yellow]"
            )
            return

        # Create table
        table = create_table(
            "Available Datasets",
            [
                ("Name", "cyan"),
                ("Version", "green"),
                ("Description", "white"),
                ("Branch", "blue"),
            ],
        )

        # Sort datasets by name and version
        sorted_datasets = sorted(datasets, key=lambda x: (x.name, x.version))

        for dataset in sorted_datasets:
            description = dataset.description or "No description"
            # Truncate long descriptions
            if len(description) > 60:
                description = description[:57] + "..."

            table.add_row(dataset.name, dataset.version, description, dataset.branch)

        rich_print(table)
        rich_print(
            "\n[yellow]Use 'apx datasets download <name>' to download a dataset[/yellow]"
        )
    except Exception as e:
        rich_print(f"[red]Failed to list datasets: {e}[/red]")


@datasets_app.command()
def download(
    name: Annotated[str, Argument(help="Dataset name to download")],
    version: Annotated[
        str,
        Option("--version", "-v", help="Dataset version (default: head)"),
    ] = "head",
    output_dir: Annotated[
        Path | None,
        Option("--output-dir", "-o", help="Output directory"),
    ] = None,
    overwrite: Annotated[
        bool,
        Option("--overwrite", help="Overwrite existing dataset"),
    ] = False,
    registry_url: Annotated[
        str | None,
        Option("--registry-url", help="Custom registry URL"),
    ] = None,
    local_registry_path: Annotated[
        str | None,
        Option("--local-registry-path", help="Path to local registry file"),
    ] = None,
):
    """Download a dataset."""
    try:
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
            from .constants import DEFAULT_REGISTRY_URL

            registry_source = DEFAULT_REGISTRY_URL

        client = RegistryClient(registry_source)

        # Check if dataset exists
        dataset = client.get_dataset(name, version)
        if not dataset:
            rich_print(f"[red]Dataset '{name}' version '{version}' not found[/red]")
            rich_print(
                "[yellow]Use 'apx datasets list' to see available datasets[/yellow]"
            )
            return

        # Download dataset
        result_path = client.download_dataset(
            name=name, version=version, output_dir=output_dir, overwrite=overwrite
        )

        rich_print("[green]✅ Dataset downloaded successfully![/green]")
        rich_print(f"[blue]📁 Location: {result_path}[/blue]")
        rich_print(
            "[yellow]💡 You can now use tasks from this dataset with 'apx tasks' commands[/yellow]"
        )
    except Exception as e:
        rich_print(f"[red]Failed to download dataset: {e}[/red]")


@datasets_app.command()
def info(
    name: Annotated[str, Argument(help="Dataset name")],
    version: Annotated[
        str | None,
        Option("--version", "-v", help="Dataset version (default: latest compatible)"),
    ] = None,
    registry_url: Annotated[
        str | None,
        Option("--registry-url", help="Custom registry URL"),
    ] = None,
    local_registry_path: Annotated[
        str | None,
        Option("--local-registry-path", help="Path to local registry file"),
    ] = None,
):
    """Show detailed information about a dataset."""
    try:
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
            from .constants import DEFAULT_REGISTRY_URL

            registry_source = DEFAULT_REGISTRY_URL

        client = RegistryClient(registry_source)
        # Resolve requested dataset version (latest compatible if not provided)
        if version is None:
            dataset = client.get_latest_dataset(name, "0.1.0")
        else:
            dataset = client.get_dataset(name, version)

        if not dataset:
            resolved_version = version or "latest"
            rich_print(
                f"[red]Dataset '{name}' version '{resolved_version}' not found[/red]"
            )
            return
        rich_print("[bold]Dataset Information[/bold]")
        rich_print(f"Name: {dataset.name}")
        rich_print(f"Version: {dataset.version}")
        rich_print(f"Description: {dataset.description or 'No description'}")
        rich_print(f"GitHub URL: {dataset.github_url}")
        rich_print(f"Dataset Path: {dataset.dataset_path}")
        rich_print(f"Branch: {dataset.branch}")
        rich_print(f"Commit Hash: {dataset.commit_hash}")
        rich_print(f"Apex-Code Version: {dataset.apex_code_version}")
        # Enhanced metadata when available
        if getattr(dataset, "author", None):
            rich_print(f"Author: {dataset.author}")
        if getattr(dataset, "license", None):
            rich_print(f"License: {dataset.license}")
        if getattr(dataset, "homepage", None):
            rich_print(f"Homepage: {dataset.homepage}")
        if getattr(dataset, "documentation", None):
            rich_print(f"Documentation: {dataset.documentation}")

    except Exception as e:
        rich_print(f"[red]Failed to get dataset info: {e}[/red]")


# Create cache subcommand group
cache_app = Typer(no_args_is_help=True)


@cache_app.command()
def list():
    """List cached datasets."""
    if not CACHE_DIR.exists():
        rich_print("[yellow]No cache directory found[/yellow]")
        return

    rich_print(f"[bold]Dataset Cache: {CACHE_DIR}[/bold]")

    if not any(CACHE_DIR.iterdir()):
        rich_print("[yellow]Cache is empty[/yellow]")
        return

    table = create_table(
        "Cached Datasets",
        [
            ("Dataset", "cyan"),
            ("Version", "green"),
            ("Size", "blue"),
            ("Path", "white"),
        ],
    )

    for dataset_dir in CACHE_DIR.iterdir():
        if dataset_dir.is_dir():
            for version_dir in dataset_dir.iterdir():
                if version_dir.is_dir():
                    # Calculate directory size
                    total_size = sum(
                        f.stat().st_size for f in version_dir.rglob("*") if f.is_file()
                    )
                    size_str = f"{total_size / (1024 * 1024):.1f} MB"

                    table.add_row(
                        dataset_dir.name, version_dir.name, size_str, str(version_dir)
                    )

    rich_print(table)


@cache_app.command()
def clean():
    """Clean dataset cache."""
    if not CACHE_DIR.exists():
        rich_print("[yellow]No cache directory found[/yellow]")
        return

    rich_print(
        f"[yellow]This will remove all cached datasets from {CACHE_DIR}[/yellow]"
    )
    confirm = input("Are you sure? (y/N): ").lower().strip()

    if confirm == "y":
        shutil.rmtree(CACHE_DIR)
        rich_print("[green]Cache cleaned successfully[/green]")
    else:
        rich_print("[yellow]Cache cleanup cancelled[/yellow]")


@cache_app.command()
def info():
    """Show cache information."""
    try:
        from .registry_client import RegistryClient

        # Check for any registry cache
        client = RegistryClient()

        # Try to load any cache first
        any_cache = client._load_any_cache()
        if any_cache is not None:
            # Get cache info from the loaded cache
            cache_info = client.get_cache_info()
            rich_print("[green]Registry cache found![/green]")
            rich_print(f"  Cached at: {cache_info.get('cached_at')}")
            rich_print(f"  Expires at: {cache_info.get('expires_at')}")
            rich_print(f"  Valid: {cache_info.get('is_valid')}")
            rich_print(f"  Dataset count: {cache_info.get('dataset_count')}")
            rich_print(f"  Registry URL: {cache_info.get('registry_url')}")
        else:
            # Fallback to checking default URL cache
            cache_info = client.get_cache_info()
            if cache_info.get("cached"):
                rich_print("[green]Registry cache found![/green]")
                rich_print(f"  Cached at: {cache_info.get('cached_at')}")
                rich_print(f"  Expires at: {cache_info.get('expires_at')}")
                rich_print(f"  Valid: {cache_info.get('is_valid')}")
                rich_print(f"  Dataset count: {cache_info.get('dataset_count')}")
                rich_print(f"  Registry URL: {cache_info.get('registry_url')}")
            else:
                rich_print("[yellow]No registry cache found[/yellow]")
        # Show cache directory info
        if CACHE_DIR.exists():
            cache_files = builtins.list(CACHE_DIR.glob("*"))
            rich_print(f"\n[blue]Cache directory: {CACHE_DIR}[/blue]")
            rich_print(f"  Files: {len(cache_files)}")

            total_size = sum(f.stat().st_size for f in cache_files if f.is_file())
            rich_print(f"  Total size: {total_size / 1024:.1f} KB")
        else:
            rich_print(
                f"\n[yellow]Cache directory does not exist: {CACHE_DIR}[/yellow]"
            )

    except Exception as e:
        rich_print(f"[red]Failed to get cache info: {e}[/red]")


@cache_app.command()
def validate():
    """Validate cache integrity."""
    try:
        from .registry_client import RegistryClient

        client = RegistryClient()

        # Try to load any cache first
        any_cache = client._load_any_cache()
        if any_cache is not None:
            # Get cache info from the loaded cache
            cache_info = client.get_cache_info()
            rich_print("[green]Cache validation passed![/green]")
            rich_print(f"  Datasets: {len(any_cache.datasets)}")
            rich_print(f"  Cache valid: {cache_info.get('is_valid')}")
        else:
            # Fallback to checking default URL cache
            cache_info = client.get_cache_info()
            if not cache_info.get("cached"):
                rich_print("[yellow]No cache to validate[/yellow]")
                return

            # Try to load registry from cache
            try:
                registry = client.get_registry()
                rich_print("[green]Cache validation passed![/green]")
                rich_print(f"  Datasets: {len(registry.datasets)}")
                rich_print(f"  Cache valid: {cache_info.get('is_valid')}")
            except Exception as e:
                rich_print(f"[red]Cache validation failed: {e}[/red]")

    except Exception as e:
        rich_print(f"[red]Failed to validate cache: {e}[/red]")


# Create validation subcommand group
validation_app = Typer(no_args_is_help=True)


@validation_app.command()
def registry(
    registry_url: Annotated[
        str | None,
        Option("--registry-url", help="Custom registry URL"),
    ] = None,
    local_registry_path: Annotated[
        str | None,
        Option("--local-registry-path", help="Path to local registry file"),
    ] = None,
    apex_code_version: Annotated[
        str,
        Option(
            "--apex-code-version", help="Apex-code version for compatibility checks"
        ),
    ] = "0.1.0",
    check_accessibility: Annotated[
        bool,
        Option("--check-accessibility", help="Check remote URL accessibility"),
    ] = False,
):
    """Validate registry integrity and compatibility."""
    try:
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
            from .constants import DEFAULT_REGISTRY_URL

            registry_source = DEFAULT_REGISTRY_URL

        client = RegistryClient(registry_source)

        # Validate registry
        rich_print("[bold]Validating registry...[/bold]")
        validation_result = client.validate_registry(apex_code_version)

        # Display results
        if validation_result["valid"]:
            rich_print("[green]Registry validation passed![/green]")
        else:
            rich_print("[red]Registry validation failed![/red]")
        # Show statistics
        stats = validation_result["stats"]
        rich_print("\n[bold]Registry Statistics:[/bold]")
        rich_print(f"Total datasets: {stats['total_datasets']}")
        rich_print(
            f"Compatible with apex-code {apex_code_version}: {stats['compatible_datasets']}"
        )
        rich_print(f"Incompatible: {stats['incompatible_datasets']}")
        rich_print(f"Duplicate datasets: {stats['duplicate_datasets']}")
        rich_print(f"Invalid versions: {stats['invalid_versions']}")

        # Show issues
        if validation_result["issues"]:
            rich_print("\n[red]Issues:[/red]")
            for issue in validation_result["issues"]:
                rich_print(f"  • {issue}")

        # Show warnings
        if validation_result["warnings"]:
            rich_print("\n[yellow]Warnings:[/yellow]")
            for warning in validation_result["warnings"]:
                rich_print(f"  • {warning}")

        # Check accessibility if requested
        if check_accessibility:
            rich_print("\n[bold]Checking remote accessibility...[/bold]")
            accessibility_result = client.validate_remote_accessibility()

            if accessibility_result["accessible"]:
                rich_print("[green]All remote URLs are accessible![/green]")
            else:
                rich_print("[red]Some remote URLs are not accessible![/red]")
            # Show accessibility statistics
            acc_stats = accessibility_result["stats"]
            rich_print("\n[bold]Accessibility Statistics:[/bold]")
            rich_print(f"Total URLs: {acc_stats['total_urls']}")
            rich_print(f"Accessible: {acc_stats['accessible_urls']}")
            rich_print(f"Inaccessible: {acc_stats['inaccessible_urls']}")
            rich_print(f"Timeouts: {acc_stats['timeout_urls']}")

            # Show accessibility issues
            if accessibility_result["issues"]:
                rich_print("\n[red]Accessibility Issues:[/red]")
                for issue in accessibility_result["issues"]:
                    rich_print(f"  • {issue}")

            # Show accessibility warnings
            if accessibility_result["warnings"]:
                rich_print("\n[yellow]Accessibility Warnings:[/yellow]")
                for warning in accessibility_result["warnings"]:
                    rich_print(f"  • {warning}")

    except Exception as e:
        rich_print(f"[red]Failed to validate registry: {e}[/red]")


@validation_app.command()
def dataset(
    name: Annotated[str, Argument(help="Dataset name to validate")],
    version: Annotated[
        str,
        Option("--version", "-v", help="Dataset version (default: head)"),
    ] = "head",
    registry_url: Annotated[
        str | None,
        Option("--registry-url", help="Custom registry URL"),
    ] = None,
    local_registry_path: Annotated[
        str | None,
        Option("--local-registry-path", help="Path to local registry file"),
    ] = None,
    apex_code_version: Annotated[
        str,
        Option(
            "--apex-code-version", help="Apex-code version for compatibility checks"
        ),
    ] = "0.1.0",
):
    """Validate a specific dataset."""
    try:
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
        elif registry_url:
            # Use custom registry URL
            registry_source = registry_url
        else:
            # Use default registry
            from .constants import DEFAULT_REGISTRY_URL

            registry_source = DEFAULT_REGISTRY_URL

        client = RegistryClient(registry_source)

        # Get dataset
        dataset = client.get_dataset(name, version)
        if not dataset:
            rich_print(f"[red]Dataset '{name}' version '{version}' not found[/red]")
            return

        rich_print(f"[bold]Validating dataset: {name}=={version}[/bold]")

        # Validate dataset
        issues = []
        warnings = []

        # Check version format
        if dataset.version != "head":
            try:
                from packaging import version

                version.parse(dataset.version)
            except version.InvalidVersion:
                issues.append(f"Invalid version format: {dataset.version}")

        # Check compatibility
        if not dataset.is_compatible_with(apex_code_version):
            warnings.append(f"Incompatible with apex-code {apex_code_version}")

        # Check required fields
        if not dataset.github_url:
            issues.append("Missing github_url")

        if not dataset.dataset_path:
            issues.append("Missing dataset_path")

        # Check URL format
        if dataset.github_url and not client._is_valid_url(dataset.github_url):
            issues.append(f"Invalid URL format: {dataset.github_url}")

        # Display results
        if not issues:
            rich_print("[green]✅ Dataset validation passed![/green]")
        else:
            rich_print("[red]❌ Dataset validation failed![/red]")
        # Show dataset info
        rich_print("\n[bold]Dataset Information:[/bold]")
        rich_print(f"Name: {dataset.name}")
        rich_print(f"Version: {dataset.version}")
        rich_print(f"Description: {dataset.description or 'No description'}")
        rich_print(f"GitHub URL: {dataset.github_url}")
        rich_print(f"Dataset Path: {dataset.dataset_path}")
        rich_print(f"Branch: {dataset.branch}")
        rich_print(f"Commit Hash: {dataset.commit_hash}")
        rich_print(f"Apex-Code Version: {dataset.apex_code_version}")
        rich_print(
            f"Compatible with {apex_code_version}: {'Yes' if dataset.is_compatible_with(apex_code_version) else 'No'}"
        )

        # Show issues
        if issues:
            rich_print("\n[red]Issues:[/red]")
            for issue in issues:
                rich_print(f"  • {issue}")

        # Show warnings
        if warnings:
            rich_print("\n[yellow]Warnings:[/yellow]")
            for warning in warnings:
                rich_print(f"  • {warning}")

    except Exception as e:
        rich_print(f"[red]Failed to validate dataset: {e}[/red]")


@datasets_app.command()
def tasks(
    name: Annotated[str, Argument(help="Dataset name")],
    version: Annotated[
        str,
        Option("--version", "-v", help="Dataset version (default: head)"),
    ] = "head",
    category: Annotated[
        str | None,
        Option("--category", "-c", help="Filter by category"),
    ] = None,
    difficulty: Annotated[
        str | None,
        Option("--difficulty", "-d", help="Filter by difficulty"),
    ] = None,
    tags: Annotated[
        str | None,
        Option("--tags", "-t", help="Comma-separated tags to filter by"),
    ] = None,
    exclude: Annotated[
        str | None,
        Option(
            "--exclude",
            "-e",
            help="Comma-separated task IDs or glob patterns to exclude",
        ),
    ] = None,
    n_tasks: Annotated[
        int | None,
        Option("--n-tasks", "-n", help="Limit number of tasks to show"),
    ] = None,
    registry_url: Annotated[
        str | None,
        Option("--registry-url", help="Custom registry URL"),
    ] = None,
    local_registry_path: Annotated[
        str | None,
        Option("--local-registry-path", help="Path to local registry file"),
    ] = None,
):
    """List tasks in a dataset with optional filtering."""
    try:
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
        elif registry_url:
            # Use custom registry URL
            registry_source = registry_url
        else:
            # Use default registry
            from .constants import DEFAULT_REGISTRY_URL

            registry_source = DEFAULT_REGISTRY_URL

        client = RegistryClient(registry_source)

        # Parse tags if provided
        tag_list = None
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",")]

        # Parse exclusions if provided
        exclude_list = None
        if exclude:
            exclude_list = [pattern.strip() for pattern in exclude.split(",")]

        # Get tasks with filtering
        task_list = client.list_tasks(
            name=name,
            version=version,
            category=category,
            difficulty=difficulty,
            tags=tag_list,
            exclude=exclude_list,
            n_tasks=n_tasks,
        )

        if not task_list:
            rich_print(
                f"[yellow]No tasks found in dataset '{name}' version '{version}'[/yellow]"
            )
            if category or difficulty or tag_list:
                rich_print(
                    "[yellow]Try removing filters: --category, --difficulty, --tags[/yellow]"
                )
            return

        # Create table
        table = create_table(
            f"Tasks in {name}=={version}",
            [
                ("Task ID", "cyan"),
                ("Category", "green"),
                ("Difficulty", "blue"),
                ("Tags", "yellow"),
                ("Description", "white"),
                ("Time (min)", "magenta"),
            ],
        )

        # Sort tasks by task_id
        sorted_tasks = sorted(task_list, key=lambda x: x["task_id"])

        for task in sorted_tasks:
            # Format tags
            tags_str = ", ".join(task["tags"]) if task["tags"] else "None"
            if len(tags_str) > 30:
                tags_str = tags_str[:27] + "..."

            # Format description
            description = task["description"] or "No description"
            if len(description) > 40:
                description = description[:37] + "..."

            # Format estimated time
            time_str = (
                str(task["estimated_time"]) if task["estimated_time"] else "Unknown"
            )

            table.add_row(
                task["task_id"],
                task["category"] or "None",
                task["difficulty"] or "None",
                tags_str,
                description,
                time_str,
            )

        rich_print(table)

        # Show filter summary
        if category or difficulty or tag_list:
            rich_print("\n[yellow]Applied filters:[/yellow]")
            if category:
                rich_print(f"  Category: {category}")
            if difficulty:
                rich_print(f"  Difficulty: {difficulty}")
            if tag_list:
                rich_print(f"  Tags: {', '.join(tag_list)}")

        rich_print(f"\n[blue]Total tasks: {len(task_list)}[/blue]")

    except Exception as e:
        rich_print(f"[red]Failed to list tasks: {e}[/red]")


@datasets_app.command()
def evaluate(
    name: Annotated[str, Argument(help="Dataset name")],
    version: Annotated[
        str | None,
        Option("--version", "-v", help="Dataset version (default: latest compatible)"),
    ] = None,
    task_subset: Annotated[
        str | None,
        Option(
            "--task-subset", help="Comma-separated task IDs or glob patterns to run"
        ),
    ] = None,
    category: Annotated[
        str | None,
        Option("--category", "-c", help="Filter by category"),
    ] = None,
    difficulty: Annotated[
        str | None,
        Option("--difficulty", "-d", help="Filter by difficulty"),
    ] = None,
    tags: Annotated[
        str | None,
        Option("--tags", "-t", help="Comma-separated tags to filter by"),
    ] = None,
    model: Annotated[
        str,
        Option("--model", "-m", help="Model to use for evaluation"),
    ] = "claude-sonnet-4-20250514",
    max_trials: Annotated[
        int,
        Option("--max-trials", "-n", help="Maximum number of trials per task"),
    ] = 3,
    registry_url: Annotated[
        str | None,
        Option("--registry-url", help="Custom registry URL"),
    ] = None,
    local_registry_path: Annotated[
        str | None,
        Option("--local-registry-path", help="Path to local registry file"),
    ] = None,
):
    """Evaluate tasks directly from remote dataset without local download."""
    try:
        # Determine registry source
        if local_registry_path:
            local_path = Path(local_registry_path)
            if not local_path.exists():
                rich_print(
                    f"[red]Error: Local registry file not found: {local_path}[/red]"
                )
                return
            registry_source = f"file://{local_path.absolute()}"
        elif registry_url:
            registry_source = registry_url
        else:
            from .constants import DEFAULT_REGISTRY_URL

            registry_source = DEFAULT_REGISTRY_URL

        client = RegistryClient(registry_source)

        # Parse filters
        task_subset_list = None
        if task_subset:
            task_subset_list = [pattern.strip() for pattern in task_subset.split(",")]

        tag_list = None
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",")]

        # Resolve dataset version
        if version is None:
            ds = client.get_latest_dataset(name, "0.1.0")
            if not ds:
                rich_print(f"[red]Dataset '{name}' not found in registry[/red]")
                return
            resolved_version = ds.version
        else:
            resolved_version = version

        # Show evaluation parameters
        rich_print(f"[blue]Evaluating dataset: {name}=={resolved_version}[/blue]")
        if task_subset_list:
            rich_print(f"[blue]Task subset: {', '.join(task_subset_list)}[/blue]")
        if category:
            rich_print(f"[blue]Category filter: {category}[/blue]")
        if difficulty:
            rich_print(f"[blue]Difficulty filter: {difficulty}[/blue]")
        if tag_list:
            rich_print(f"[blue]Tags filter: {', '.join(tag_list)}[/blue]")
        rich_print(f"[blue]Model: {model}, Max trials: {max_trials}[/blue]")

        # Progress callback
        def progress_callback(current: int, total: int, task_id: str):
            rich_print(f"[yellow]Evaluating task {current}/{total}: {task_id}[/yellow]")

        # Run direct evaluation
        result = client.evaluate_dataset_directly(
            name=name,
            version=resolved_version,
            task_subset=task_subset_list,
            category=category,
            difficulty=difficulty,
            tags=tag_list,
            agent_type=model,
            max_trials=max_trials,
            evaluation_callback=progress_callback,
        )

        # Display results
        if "error" in result:
            rich_print(f"[red]Evaluation failed: {result['error']}[/red]")
            return

        # Show summary
        rich_print("\n[green]Evaluation completed![/green]")
        rich_print(f"[green]Dataset: {result['dataset']}[/green]")
        rich_print(f"[green]Total tasks: {result['total_tasks']}[/green]")
        rich_print(f"[green]Successful: {result['successful']}[/green]")
        rich_print(f"[green]Failed: {result['failed']}[/green]")

        # Show detailed results
        if result["results"]:
            table = create_table(
                "Evaluation Results",
                [
                    ("Task ID", "cyan"),
                    ("Status", "green"),
                    ("Result", "white"),
                ],
            )

            for task_result in result["results"]:
                status_color = (
                    "green" if task_result.get("status") != "failed" else "red"
                )
                status_text = task_result.get("status", "unknown")
                result_text = task_result.get(
                    "result", task_result.get("error", "No result")
                )

                if len(result_text) > 50:
                    result_text = result_text[:47] + "..."

                table.add_row(
                    task_result.get("task_id", "unknown"),
                    f"[{status_color}]{status_text}[/{status_color}]",
                    result_text,
                )

            rich_print(table)

    except Exception as e:
        rich_print(f"[red]Failed to evaluate dataset: {e}[/red]")


@datasets_app.command()
def create_config(
    output_path: Annotated[
        Path,
        Argument(help="Path to save the YAML configuration file"),
    ],
    name: Annotated[
        str | None,
        Option("--name", "-n", help="Dataset name"),
    ] = None,
    version: Annotated[
        str | None,
        Option("--version", "-v", help="Dataset version"),
    ] = None,
    path: Annotated[
        Path | None,
        Option("--path", "-p", help="Local dataset path"),
    ] = None,
    task_ids: Annotated[
        str | None,
        Option("--task-ids", help="Comma-separated task IDs or glob patterns"),
    ] = None,
    exclude_task_ids: Annotated[
        str | None,
        Option(
            "--exclude", help="Comma-separated task IDs or glob patterns to exclude"
        ),
    ] = None,
    n_tasks: Annotated[
        int | None,
        Option("--n-tasks", help="Limit number of tasks"),
    ] = None,
    category: Annotated[
        str | None,
        Option("--category", help="Filter by category"),
    ] = None,
    difficulty: Annotated[
        str | None,
        Option("--difficulty", help="Filter by difficulty"),
    ] = None,
    tags: Annotated[
        str | None,
        Option("--tags", help="Comma-separated tags to filter by"),
    ] = None,
):
    """Create a YAML dataset configuration file."""
    try:
        # Parse comma-separated values
        task_ids_list = None
        if task_ids:
            task_ids_list = [tid.strip() for tid in task_ids.split(",")]

        exclude_list = None
        if exclude_task_ids:
            exclude_list = [tid.strip() for tid in exclude_task_ids.split(",")]

        tags_list = None
        if tags:
            tags_list = [tag.strip() for tag in tags.split(",")]

        # Create configuration
        config = DatasetConfig(
            name=name,
            version=version,
            path=path,
            task_ids=task_ids_list,
            exclude_task_ids=exclude_list,
            n_tasks=n_tasks,
            category=category,
            difficulty=difficulty,
            tags=tags_list,
        )

        # Convert to YAML
        import yaml

        config_dict = config.model_dump(exclude_none=True)

        # Convert Path objects to strings for YAML serialization
        if "path" in config_dict and config_dict["path"] is not None:
            config_dict["path"] = str(config_dict["path"])
        if (
            "local_registry_path" in config_dict
            and config_dict["local_registry_path"] is not None
        ):
            config_dict["local_registry_path"] = str(config_dict["local_registry_path"])

        # Write to file
        with open(output_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        rich_print(f"[green]Dataset configuration saved to: {output_path}[/green]")

    except Exception as e:
        rich_print(f"[red]Failed to create configuration: {e}[/red]")


@datasets_app.command()
def load_config(
    config_path: Annotated[
        Path,
        Argument(help="Path to the YAML configuration file"),
    ],
):
    """Load and display information about a dataset from a YAML configuration."""
    try:
        # Load dataset from YAML
        dataset = Dataset.from_yaml(config_path)

        rich_print(f"[bold]Dataset Configuration: {config_path}[/bold]")
        rich_print(f"Number of tasks: {len(dataset)}")
        rich_print(f"Task IDs: {', '.join(dataset.task_ids)}")

        # Show task details
        if dataset.tasks:
            table = create_table("Tasks", [("Task ID", "cyan"), ("Path", "blue")])

            for task_path in dataset.tasks:
                table.add_row(task_path.name, str(task_path))

            rich_print(table)

    except Exception as e:
        rich_print(f"[red]Failed to load configuration: {e}[/red]")


# Add cache and validation subcommands to main app
datasets_app.add_typer(cache_app, name="cache", help="Manage dataset cache.")
datasets_app.add_typer(
    validation_app, name="validate", help="Validate registry and datasets."
)
