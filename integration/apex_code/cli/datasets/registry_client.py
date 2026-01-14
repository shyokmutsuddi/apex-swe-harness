"""Registry client for managing datasets."""

import fnmatch
import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import requests
from packaging import version
from pydantic import ValidationError
from rich import print as rich_print

from .constants import CACHE_DIR, DEFAULT_REGISTRY_URL
from .models import DatasetInfo, Registry, TaskInfo


class RegistryClient:
    """Client for managing dataset registry operations with caching and offline support."""

    def __init__(
        self,
        registry_url: str | None = None,
        cache_ttl: int = 3600,
        enable_offline: bool = True,
    ):
        """
        Initialize registry client.

        Args:
            registry_url: URL to the registry JSON file
            cache_ttl: Cache time-to-live in seconds (default: 1 hour)
            enable_offline: Enable offline mode with cached data
        """
        self.registry_url = registry_url or DEFAULT_REGISTRY_URL
        self.cache_ttl = cache_ttl
        self.enable_offline = enable_offline
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._registry_cache: Registry | None = None
        self._cache_metadata: dict[str, Any] | None = None

    def _get_cache_key(self) -> str:
        """Generate cache key for registry URL."""
        return hashlib.md5(self.registry_url.encode()).hexdigest()

    def _get_cache_path(self) -> Path:
        """Get cache file path for registry."""
        cache_key = self._get_cache_key()
        return self.cache_dir / f"registry_{cache_key}.json"

    def _get_cache_metadata_path(self) -> Path:
        """Get cache metadata file path."""
        cache_key = self._get_cache_key()
        return self.cache_dir / f"registry_{cache_key}_metadata.json"

    def _is_cache_valid(self) -> bool:
        """Check if cache is valid and not expired."""
        if not self.enable_offline:
            return False

        cache_path = self._get_cache_path()
        metadata_path = self._get_cache_metadata_path()

        if not cache_path.exists() or not metadata_path.exists():
            return False

        try:
            with open(metadata_path) as f:
                metadata = json.load(f)

            cache_time = datetime.fromisoformat(metadata.get("cached_at", ""))
            expiry_time = cache_time + timedelta(seconds=self.cache_ttl)

            return datetime.now(UTC) < expiry_time
        except (json.JSONDecodeError, ValueError, KeyError):
            return False

    def _has_ever_cached(self) -> bool:
        """Check if there has ever been a cache for this registry."""
        cache_path = self._get_cache_path()
        metadata_path = self._get_cache_metadata_path()
        return cache_path.exists() or metadata_path.exists()

    def _load_from_cache(self) -> Registry | None:
        """Load registry from cache if valid."""
        if not self._is_cache_valid():
            return None

        try:
            cache_path = self._get_cache_path()
            with open(cache_path) as f:
                registry_data = json.load(f)

            # Load cache metadata
            metadata_path = self._get_cache_metadata_path()
            with open(metadata_path) as f:
                self._cache_metadata = json.load(f)

            return Registry(
                datasets=[DatasetInfo(**dataset) for dataset in registry_data]
            )
        except (json.JSONDecodeError, FileNotFoundError, ValidationError):
            return None

    def _save_to_cache(self, registry: Registry) -> None:
        """Save registry to cache."""
        if not self.enable_offline:
            return

        try:
            cache_path = self._get_cache_path()
            metadata_path = self._get_cache_metadata_path()

            # Save registry data
            registry_data = [dataset.model_dump() for dataset in registry.datasets]
            with open(cache_path, "w") as f:
                json.dump(registry_data, f, indent=2)

            # Save cache metadata
            metadata = {
                "cached_at": datetime.now(UTC).isoformat(),
                "registry_url": self.registry_url,
                "cache_ttl": self.cache_ttl,
                "dataset_count": len(registry.datasets),
            }
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            self._cache_metadata = metadata
        except Exception as e:
            rich_print(f"[yellow]Warning: Failed to save registry cache: {e}[/yellow]")

    def _clear_cache(self) -> None:
        """Clear registry cache."""
        cache_path = self._get_cache_path()
        metadata_path = self._get_cache_metadata_path()

        for path in [cache_path, metadata_path]:
            if path.exists():
                try:
                    path.unlink()
                except Exception:
                    pass

        self._registry_cache = None
        self._cache_metadata = None

    def _load_any_cache(self) -> Registry | None:
        """Load the most recent cached registry regardless of URL key."""
        try:
            candidates = sorted(
                self.cache_dir.glob("registry_*_metadata.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for meta_path in candidates:
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                    # Pair with data file
                    data_path = Path(str(meta_path).replace("_metadata.json", ".json"))
                    if data_path.exists():
                        with open(data_path) as df:
                            data = json.load(df)
                        self._cache_metadata = meta
                        return Registry(datasets=[DatasetInfo(**d) for d in data])
                except Exception:
                    continue
        except Exception:
            return None
        return None

    def get_cache_info(self) -> dict[str, Any]:
        """Get cache information."""
        if not self._cache_metadata:
            return {"cached": False}

        cache_time = datetime.fromisoformat(self._cache_metadata.get("cached_at", ""))
        expiry_time = cache_time + timedelta(seconds=self.cache_ttl)
        is_valid = datetime.now(UTC) < expiry_time

        return {
            "cached": True,
            "cached_at": self._cache_metadata.get("cached_at"),
            "expires_at": expiry_time.isoformat(),
            "is_valid": is_valid,
            "dataset_count": self._cache_metadata.get("dataset_count", 0),
            "registry_url": self._cache_metadata.get("registry_url"),
        }

    def get_registry(self) -> Registry:
        """Fetch registry from remote URL or local file (with caching and offline support).

        Returns:
            Registry: The registry data

        Raises:
            requests.RequestException: If registry fetch fails and no cache available
            json.JSONDecodeError: If registry JSON is invalid
        """
        # Return cached registry if available
        if self._registry_cache is not None:
            return self._registry_cache

        # Try to load from cache first
        cached_registry = self._load_from_cache()
        if cached_registry is not None:
            self._registry_cache = cached_registry
            rich_print(
                f"[blue]Using cached registry (cached at: {self._cache_metadata.get('cached_at', 'unknown')})[/blue]"
            )
            return cached_registry

        # If offline mode and no cache, raise error (but allow first fetch for local files)
        if (
            self.enable_offline
            and not self._has_ever_cached()
            and not self.registry_url.startswith("file://")
        ):
            raise ConnectionError(
                "No cached registry available and offline mode is enabled"
            )

        try:
            # Handle local file URLs
            if self.registry_url.startswith("file://"):
                file_path = Path(self.registry_url[7:])  # Remove "file://" prefix
                if not file_path.exists():
                    # Try any cache fallback when offline
                    if self.enable_offline:
                        fallback = self._load_any_cache()
                        if fallback is not None:
                            rich_print(
                                "[yellow]Registry file missing; using most recent cached registry[/yellow]"
                            )
                            self._registry_cache = fallback
                            return fallback
                    raise FileNotFoundError(f"Registry file not found: {file_path}")
                with open(file_path) as f:
                    data = json.load(f)
                registry = Registry.from_json_list(data)
            else:
                # Handle remote URLs
                response = requests.get(self.registry_url, timeout=30)
                response.raise_for_status()
                registry = Registry.from_json_list(response.json())

            # Cache the registry
            self._save_to_cache(registry)
            self._registry_cache = registry
            return registry

        except Exception as e:
            # If we have cached data, use it even if expired
            if self.enable_offline:
                cached_registry = self._load_from_cache() or self._load_any_cache()
                if cached_registry is not None:
                    rich_print(
                        f"[yellow]Network error, using cached registry: {e}[/yellow]"
                    )
                    self._registry_cache = cached_registry
                    return cached_registry

            rich_print(f"[red]Failed to fetch registry: {e}[/red]")
            raise

    def get_datasets(self) -> list[DatasetInfo]:
        """Get all datasets from registry.

        Returns:
            list[DatasetInfo]: List of all datasets
        """
        registry = self.get_registry()
        return registry.datasets

    def get_dataset(self, name: str, version: str = "head") -> DatasetInfo | None:
        """Get specific dataset by name and version.

        Args:
            name: Dataset name
            version: Dataset version or version range (default: "head")

        Returns:
            Optional[DatasetInfo]: Dataset info if found, None otherwise
        """
        registry = self.get_registry()
        return registry.get_dataset(name, version)

    def get_latest_dataset(
        self, name: str, apex_code_version: str = "0.1.0"
    ) -> DatasetInfo | None:
        """Get the latest compatible dataset by name.

        Args:
            name: Dataset name
            apex_code_version: Current apex-code version for compatibility check

        Returns:
            Optional[DatasetInfo]: Latest compatible dataset if found, None otherwise
        """
        registry = self.get_registry()
        compatible_datasets = []

        for dataset in registry.datasets:
            if dataset.name == name and dataset.is_compatible_with(apex_code_version):
                compatible_datasets.append(dataset)

        if not compatible_datasets:
            return None

        # Sort by version (head is always latest)
        def sort_key(dataset):
            if dataset.version == "head":
                return (1, "head")  # head is always first
            else:
                from packaging import version

                try:
                    return (0, version.parse(dataset.version))
                except version.InvalidVersion:
                    return (-1, dataset.version)

        compatible_datasets.sort(key=sort_key, reverse=True)
        return compatible_datasets[0]

    def get_tasks_with_metadata(
        self,
        name: str,
        version: str = "head",
        category: str | None = None,
        difficulty: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Get tasks with metadata filtering.

        Args:
            name: Dataset name
            version: Dataset version
            category: Category to filter by
            difficulty: Difficulty to filter by
            tags: Tags to filter by

        Returns:
            Dict containing filtered tasks and metadata
        """
        dataset = self.get_dataset(name, version)
        if not dataset:
            return {"tasks": [], "metadata": {}}

        # Use streaming approach to get filtered tasks
        filtered_tasks = self._get_filtered_tasks_streaming(
            dataset, None, category, difficulty, tags
        )

        # Get metadata for each task using streaming
        task_metadata = {}
        for task_id in filtered_tasks:
            metadata = self._get_single_task_metadata(dataset, task_id)
            if metadata:
                task_metadata[task_id] = metadata

        return {"tasks": filtered_tasks, "metadata": task_metadata}

    def list_tasks(
        self,
        name: str,
        version: str = "head",
        category: str | None = None,
        difficulty: str | None = None,
        tags: list[str] | None = None,
        exclude: list[str] | None = None,
        n_tasks: int | None = None,
    ) -> list[dict[str, Any]]:
        """List tasks with optional filtering.

        Args:
            name: Dataset name
            version: Dataset version
            category: Category to filter by
            difficulty: Difficulty to filter by
            tags: Tags to filter by

        Returns:
            List of task information dictionaries
        """
        result = self.get_tasks_with_metadata(name, version, category, difficulty, tags)

        # Apply CLI-level exclusions if provided
        filtered_metadata = result["metadata"]
        if exclude:
            filtered_metadata = {
                task_id: task_info
                for task_id, task_info in result["metadata"].items()
                if not any(fnmatch.fnmatch(task_id, pattern) for pattern in exclude)
            }

        task_list = [
            {
                "task_id": task_id,
                "category": task_info.category,
                "difficulty": task_info.difficulty,
                "tags": task_info.tags,
                "description": task_info.description,
                "estimated_time": task_info.estimated_time,
                "dependencies": task_info.dependencies,
            }
            for task_id, task_info in filtered_metadata.items()
        ]

        # Apply n_tasks limit if specified
        if n_tasks is not None and n_tasks > 0:
            task_list = task_list[:n_tasks]

        return task_list

    def evaluate_dataset_directly(
        self,
        name: str,
        version: str = "head",
        task_subset: list[str] | None = None,
        category: str | None = None,
        difficulty: str | None = None,
        tags: list[str] | None = None,
        exclude: list[str] | None = None,
        n_tasks: int | None = None,
        agent_type: str = "claude-sonnet-4-20250514",
        max_trials: int = 3,
        evaluation_callback=None,
    ) -> dict[str, Any]:
        """Evaluate tasks directly from remote dataset without local download.

        Args:
            name: Dataset name
            version: Dataset version
            task_subset: List of task IDs or glob patterns to run
            category: Category to filter by
            difficulty: Difficulty to filter by
            tags: Tags to filter by
            evaluation_callback: Callback function for each task evaluation

        Returns:
            Dict containing evaluation results and metadata
        """
        dataset = self.get_dataset(name, version)
        if not dataset:
            return {"error": f"Dataset {name}=={version} not found", "results": []}

        # Get filtered tasks using streaming approach
        try:
            filtered_tasks = self._get_filtered_tasks_streaming(
                dataset, task_subset, category, difficulty, tags
            )

            if not filtered_tasks:
                return {"error": "No tasks found matching criteria", "results": []}

            # Apply n_tasks limit if specified
            if n_tasks is not None and n_tasks > 0:
                filtered_tasks = filtered_tasks[:n_tasks]

            # Evaluate tasks one by one
            results = []
            for i, task_id in enumerate(filtered_tasks):
                try:
                    if evaluation_callback:
                        evaluation_callback(i + 1, len(filtered_tasks), task_id)

                    # Get single task for evaluation
                    task_result = self._evaluate_single_task(
                        dataset, task_id, agent_type=agent_type, max_trials=max_trials
                    )
                    results.append(task_result)

                except Exception as e:
                    results.append(
                        {"task_id": task_id, "error": str(e), "status": "failed"}
                    )

            return {
                "dataset": f"{name}=={version}",
                "total_tasks": len(filtered_tasks),
                "successful": len([r for r in results if r.get("status") != "failed"]),
                "failed": len([r for r in results if r.get("status") == "failed"]),
                "results": results,
            }

        except Exception as e:
            return {"error": f"Failed to evaluate dataset: {str(e)}", "results": []}

    def _get_filtered_tasks_streaming(
        self,
        dataset: "DatasetInfo",
        task_subset: list[str] | None = None,
        category: str | None = None,
        difficulty: str | None = None,
        tags: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> list[str]:
        """Get filtered task list using true streaming approach - no downloads."""

        # True streaming: discover tasks directly from source without downloading
        available_tasks = self._discover_tasks_from_source(dataset)

        if not available_tasks:
            rich_print(f"[yellow]No tasks found in dataset {dataset.name}[/yellow]")
            return []

        # Apply filters in memory
        filtered_tasks = self._apply_filters_in_memory(
            available_tasks, dataset, task_subset, category, difficulty, tags, exclude
        )

        return filtered_tasks

    def _discover_tasks_from_source(self, dataset: "DatasetInfo") -> list[str]:
        """Discover tasks directly from source without downloading."""

        # Handle local file URLs
        if dataset.github_url.startswith("file://"):
            return self._discover_local_tasks(dataset)

        # Handle remote URLs - use git ls-remote for efficiency
        return self._discover_remote_tasks(dataset)

    def _discover_local_tasks(self, dataset: "DatasetInfo") -> list[str]:
        """Discover tasks from local file system."""
        try:
            # Parse file:// URL
            file_path = Path(dataset.github_url[7:])  # Remove "file://" prefix
            tasks_dir = file_path / dataset.dataset_path.lstrip("./")

            if not tasks_dir.exists():
                rich_print(f"[red]Local tasks directory not found: {tasks_dir}[/red]")
                return []

            # Discover tasks
            tasks = []
            for item in tasks_dir.iterdir():
                if item.is_dir() and (item / "task.yaml").exists():
                    tasks.append(item.name)
                    rich_print(f"[blue]Found local task: {item.name}[/blue]")

            return tasks

        except Exception as e:
            rich_print(f"[red]Error discovering local tasks: {e}[/red]")
            return []

    def _discover_remote_tasks(self, dataset: "DatasetInfo") -> list[str]:
        """Discover tasks from remote repository using git ls-remote."""
        try:
            # Use git ls-remote to get directory listing without full clone
            cmd = ["git", "ls-remote", "--heads", dataset.github_url, dataset.branch]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                rich_print(
                    f"[red]Failed to discover remote tasks: {result.stderr}[/red]"
                )
                return []

            # For now, fall back to download for remote repos
            # TODO: Implement true remote streaming with git sparse-checkout
            rich_print(
                "[yellow]Remote streaming not yet implemented, using download fallback[/yellow]"
            )
            return self._discover_remote_tasks_fallback(dataset)

        except Exception as e:
            rich_print(f"[red]Error discovering remote tasks: {e}[/red]")
            return []

    def _discover_remote_tasks_fallback(self, dataset: "DatasetInfo") -> list[str]:
        """Fallback: discover tasks by downloading (for remote repos)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dataset_path = self.download_dataset(
                dataset.name, dataset.version, temp_path
            )
            tasks_dir = dataset_path / dataset.dataset_path.lstrip("./")

            tasks = []
            if tasks_dir.exists():
                for item in tasks_dir.iterdir():
                    if item.is_dir() and (item / "task.yaml").exists():
                        tasks.append(item.name)

            return tasks

    def _apply_filters_in_memory(
        self,
        available_tasks: list[str],
        dataset: "DatasetInfo",
        task_subset: list[str] | None = None,
        category: str | None = None,
        difficulty: str | None = None,
        tags: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> list[str]:
        """Apply all filters in memory without downloading task metadata."""

        filtered_tasks = available_tasks.copy()

        # Apply task subset filtering (glob patterns)
        if task_subset:
            subset_filtered = []
            for task_id in filtered_tasks:
                for pattern in task_subset:
                    if fnmatch.fnmatch(task_id, pattern):
                        subset_filtered.append(task_id)
                        break
            filtered_tasks = subset_filtered

        # Apply dataset-level task exclusions
        filtered_tasks = dataset.apply_task_exclusions(filtered_tasks)

        # Apply CLI-level exclusions
        if exclude:
            filtered_tasks = [
                task_id
                for task_id in filtered_tasks
                if not any(fnmatch.fnmatch(task_id, pattern) for pattern in exclude)
            ]

        # Apply metadata filtering (requires task.yaml access)
        if category or difficulty or tags:
            metadata_filtered = self._filter_by_metadata_streaming(
                filtered_tasks, dataset, category, difficulty, tags
            )
            filtered_tasks = metadata_filtered

        return filtered_tasks

    def _filter_by_metadata_streaming(
        self,
        task_ids: list[str],
        dataset: "DatasetInfo",
        category: str | None = None,
        difficulty: str | None = None,
        tags: list[str] | None = None,
    ) -> list[str]:
        """Filter tasks by metadata using streaming approach."""

        filtered_tasks = []

        for task_id in task_ids:
            # Get task metadata without downloading entire dataset
            task_metadata = self._get_single_task_metadata(dataset, task_id)

            if task_metadata and task_metadata.matches_filter(
                category, difficulty, tags
            ):
                filtered_tasks.append(task_id)

        return filtered_tasks

    def _get_single_task_metadata(
        self, dataset: "DatasetInfo", task_id: str
    ) -> Optional["TaskInfo"]:
        """Get metadata for a single task without downloading entire dataset."""

        try:
            # Handle local tasks
            if dataset.github_url.startswith("file://"):
                return self._get_local_task_metadata(dataset, task_id)

            # Handle remote tasks - use git show for single file
            return self._get_remote_task_metadata(dataset, task_id)

        except Exception as e:
            rich_print(f"[yellow]Could not get metadata for {task_id}: {e}[/yellow]")
            return None

    def _get_local_task_metadata(
        self, dataset: "DatasetInfo", task_id: str
    ) -> Optional["TaskInfo"]:
        """Get metadata for local task."""
        try:
            file_path = Path(dataset.github_url[7:])
            task_yaml_path = (
                file_path / dataset.dataset_path.lstrip("./") / task_id / "task.yaml"
            )

            if not task_yaml_path.exists():
                return None

            import yaml

            with open(task_yaml_path) as f:
                task_data = yaml.safe_load(f)

            return TaskInfo(
                task_id=task_id,
                category=task_data.get("category"),
                difficulty=task_data.get("difficulty"),
                tags=task_data.get("tags", []),
                description=task_data.get("description"),
                estimated_time=task_data.get("estimated_time"),
                dependencies=task_data.get("dependencies", []),
            )

        except Exception as e:
            rich_print(
                f"[yellow]Error reading local task metadata for {task_id}: {e}[/yellow]"
            )
            return None

    def _get_remote_task_metadata(
        self, dataset: "DatasetInfo", task_id: str
    ) -> Optional["TaskInfo"]:
        """Get metadata for remote task using git show."""
        try:
            # Use git show to get single file without full clone
            task_yaml_path = f"{dataset.dataset_path.lstrip('./')}/{task_id}/task.yaml"
            cmd = ["git", "show", f"{dataset.github_url}:{task_yaml_path}"]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                return None

            import yaml

            task_data = yaml.safe_load(result.stdout)

            return TaskInfo(
                task_id=task_id,
                category=task_data.get("category"),
                difficulty=task_data.get("difficulty"),
                tags=task_data.get("tags", []),
                description=task_data.get("description"),
                estimated_time=task_data.get("estimated_time"),
                dependencies=task_data.get("dependencies", []),
            )

        except Exception as e:
            rich_print(
                f"[yellow]Error reading remote task metadata for {task_id}: {e}[/yellow]"
            )
            return None

    def _evaluate_single_task(
        self,
        dataset: "DatasetInfo",
        task_id: str,
        agent_type: str = "claude-sonnet-4-20250514",
        max_trials: int = 3,
    ) -> dict[str, Any]:
        """Evaluate a single task from the dataset using real EvaluationExecutor."""
        try:
            # Import here to avoid circular imports
            import tempfile

            from ...harness.executor import EvaluationExecutor
            from ...harness.models import EvaluationConfig, ModelType

            # Create temporary directory for task evaluation
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Get the specific task path (streaming for local, download for remote)
                task_path = self._get_single_task_path(dataset, task_id)
                if not task_path:
                    return {
                        "task_id": task_id,
                        "status": "failed",
                        "error": "Failed to download task",
                        "metadata": {
                            "dataset": dataset.name,
                            "version": dataset.version,
                        },
                    }

                # Set up evaluation configuration
                config = EvaluationConfig(
                    run_id=f"direct_eval_{task_id}_{int(time.time())}",
                    task_id=task_id,
                    tasks_dir=task_path.parent,
                    model=ModelType(agent_type),
                    max_trials=max_trials,
                    timeout=1800,  # 30 minutes per trial
                )

                # Create evaluation executor
                executor = EvaluationExecutor(max_workers=1)

                # Run evaluation
                result = executor.execute_run(config)

                # Extract results
                if result.trials:
                    trial = result.trials[0]  # Get first trial
                    return {
                        "task_id": task_id,
                        "status": "completed" if trial.success else "failed",
                        "success": trial.success,
                        "score": trial.score,
                        "error": trial.error if not trial.success else None,
                        "execution_time": trial.execution_time,
                        "metadata": {
                            "dataset": dataset.name,
                            "version": dataset.version,
                            "agent": agent_type,
                            "trial_count": len(result.trials),
                        },
                    }
                else:
                    return {
                        "task_id": task_id,
                        "status": "failed",
                        "error": "No trials completed",
                        "metadata": {
                            "dataset": dataset.name,
                            "version": dataset.version,
                        },
                    }

        except Exception as e:
            return {
                "task_id": task_id,
                "status": "failed",
                "error": str(e),
                "metadata": {"dataset": dataset.name, "version": dataset.version},
            }

    def _get_single_task_path(
        self, dataset: "DatasetInfo", task_id: str
    ) -> Path | None:
        """Get path to a single task without downloading."""
        try:
            # Handle local tasks
            if dataset.github_url.startswith("file://"):
                file_path = Path(dataset.github_url[7:])
                task_dir = file_path / dataset.dataset_path.lstrip("./") / task_id

                if task_dir.exists() and (task_dir / "task.yaml").exists():
                    return task_dir
                else:
                    return None

            # For remote tasks, we still need to download (for now)
            # TODO: Implement git sparse-checkout for single task
            rich_print(
                "[yellow]Remote single task access not yet implemented, using download fallback[/yellow]"
            )
            return self._download_single_task_fallback(dataset, task_id)

        except Exception as e:
            rich_print(f"[yellow]Error getting task path for {task_id}: {e}[/yellow]")
            return None

    def _download_single_task_fallback(
        self, dataset: "DatasetInfo", task_id: str
    ) -> Path | None:
        """Fallback: download single task (for remote repos)."""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                dataset_path = self.download_dataset(
                    dataset.name, dataset.version, temp_path
                )
                tasks_dir = dataset_path / dataset.dataset_path.lstrip("./")

                # Find the specific task
                task_dir = tasks_dir / task_id
                if task_dir.exists() and (task_dir / "task.yaml").exists():
                    return task_dir
                else:
                    return None

        except Exception:
            return None

    def validate_registry(self, apex_code_version: str = "0.1.0") -> dict[str, Any]:
        """Validate registry integrity and compatibility.

        Args:
            apex_code_version: Current apex-code version for compatibility checks

        Returns:
            Dict containing validation results and issues
        """
        validation_result = {
            "valid": True,
            "issues": [],
            "warnings": [],
            "stats": {
                "total_datasets": 0,
                "compatible_datasets": 0,
                "incompatible_datasets": 0,
                "duplicate_datasets": 0,
                "invalid_versions": 0,
            },
        }

        try:
            registry = self.get_registry()
            validation_result["stats"]["total_datasets"] = len(registry.datasets)

            # Track dataset names and versions for duplicate detection
            dataset_combinations = {}

            for dataset in registry.datasets:
                # Check for duplicate dataset name+version combinations
                key = f"{dataset.name}=={dataset.version}"
                if key in dataset_combinations:
                    validation_result["issues"].append(
                        f"Duplicate dataset: {key} (found {dataset_combinations[key] + 1} times)"
                    )
                    validation_result["stats"]["duplicate_datasets"] += 1
                    validation_result["valid"] = False
                else:
                    dataset_combinations[key] = 1

                # Validate dataset version format
                if dataset.version != "head":
                    try:
                        version.parse(dataset.version)
                    except version.InvalidVersion:
                        validation_result["issues"].append(
                            f"Invalid version format for {dataset.name}: {dataset.version}"
                        )
                        validation_result["stats"]["invalid_versions"] += 1
                        validation_result["valid"] = False

                # Check compatibility
                if dataset.is_compatible_with(apex_code_version):
                    validation_result["stats"]["compatible_datasets"] += 1
                else:
                    validation_result["stats"]["incompatible_datasets"] += 1
                    validation_result["warnings"].append(
                        f"Dataset {dataset.name}=={dataset.version} incompatible with apex-code {apex_code_version}"
                    )

                # Validate required fields
                if not dataset.github_url:
                    validation_result["issues"].append(
                        f"Missing github_url for {dataset.name}=={dataset.version}"
                    )
                    validation_result["valid"] = False

                if not dataset.dataset_path:
                    validation_result["issues"].append(
                        f"Missing dataset_path for {dataset.name}=={dataset.version}"
                    )
                    validation_result["valid"] = False

                # Validate URL format
                if dataset.github_url and not self._is_valid_url(dataset.github_url):
                    validation_result["issues"].append(
                        f"Invalid URL format for {dataset.name}=={dataset.version}: {dataset.github_url}"
                    )
                    validation_result["valid"] = False

            # Check for empty registry
            if validation_result["stats"]["total_datasets"] == 0:
                validation_result["issues"].append("Registry is empty")
                validation_result["valid"] = False

        except Exception as e:
            validation_result["valid"] = False
            validation_result["issues"].append(f"Registry validation failed: {str(e)}")

        return validation_result

    def validate_remote_accessibility(self) -> dict[str, Any]:
        """Validate that remote registry URLs are accessible.

        Returns:
            Dict containing accessibility validation results
        """
        accessibility_result = {
            "accessible": True,
            "issues": [],
            "warnings": [],
            "stats": {
                "total_urls": 0,
                "accessible_urls": 0,
                "inaccessible_urls": 0,
                "timeout_urls": 0,
            },
        }

        try:
            registry = self.get_registry()
            unique_urls = set()

            # Collect unique URLs from datasets
            for dataset in registry.datasets:
                if dataset.github_url and not dataset.github_url.startswith("file://"):
                    unique_urls.add(dataset.github_url)

            accessibility_result["stats"]["total_urls"] = len(unique_urls)

            # Test each unique URL
            for url in unique_urls:
                try:
                    response = requests.head(url, timeout=10)
                    if response.status_code == 200:
                        accessibility_result["stats"]["accessible_urls"] += 1
                    else:
                        accessibility_result["stats"]["inaccessible_urls"] += 1
                        accessibility_result["issues"].append(
                            f"URL returned status {response.status_code}: {url}"
                        )
                        accessibility_result["accessible"] = False
                except requests.exceptions.Timeout:
                    accessibility_result["stats"]["timeout_urls"] += 1
                    accessibility_result["warnings"].append(
                        f"Timeout accessing URL: {url}"
                    )
                except requests.exceptions.RequestException as e:
                    accessibility_result["stats"]["inaccessible_urls"] += 1
                    accessibility_result["issues"].append(
                        f"Cannot access URL {url}: {str(e)}"
                    )
                    accessibility_result["accessible"] = False

        except Exception as e:
            accessibility_result["accessible"] = False
            accessibility_result["issues"].append(
                f"Accessibility validation failed: {str(e)}"
            )

        return accessibility_result

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL has valid format.

        Args:
            url: URL to validate

        Returns:
            bool: True if URL format is valid
        """
        try:
            result = urlparse(url)
            # For file:// URLs, netloc can be empty (file:///path)
            # For other URLs, both scheme and netloc are required
            if result.scheme == "file":
                return bool(result.scheme and result.path)
            else:
                return all([result.scheme, result.netloc])
        except Exception:
            return False

    def download_dataset(
        self,
        name: str,
        version: str = "head",
        output_dir: Path | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Download dataset from GitHub repository.

        Args:
            name: Dataset name
            version: Dataset version
            output_dir: Output directory (default: cache directory)
            overwrite: Whether to overwrite existing dataset

        Returns:
            Path: Path to downloaded dataset

        Raises:
            ValueError: If dataset not found
            subprocess.CalledProcessError: If git operations fail
        """
        dataset = self.get_dataset(name, version)
        if not dataset:
            raise ValueError(f"Dataset '{name}' version '{version}' not found")

        # Determine output directory
        if output_dir is None:
            output_dir = self.cache_dir / name / version
        else:
            output_dir = output_dir / name / version

        # Check if already exists
        if output_dir.exists() and not overwrite:
            rich_print(f"[yellow]Dataset already exists at {output_dir}[/yellow]")
            return output_dir

        # Create output directory
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        if output_dir.exists():
            shutil.rmtree(output_dir)

        rich_print(f"[blue]Downloading dataset '{name}' version '{version}'...[/blue]")

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Clone repository
                subprocess.run(
                    [
                        "git",
                        "clone",
                        "--branch",
                        dataset.branch,
                        dataset.github_url,
                        str(temp_path),
                    ],
                    check=True,
                    capture_output=True,
                )

                # Checkout specific commit if not head
                if dataset.commit_hash != "head":
                    subprocess.run(
                        ["git", "reset", "--hard", dataset.commit_hash],
                        check=True,
                        capture_output=True,
                        cwd=temp_path,
                    )

                # Remove git history
                subprocess.run(
                    ["rm", "-rf", ".git"],
                    check=True,
                    capture_output=True,
                    cwd=temp_path,
                )

                # Copy dataset to output directory
                source_path = temp_path / dataset.dataset_path
                if source_path.exists():
                    shutil.copytree(source_path, output_dir)
                else:
                    rich_print(
                        f"[red]Dataset path '{dataset.dataset_path}' not found in repository[/red]"
                    )
                    raise ValueError(f"Dataset path not found: {dataset.dataset_path}")

                rich_print(
                    f"[green]Successfully downloaded dataset to {output_dir}[/green]"
                )
                return output_dir

        except subprocess.CalledProcessError as e:
            rich_print(f"[red]Git operation failed: {e}[/red]")
            raise
        except Exception as e:
            rich_print(f"[red]Download failed: {e}[/red]")
            raise
