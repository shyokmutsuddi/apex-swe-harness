"""Data models for datasets."""

import fnmatch
from datetime import UTC
from pathlib import Path
from typing import Any, Self

from packaging import specifiers, version
from pydantic import BaseModel, field_validator, model_validator


class TaskInfo(BaseModel):
    """Individual task metadata information."""

    task_id: str
    category: str | None = None
    difficulty: str | None = None
    tags: list[str] | None = None
    description: str | None = None
    estimated_time: int | None = None  # in minutes
    dependencies: list[str] | None = None

    def matches_filter(
        self,
        category: str | None = None,
        difficulty: str | None = None,
        tags: list[str] | None = None,
    ) -> bool:
        """Check if task matches filtering criteria.

        Args:
            category: Category to filter by (exact match)
            difficulty: Difficulty to filter by (exact match)
            tags: Tags to filter by (any match)

        Returns:
            bool: True if task matches all provided criteria
        """
        # Category filter
        if category is not None:
            if self.category != category:
                return False

        # Difficulty filter
        if difficulty is not None:
            if self.difficulty != difficulty:
                return False

        # Tags filter (any tag match)
        if tags is not None and tags:
            if not self.tags:
                return False
            if not any(tag in self.tags for tag in tags):
                return False

        return True


class DatasetConfig(BaseModel):
    """Configuration for loading a dataset."""

    # Dataset identification
    name: str | None = None
    version: str | None = None
    registry_url: str | None = None
    local_registry_path: Path | None = None

    # Local dataset path
    path: Path | None = None

    # Task filtering
    task_ids: list[str] | None = None
    n_tasks: int | None = None
    exclude_task_ids: list[str] | None = None

    # Metadata filtering
    category: str | None = None
    difficulty: str | None = None
    tags: list[str] | None = None

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        """Validate the dataset configuration."""
        if self.task_ids is not None and self.n_tasks is not None:
            raise ValueError("Cannot specify both task_ids and n_tasks")

        if self.path is None and (self.version is None or self.name is None):
            raise ValueError("If path is not set, both version and name must be set")
        elif self.path is not None and (
            self.version is not None or self.name is not None
        ):
            raise ValueError("If path is set, version and name should not be set")

        return self


class Dataset:
    """A class for loading and iterating over tasks in a dataset."""

    def __init__(self, config: DatasetConfig):
        """Initialize the dataset from a configuration."""
        self.config = config
        self._tasks: list[Path] = []
        self._path: Path | None = None

        self._maybe_cache_dataset()
        self._init_dataset()

    def _maybe_cache_dataset(self) -> None:
        """Prepare the dataset path by using local path or caching from server."""
        if self.config.path is not None:
            self._path = self.config.path
            return

        if self.config.name is None or self.config.version is None:
            raise ValueError(
                "Both name and version must be set when path is not provided"
            )

        # Import here to avoid circular imports
        from .registry_client import RegistryClient

        client = RegistryClient(registry_url=self.config.registry_url)

        self._path = client.download_dataset(self.config.name, self.config.version)

    def _init_dataset(self) -> None:
        """Initialize the dataset by loading and filtering task paths."""
        if self._path is None:
            raise ValueError("Dataset path not set")

        # Get all task paths
        all_tasks = [item for item in self._path.iterdir() if item.is_dir()]

        # Apply task filtering
        if self.config.task_ids is not None:
            # Filter by specific task IDs or glob patterns
            filtered_tasks = []
            for task_path in all_tasks:
                for pattern in self.config.task_ids:
                    if fnmatch.fnmatch(task_path.name, pattern):
                        filtered_tasks.append(task_path)
                        break
            all_tasks = filtered_tasks

        # Apply exclusions
        if self.config.exclude_task_ids is not None:
            all_tasks = [
                task_path
                for task_path in all_tasks
                if not any(
                    fnmatch.fnmatch(task_path.name, pattern)
                    for pattern in self.config.exclude_task_ids
                )
            ]

        # Apply n_tasks limit
        if self.config.n_tasks is not None:
            all_tasks = all_tasks[: self.config.n_tasks]

        self._tasks = all_tasks

    def __iter__(self):
        """Iterate over the tasks in the dataset."""
        return iter(self._tasks)

    def __len__(self) -> int:
        """Get the number of tasks in the dataset."""
        return len(self._tasks)

    @property
    def tasks(self) -> list[Path]:
        """Get the list of tasks in the dataset."""
        return self._tasks

    @property
    def task_ids(self) -> list[str]:
        """Get the list of task IDs in the dataset."""
        return [path.name for path in self._tasks]

    @classmethod
    def from_config(cls, config: DatasetConfig) -> "Dataset":
        """Create a Dataset instance from a DatasetConfig."""
        return cls(config)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "Dataset":
        """Create a Dataset instance from a YAML configuration file."""
        import yaml

        with open(yaml_path) as f:
            config_data = yaml.safe_load(f)

        # Convert string paths back to Path objects
        if "path" in config_data and isinstance(config_data["path"], str):
            config_data["path"] = Path(config_data["path"])
        if "local_registry_path" in config_data and isinstance(
            config_data["local_registry_path"], str
        ):
            config_data["local_registry_path"] = Path(
                config_data["local_registry_path"]
            )

        config = DatasetConfig(**config_data)
        return cls.from_config(config)


class DatasetInfo(BaseModel):
    """Dataset metadata information."""

    name: str
    version: str
    description: str | None = None
    github_url: str
    dataset_path: str = "./tasks"
    branch: str = "main"
    commit_hash: str = "head"
    apex_code_version: str = ">=0.1.0"
    task_subset: list[str] | None = None
    exclude_task_ids: list[str] | None = None

    # Enhanced metadata fields
    author: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    license: str | None = None
    size: int | None = None  # in bytes
    task_count: int | None = None
    tags: list[str] | None = None
    homepage: str | None = None
    documentation: str | None = None

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate that version is a valid semantic version or 'head'."""
        if v == "head":
            return v
        try:
            version.parse(v)
            return v
        except version.InvalidVersion:
            raise ValueError(
                f"Invalid version format: {v}. Must be semantic version (x.y.z) or 'head'"
            )

    @field_validator("apex_code_version")
    @classmethod
    def validate_apex_code_version(cls, v: str) -> str:
        """Validate that apex_code_version is a valid version specifier."""
        try:
            specifiers.SpecifierSet(v)
            return v
        except specifiers.InvalidSpecifier:
            raise ValueError(
                f"Invalid version specifier: {v}. Must be valid version range (e.g., '>=1.0.0')"
            )

    @field_validator("created_at", "updated_at")
    @classmethod
    def validate_timestamp(cls, v: str | None) -> str | None:
        """Validate timestamp format (ISO 8601)."""
        if v is None:
            return v
        try:
            from datetime import datetime

            datetime.fromisoformat(v.replace("Z", "+00:00"))
            return v
        except ValueError:
            raise ValueError(
                f"Invalid timestamp format: {v}. Must be ISO 8601 format (e.g., '2023-01-01T00:00:00Z')"
            )

    @field_validator("size")
    @classmethod
    def validate_size(cls, v: int | None) -> int | None:
        """Validate size is non-negative."""
        if v is not None and v < 0:
            raise ValueError(f"Size must be non-negative, got: {v}")
        return v

    @field_validator("task_count")
    @classmethod
    def validate_task_count(cls, v: int | None) -> int | None:
        """Validate task count is non-negative."""
        if v is not None and v < 0:
            raise ValueError(f"Task count must be non-negative, got: {v}")
        return v

    def is_compatible_with(self, apex_code_version: str) -> bool:
        """Check if dataset is compatible with apex-code version.

        Args:
            apex_code_version: The apex-code version to check

        Returns:
            bool: True if compatible
        """
        try:
            spec_set = specifiers.SpecifierSet(self.apex_code_version)
            return spec_set.contains(apex_code_version)
        except (specifiers.InvalidSpecifier, version.InvalidVersion):
            return False

    def version_compare(self, other_version: str) -> int:
        """Compare this dataset version with another version.

        Args:
            other_version: Version to compare with

        Returns:
            int: -1 if this < other, 0 if equal, 1 if this > other
        """
        if self.version == "head":
            return 1  # head is always considered newer
        if other_version == "head":
            return -1

        try:
            this_ver = version.parse(self.version)
            other_ver = version.parse(other_version)
            if this_ver < other_ver:
                return -1
            elif this_ver > other_ver:
                return 1
            else:
                return 0
        except version.InvalidVersion:
            return 0  # Invalid versions are considered equal

    def filter_tasks(self, task_ids: list[str]) -> list[str]:
        """Filter task IDs based on dataset subset.

        Args:
            task_ids: List of available task IDs

        Returns:
            List[str]: Filtered task IDs based on subset rules
        """
        if not self.task_subset:
            return task_ids

        filtered_tasks = []
        for task_id in task_ids:
            # Check if task matches any subset pattern
            for pattern in self.task_subset:
                if fnmatch.fnmatch(task_id, pattern):
                    filtered_tasks.append(task_id)
                    break

        return filtered_tasks

    def get_task_metadata(self, tasks_dir: Path) -> dict[str, TaskInfo]:
        """Get metadata for all tasks in the dataset.

        Args:
            tasks_dir: Directory containing tasks

        Returns:
            Dict mapping task_id to TaskInfo
        """
        task_metadata = {}

        if not tasks_dir.exists():
            return task_metadata

        for task_dir in tasks_dir.iterdir():
            if task_dir.is_dir():
                task_id = task_dir.name
                task_yaml = task_dir / "task.yaml"

                if task_yaml.exists():
                    try:
                        import yaml

                        with open(task_yaml) as f:
                            task_data = yaml.safe_load(f)

                        # Extract metadata from task.yaml
                        task_info = TaskInfo(
                            task_id=task_id,
                            category=task_data.get("category"),
                            difficulty=task_data.get("difficulty"),
                            tags=task_data.get("tags", []),
                            description=task_data.get("description"),
                            estimated_time=task_data.get("estimated_time"),
                            dependencies=task_data.get("dependencies", []),
                        )
                        task_metadata[task_id] = task_info
                    except Exception:
                        # If we can't parse the task.yaml, create basic metadata
                        task_metadata[task_id] = TaskInfo(task_id=task_id)

        return task_metadata

    def filter_tasks_by_metadata(
        self,
        tasks_dir: Path,
        category: str | None = None,
        difficulty: str | None = None,
        tags: list[str] | None = None,
    ) -> list[str]:
        """Filter tasks by metadata criteria.

        Args:
            tasks_dir: Directory containing tasks
            category: Category to filter by
            difficulty: Difficulty to filter by
            tags: Tags to filter by

        Returns:
            List of task IDs that match the criteria
        """
        task_metadata = self.get_task_metadata(tasks_dir)
        filtered_tasks = []

        for task_id, task_info in task_metadata.items():
            if task_info.matches_filter(
                category=category, difficulty=difficulty, tags=tags
            ):
                filtered_tasks.append(task_id)

        return filtered_tasks

    def calculate_metadata(self, tasks_dir: Path) -> dict[str, Any]:
        """Calculate metadata for the dataset.

        Args:
            tasks_dir: Directory containing tasks

        Returns:
            Dict containing calculated metadata
        """
        metadata = {}

        if not tasks_dir.exists():
            return metadata

        # Calculate task count
        task_count = 0
        total_size = 0

        for task_dir in tasks_dir.iterdir():
            if task_dir.is_dir() and (task_dir / "task.yaml").exists():
                task_count += 1
                # Calculate directory size
                for file_path in task_dir.rglob("*"):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size

        metadata["task_count"] = task_count
        metadata["size"] = total_size

        # Set updated_at to current timestamp if not already set
        if not self.updated_at:
            from datetime import datetime

            metadata["updated_at"] = datetime.now(UTC).isoformat()

        return metadata

    def update_metadata(self, tasks_dir: Path) -> "DatasetInfo":
        """Update dataset metadata with calculated values.

        Args:
            tasks_dir: Directory containing tasks

        Returns:
            Updated DatasetInfo instance
        """
        calculated = self.calculate_metadata(tasks_dir)

        # Create updated instance
        updated_data = self.model_dump()
        updated_data.update(calculated)

        return DatasetInfo(**updated_data)

    def apply_task_exclusions(self, task_ids: list[str]) -> list[str]:
        """Apply task exclusions based on exclude_task_ids patterns.

        Args:
            task_ids: List of task IDs to filter

        Returns:
            Filtered list of task IDs with exclusions applied
        """
        if not self.exclude_task_ids:
            return task_ids

        import fnmatch

        return [
            task_id
            for task_id in task_ids
            if not any(
                fnmatch.fnmatch(task_id, pattern) for pattern in self.exclude_task_ids
            )
        ]


class Registry(BaseModel):
    """Registry containing dataset information."""

    datasets: list[DatasetInfo]

    @classmethod
    def from_json_list(cls, json_list: list[dict]) -> "Registry":
        """Create registry from JSON list."""
        return cls(datasets=[DatasetInfo.model_validate(row) for row in json_list])

    def get_dataset(self, name: str, version: str = "head") -> DatasetInfo | None:
        """Get specific dataset by name and version."""
        for dataset in self.datasets:
            if dataset.name == name and dataset.version == version:
                return dataset
        return None
