"""Common utilities for CLI operations."""

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from typer import Option

# Default directories
DEFAULT_DIRS = {
    "tasks": Path("tasks"),
    "runs": Path("runs"),
    "datasets": Path("datasets"),
}


def create_dir_option(
    dir_name: str,
    short_help: str,
    long_help: str,
    show_default: bool = True,
    short_flag: str | None = None,
) -> Annotated[Path, Option]:
    """Create a standardized directory option for CLI commands."""
    default_path = DEFAULT_DIRS.get(dir_name, Path(dir_name))

    if show_default:
        help_text = f"{long_help} (default: {default_path}/)"
    else:
        help_text = long_help

    # Build option arguments - only include short_flag if provided
    option_args = [f"--{dir_name}-dir"]
    if short_flag:
        option_args.append(short_flag)

    return Annotated[
        Path,
        Option(
            *option_args,
            help=help_text,
        ),
    ]


def load_json_file(file_path: Path) -> dict[str, Any] | None:
    """Safely load JSON from a file path."""
    if not file_path.exists():
        return None

    try:
        with open(file_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def safe_json_load(file_path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Load JSON with error reporting."""
    if not file_path.exists():
        return None, f"File not found: {file_path}"

    try:
        with open(file_path) as f:
            data = json.load(f)
        return data, None
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"
    except OSError as e:
        return None, f"File error: {e}"


def format_timestamp(timestamp: str, format_str: str = "%Y-%m-%d %H:%M") -> str:
    """Format ISO timestamp for display."""
    if not timestamp:
        return "unknown"

    try:
        dt = datetime.fromisoformat(timestamp)
        return dt.strftime(format_str)
    except ValueError:
        return timestamp


def get_directories_with_file(
    parent_dir: Path, required_file: str, include_hidden: bool = False
) -> list[Path]:
    """Get all directories containing a specific file."""
    if not parent_dir.exists():
        return []

    directories = []
    for item in parent_dir.iterdir():
        if (
            item.is_dir()
            and (include_hidden or not item.name.startswith("."))
            and (item / required_file).exists()
        ):
            directories.append(item)

    return directories
