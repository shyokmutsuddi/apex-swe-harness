"""Constants for datasets module."""

from pathlib import Path

# Cache directory
CACHE_DIR = Path.home() / ".cache" / "apex-code"

# Default registry URL
DEFAULT_REGISTRY_URL = (
    "https://raw.githubusercontent.com/mercor-io/apex-v3/main/registry.json"
)
