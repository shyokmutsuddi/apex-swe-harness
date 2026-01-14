"""
Docker management utilities.

Provides cleanup and maintenance functions for Docker resources.
"""

import subprocess
import sys


def cleanup_docker_networks() -> None:
    """
    Clean up unused Docker networks to prevent subnet exhaustion.

    Removes all unused networks without touching networks attached to running containers.
    Failures are logged but don't interrupt execution.
    """
    try:
        subprocess.run(
            ["docker", "network", "prune", "-f"],
            capture_output=True,
            timeout=30,
            check=False,
        )
    except Exception as e:
        print(f"Warning: Docker network cleanup failed: {e}", file=sys.stderr)


def cleanup_docker_containers() -> None:
    """
    Clean up stopped Docker containers.

    Removes all stopped containers.
    Failures are logged but don't interrupt execution.
    """
    try:
        subprocess.run(
            ["docker", "container", "prune", "-f"],
            capture_output=True,
            timeout=30,
            check=False,
        )
    except Exception as e:
        print(f"Warning: Docker container cleanup failed: {e}", file=sys.stderr)


def cleanup_docker() -> None:
    """
    Comprehensive Docker cleanup.

    Cleans up both networks and containers.
    """
    cleanup_docker_networks()
    cleanup_docker_containers()
