"""Apex-Code evaluation logging utilities."""

import json
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Global logger instance and thread safety
_apex_logger: Optional["ApexLogger"] = None
_print_lock = threading.Lock()  # Prevent concurrent stdout/stderr output


class TaskLogger:
    """Manages logging for a single task episode in Apex-Code evaluation style."""

    def __init__(self, run_id: str, task_id: str, episode_num: int, log_dir: Path):
        self.run_id = run_id
        self.task_id = task_id
        self.episode_num = episode_num
        self.log_dir = log_dir
        self.episode_dir = log_dir / "agent-logs" / f"episode-{episode_num}"
        self.command_history_path = log_dir / "commands.txt"
        self.start_time = time.time()

        # Create episode directory structure
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.panes_dir = log_dir / "panes"
        self.sessions_dir = log_dir / "sessions"
        self.panes_dir.mkdir(exist_ok=True)
        self.sessions_dir.mkdir(exist_ok=True)

        # File paths for Apex-Code evaluation logging
        self.prompt_path = self.episode_dir / "prompt.txt"
        self.response_path = self.episode_dir / "response.json"
        self.debug_path = self.episode_dir / "debug.json"

        # Initialize debug data
        self.debug_data = {
            "episode": episode_num,
            "task_id": task_id,
            "start_time": datetime.now().isoformat(),
            "commands": [],
            "tool_calls": [],
        }

        # Command list for commands.txt
        self.commands = []

    def _log(self, message: str):
        """Lightweight helper to log task-scoped messages."""
        apex_logger = get_logger()
        if apex_logger:
            apex_logger._log(f"[{self.task_id}] {message}")
        else:
            logging.getLogger(__name__).info(f"[{self.task_id}] {message}")

    def log_prompt(self, prompt: str):
        """Log the prompt sent to the agent."""
        with open(self.prompt_path, "w") as f:
            f.write(prompt)

    def log_response(self, response: dict[str, Any]):
        """Log the agent's response."""
        with open(self.response_path, "w") as f:
            json.dump(response, f, indent=4)

    def log_command(
        self, command: str, is_blocking: bool = False, timeout: float | None = None
    ):
        """Log a command execution in Apex-Code evaluation format."""
        # Add to commands list for commands.txt
        if "\n" in command:
            # Multi-line commands are logged differently
            self.commands.append(command)
        else:
            self.commands.append(f"{command}\\n")

        # Update commands.txt incrementally
        with open(self.command_history_path, "a") as f:
            f.write(f"{command}\\n\n")

        # Add to debug data
        self.debug_data["commands"].append(
            {
                "command": command,
                "is_blocking": is_blocking,
                "timeout": timeout,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def log_tool_execution(
        self, tool_name: str, tool_call: dict[str, Any], tool_result: dict[str, Any]
    ):
        """Log tool execution details."""
        self.debug_data["tool_calls"].append(
            {
                "tool": tool_name,
                "call": tool_call,
                "result": tool_result,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Log command-style for terminal tool
        if tool_name == "terminal" and "command" in tool_call:
            self.log_command(
                tool_call["command"],
                is_blocking=tool_call.get("is_blocking", True),
                timeout=tool_call.get("timeout"),
            )

    def log_agent_action(self, action_type: str, data: dict[str, Any]):
        """Log agent actions."""
        self.debug_data.setdefault("agent_actions", []).append(
            {
                "action": action_type,
                "data": data,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def log_evaluation_result(self, result: dict[str, Any]):
        """Log evaluation results."""
        self.debug_data["evaluation"] = result

    def log_pane_capture(
        self, tmux_session, filename: str, capture_entire: bool = True
    ) -> Path | None:
        """Capture terminal pane state and save to panes directory.

        Args:
            tmux_session: TmuxSession instance
            filename: Output filename (e.g., 'pre-agent.txt')
            capture_entire: Whether to capture full scrollback

        Returns:
            Path to saved file or None if failed
        """
        if hasattr(tmux_session, "save_pane_capture"):
            return tmux_session.save_pane_capture(
                self.panes_dir, filename, capture_entire
            )
        return None

    def log_session_files(
        self, tmux_session, session_type: str = "agent"
    ) -> dict[str, Path | None]:
        """Copy session logs from container to host with terminal-bench naming.

        Args:
            tmux_session: TmuxSession instance
            session_type: Session type for naming ("agent" or "test")

        Returns:
            Dict with paths to copied files
        """
        if hasattr(tmux_session, "copy_session_logs_to_host"):
            return tmux_session.copy_session_logs_to_host(
                self.sessions_dir, session_type
            )
        return {"log_file": None, "cast_file": None}

    def finalize(self):
        """Write debug data to file."""
        with open(self.debug_path, "w") as f:
            json.dump(self.debug_data, f, indent=2)

    def start_episode(self, episode_num: int):
        """Log episode start."""
        pass  # Already handled in __init__

    def end_episode(self, status: str):
        """Log episode end."""
        self.debug_data["end_time"] = datetime.now().isoformat()
        self.debug_data["status"] = status
        self.debug_data["duration"] = time.time() - self.start_time
        self.finalize()


class ApexLogger:
    """Centralized logger for Apex-Code evaluation runs."""

    def __init__(self, run_id: str, runs_dir: Path):
        self.run_id = run_id
        # Add process ID to avoid collisions in parallel execution
        base_timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        process_id = os.getpid()
        self.timestamp = f"{base_timestamp}-{process_id}"
        self.run_dir = runs_dir / self.timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.run_log_path = self.run_dir / "run.log"
        self.run_metadata_path = self.run_dir / "run_metadata.json"
        self.apex_lock_path = self.run_dir / "apex.lock"

        # Create lock file
        self.apex_lock_path.touch()

        # Task loggers
        self.task_loggers: dict[str, TaskLogger] = {}

        # Track run start time
        self.run_start_time = time.time()

        # Set up logging
        self._setup_logging()

        # Log initial message
        self._log("Starting harness run")
        self._log(f"Run ID: {self.run_id}")

    def _setup_logging(self):
        """Set up file logging in Apex-Code evaluation format."""
        # File handler for run.log
        self.file_handler = logging.FileHandler(self.run_log_path)
        self.file_handler.setLevel(logging.INFO)

        # Custom formatter that doesn't include timestamp/level
        class PlainFormatter(logging.Formatter):
            def format(self, record):
                return record.getMessage()

        self.file_handler.setFormatter(PlainFormatter())

        # Create logger
        self.logger = logging.getLogger(f"tb.{self.run_id}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        self.logger.addHandler(self.file_handler)
        self.logger.propagate = False

    def _log(self, message: str):
        """Log a message to run.log and console with thread safety."""
        self.logger.info(message)
        with _print_lock:
            logging.getLogger(__name__).info(message)

    def _log_atomic(self, multi_line_message: str):
        """Log a multi-line message atomically to prevent interleaving."""
        # Log each line to the file logger
        for line in multi_line_message.split("\n"):
            self.logger.info(line)

        with _print_lock:
            logging.getLogger(__name__).info(multi_line_message)

    def log_run_metadata(self, metadata: dict[str, Any]):
        """Log run metadata with helpful trial information instead of repetitive task tables."""
        # Extract useful information for this specific trial
        task_id = metadata.get("task_id", "unknown")
        agent = metadata.get("agent", "unknown")
        trial_info = f"Starting {agent} on {task_id}"

        # Add timeout information if available
        timeout = metadata.get("timeout", "unknown")
        if timeout != "unknown":
            trial_info += f" (timeout: {timeout}s)"

        self._log(trial_info)

        # Save metadata
        with open(self.run_metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def log_task_start(self, task_id: str):
        """Log task start."""
        self._log(f"Executing {task_id}...")

    def log_docker_compose(self, command: str, project_name: str):
        """Log docker compose commands."""
        self._log(f"Running docker compose command: {command}")

    def log_command_execution(
        self,
        command: str,
        is_blocking: bool = False,
        timeout: float | None = None,
        duration: float | None = None,
    ):
        """Log command execution in Apex-Code evaluation format."""
        if is_blocking and duration is not None:
            self._log(
                f"Sending keys: {repr(command)} min_timeout_sec: {timeout or 0.0} max_timeout_sec: {timeout or 180.0}"
            )
            self._log(f"Blocking command completed in {duration:.2f}s.")
        else:
            self._log(
                f"Sending keys: {repr(command)} min_timeout_sec: {timeout or 0.0} max_timeout_sec: {timeout or 180.0}"
            )

    def log_model_call(self, model: str):
        """Log model API calls."""
        self._log(f"Making call to {model}")

    def create_task_logger(self, task_id: str) -> TaskLogger:
        """Create a task-specific logger."""
        # Create task directory with episode suffix
        task_log_dir = self.run_dir / task_id / f"{task_id}.1-of-1.{self.timestamp}"
        task_log_dir.mkdir(parents=True, exist_ok=True)

        # Determine episode number
        agent_logs_dir = task_log_dir / "agent-logs"
        if agent_logs_dir.exists():
            existing_episodes = [
                d
                for d in agent_logs_dir.iterdir()
                if d.is_dir() and d.name.startswith("episode-")
            ]
            episode_num = len(existing_episodes)
        else:
            episode_num = 0

        self.log_task_start(task_id)

        task_logger = TaskLogger(self.run_id, task_id, episode_num, task_log_dir)
        self.task_loggers[task_id] = task_logger
        return task_logger

    def finalize_run(self, summary: dict[str, Any]):
        """Finalize the run."""
        # Log any unresolved tasks
        if "unresolved_tasks" in summary:
            for task in summary["unresolved_tasks"]:
                self._log(f"Unresolved task {task}")

        # Close file handler
        self.file_handler.close()


def init_apex_logger(run_id: str, runs_dir: Path) -> ApexLogger:
    """Initialize Apex-Code evaluation logger."""
    global _apex_logger
    _apex_logger = ApexLogger(run_id, runs_dir)
    return _apex_logger


def get_logger() -> ApexLogger | None:
    """Get the current Apex-Code evaluation logger."""
    return _apex_logger


def log_step(task_id: str, step_num: int, message: str, level: str = "INFO"):
    """Log a step in Apex-Code evaluation format."""
    logger = get_logger()
    if logger:
        # Format: Sending keys: ['command', 'Enter'] min_timeout_sec: X max_timeout_sec: Y
        if "Executing:" in message:
            # Extract command from message
            command = message.replace("Executing:", "").strip()
            logger.log_command_execution(command, is_blocking=True, timeout=180.0)
        else:
            logger._log(message)
