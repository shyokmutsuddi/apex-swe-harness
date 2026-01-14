"""Multi-step task runner that allows agents to take multiple actions."""

import concurrent.futures
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil

from ..llms import BaseLLM
from ..tools import ToolExecutor
from ..utils.prompt_utils import build_episode_prompt, build_initial_prompt
from .docker_manager import docker_environment
from .evaluator import TaskEvaluator
from .logging_utils import TaskLogger, get_logger
from .models import (
    ExecutionStatus,
    TaskContext,
    TaskExecution,
)
from .terminal_manager import TerminalSessionManager

# Conversation management is now integrated into BaseLLM
from .utils import (
    cleanup_environment,
    setup_task_environment,
)

# Set up logger
logger = logging.getLogger(__name__)


def _ensure_git_baseline(
    docker_manager, repo_path: str = "/app", log: TaskLogger | None = None
) -> bool:
    """Initialize git inside the container to enable diff capture."""
    try:
        check_git = docker_manager.exec_command("which git", timeout=5)
        if check_git["exit_code"] != 0:
            if log:
                log._log("git not available in container; cannot capture agent.patch")
            return False

        is_repo = (
            docker_manager.exec_command(
                f"cd {repo_path} && git rev-parse --is-inside-work-tree",
                timeout=5,
            )["exit_code"]
            == 0
        )

        if not is_repo:
            docker_manager.exec_command(f"cd {repo_path} && git init", timeout=10)
            docker_manager.exec_command(
                f"cd {repo_path} && git config user.email 'apex@eval.local'", timeout=5
            )
            docker_manager.exec_command(
                f"cd {repo_path} && git config user.name 'APEX Evaluator'", timeout=5
            )

        # Ensure we have a baseline commit and include any newly copied files
        head_exists = (
            docker_manager.exec_command(
                f"cd {repo_path} && git rev-parse --verify HEAD", timeout=5
            )["exit_code"]
            == 0
        )
        if not head_exists:
            docker_manager.exec_command(f"cd {repo_path} && git add -A", timeout=60)
            docker_manager.exec_command(
                f"cd {repo_path} && git commit -m 'Baseline state for APEX evaluation' --allow-empty",
                timeout=60,
            )
        else:
            # If the working tree is dirty (newly copied task files), commit them to baseline
            status = docker_manager.exec_command(
                f"cd {repo_path} && git status --porcelain", timeout=30
            )
            if status.get("stdout", "").strip():
                docker_manager.exec_command(f"cd {repo_path} && git add -A", timeout=60)
                docker_manager.exec_command(
                    f"cd {repo_path} && git commit -m 'Baseline state for APEX evaluation' --allow-empty",
                    timeout=60,
                )

        return True
    except Exception as e:
        if log:
            log._log(f"Failed to initialize git for agent patch capture: {e}")
        return False


def _capture_agent_patch(
    docker_manager, task_logger: TaskLogger | None, repo_path: str = "/app"
) -> bool:
    """Write git diff to agent.patch inside the task log directory."""
    try:
        docker_manager.exec_command(f"cd {repo_path} && git add -A", timeout=60)
        diff_result = docker_manager.exec_command(
            f"cd {repo_path} && git diff --cached HEAD", timeout=30
        )
        patch_text = diff_result.get("stdout", "")
        if not patch_text.strip():
            # Try non-cached diff as fallback
            diff_result = docker_manager.exec_command(
                f"cd {repo_path} && git diff HEAD", timeout=30
            )
            patch_text = diff_result.get("stdout", "")

        if patch_text and patch_text.strip() and task_logger:
            agent_patch_file = task_logger.log_dir / "agent.patch"
            agent_patch_file.write_text(patch_text)
            task_logger._log(
                f"Captured agent.patch ({len(patch_text.encode('utf-8'))} bytes)"
            )
            return True
    except Exception as e:
        if task_logger:
            task_logger._log(f"Failed to capture agent.patch: {e}")
    return False


class ContextLengthExceededError(Exception):
    """Raised when the conversation exceeds the model's context length."""

    pass


class ConversationTooLongError(Exception):
    """Raised when conversation history becomes too long to manage effectively."""

    pass


class MultiStepRunner:
    """Handles multi-step task execution with tool support."""

    def __init__(
        self,
        llm: BaseLLM,
        max_steps: int | None = None,
        monitor_memory: bool = True,
        log_level: str = "INFO",
        todo_tool_enabled: bool = False,
    ):
        """
        Initialize multi-step runner.

        Args:
            llm: LLM for task execution
            max_steps: Maximum number of steps allowed (None = unlimited)
            monitor_memory: Whether to monitor memory usage
            log_level: Logging level for execution logs
            todo_tool_enabled: Whether to enable the todo tool
        """
        self.llm = llm
        self.max_steps = max_steps
        self.monitor_memory = monitor_memory
        self.todo_tool_enabled = todo_tool_enabled

        # Initialize conversation manager
        # Conversation management is now integrated into the LLM
        self.log_level = log_level

    def calculate_max_memory(self, steps: list[dict[str, Any]]) -> int | None:
        """Calculate maximum memory usage across all steps."""
        try:
            return int(psutil.Process().memory_info().rss / 1024 / 1024)
        except:
            return None

    def check_completion_indication(self, content: str) -> bool:
        """Check if agent indicates task completion."""

        # Check for XML task completion tags
        xml_completion_tags = [
            "<task_complete>true</task_complete>",
            "task_complete>true<",
            "task_complete>true",
        ]

        content_lower = content.lower()

        # Check XML tag completion
        return any(tag in content_lower for tag in xml_completion_tags)

    def run_single_trial(
        self,
        task_context: TaskContext,
        trial_number: int,
        working_dir: Path | None = None,
    ) -> TaskExecution:
        """
        Run a single trial with multiple steps.

        Args:
            task_context: Task context with instructions and files
            trial_number: Trial number (1-based)
            working_dir: Optional working directory (creates temp if None)

        Returns:
            TaskExecution result
        """
        start_time = datetime.now()
        logs = []
        steps = []

        # Get timeout - use task-specific timeout if available, otherwise use default
        max_timeout = float(task_context.timeout)
        if (
            hasattr(task_context, "max_agent_timeout_sec")
            and task_context.max_agent_timeout_sec is not None
        ):
            max_timeout = float(task_context.max_agent_timeout_sec)

        # Set up environment
        working_dir, setup_metadata = setup_task_environment(
            task_context, working_dir, use_docker=True
        )

        # Store original task directory for evaluation
        task_dir = task_context.task_dir

        docker_ctx = docker_environment(
            task_context,
            working_dir,
            sessions_logs_path=working_dir / "sessions",
            agent_logs_path=working_dir / "agent-logs",
        )

        # Get logger instances first
        logger = get_logger()
        task_logger = None
        if logger:
            # This will either get existing or create new task logger
            if task_context.task_id not in logger.task_loggers:
                task_logger = logger.create_task_logger(task_context.task_id)
            else:
                task_logger = logger.task_loggers[task_context.task_id]

        try:
            # Start Docker container if needed
            docker_manager = None

            def start_docker():
                return docker_ctx.__enter__()

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(start_docker)

                try:
                    docker_manager = future.result(timeout=max_timeout)

                    if logger:
                        logger.log_docker_compose(
                            "docker compose up -d", docker_manager._container_name
                        )
                except concurrent.futures.TimeoutError:
                    future.cancel()
                    raise RuntimeError(
                        f"Docker container startup timed out after {max_timeout}s"
                    )
                except Exception as e:
                    raise RuntimeError(f"Docker startup failed: {e}")

            # Initialize git baseline so we can capture agent.patch later
            git_ready = _ensure_git_baseline(
                docker_manager, repo_path="/app", log=task_logger
            )

            # Wait for MCP configuration to be ready (only for tasks with MCP servers); doesn't apply to original tasks or Oracle runs
            # Check if this task has tool containers by examining the docker-compose.yaml file
            # Skip MCP wait for Oracle runs since Oracle solutions use direct API calls, not MCP
            is_oracle_run = getattr(self.llm, 'model_name', '') == 'oracle'
            has_tool_containers = docker_manager.has_tool_containers(task_context)
            if has_tool_containers and not is_oracle_run:
                # Compute required MCP requirements and write to /config/mcp-required.txt inside the client container
                try:
                    requirements = docker_manager.get_required_mcp_requirements()
                    if requirements:
                        print(
                            f"Required MCP requirements: {', '.join(requirements)}"
                        )  # logger statements don't show up in terminal as of now, adding this to show in terminal
                        logger._log(
                            f"Required MCP requirements: {', '.join(requirements)}"
                        )
                        # Build content as newline-separated tokens
                        req_text = "\n".join(requirements) + "\n"
                        # Atomically write within /config
                        docker_manager.exec_command(
                            f"mkdir -p /config && printf '%s' '{req_text}' > /config/mcp-required.txt.tmp && mv /config/mcp-required.txt.tmp /config/mcp-required.txt"
                        )
                except Exception as e:
                    if logger:
                        logger._log(f"Failed to write mcp-required.txt: {e}")

                mcp_ready = False
                for i in range(
                    120
                ):  # Wait up to 20 minutes for MCP config to be ready (Plane takes 15+ min)
                    try:
                        # Check if MCP config is ready by running the health check script
                        # Execute via sh to avoid requiring the executable bit on the script
                        result = docker_manager._container.exec_run(
                            cmd=["sh", "-lc", "sh /app/wait-for-mcp-config.sh"]
                        )
                        if result.exit_code == 0:
                            mcp_ready = True
                            logger._log("MCP config is ready and task can begin")
                            break
                        else:
                            print("MCP config is not ready, waiting for 10 seconds")
                            logger._log(
                                "MCP config is not ready, waiting for 10 seconds"
                            )
                    except Exception as e:
                        if logger:
                            logger._log(f"⚠️ MCP config check failed: {e}")
                    if logger and i % 6 == 0:  # Log every minute
                        logger._log(
                            f"⏳ Waiting for API keys in MCP config to be ready... ({i + 1}/120)"
                        )
                    time.sleep(10)
                if not mcp_ready:
                    if logger:
                        logger._log(
                            "MCP config not ready after 20 minutes, aborting task execution"
                        )
                    raise RuntimeError("MCP configuration not ready after 20 minutes")

            # Initialize tool executor with Docker support
            tool_executor = ToolExecutor(
                working_dir,
                docker_manager=docker_manager,
                todo_tool_enabled=self.todo_tool_enabled,
            )

            # Capture initial terminal state before agent starts
            terminal_manager = TerminalSessionManager(docker_manager.container)
            terminal_manager.capture_pane_safely(
                tool_executor, task_logger, "pre-agent.txt"
            )

            # Initialize hybrid context (stateless prompts + conversation history)
            todo_list_text = tool_executor.get_todo_list_text()
            initial_prompt = build_initial_prompt(
                task_context, working_dir, tool_executor, max_timeout
            )
            # Append initial todo list to the first prompt so the agent sees it from the start
            if todo_list_text:
                initial_prompt = (
                    initial_prompt + f"\n\nINITIAL TODO LIST:\n{todo_list_text}"
                )
            current_prompt = initial_prompt

            # Reset conversation history for this trial
            self.llm.conversation_history = []

            # Determine max steps: task context > runner config > unlimited
            effective_max_steps = task_context.max_steps or self.max_steps

            # Multi-step execution loop
            step_num = 0
            episode_task_logger = None
            while True:
                step_num += 1

                # Check timeout
                elapsed_time = (datetime.now() - start_time).total_seconds()
                if elapsed_time > max_timeout:
                    if logger:
                        logger._log(f"Reached timeout limit ({max_timeout}s)")
                    break

                # Check if we've reached max steps
                if effective_max_steps and step_num > effective_max_steps:
                    if logger:
                        logger._log(
                            f"Reached maximum steps limit ({effective_max_steps})"
                        )
                    break

                # Check if conversation is becoming unmanageably long
                status = self.llm.get_conversation_status()
                if status["current_tokens"] > self.llm.max_tokens * 0.9:
                    if logger:
                        logger._log(
                            f"Conversation approaching hard token limit ({status['current_tokens']} tokens), terminating"
                        )
                    raise ConversationTooLongError(
                        f"Conversation reached {status['current_tokens']} tokens, approaching model limit"
                    )

                # Create a new task logger for each episode/step
                if logger:
                    # Create new episode logger for each step
                    episode_num = step_num - 1  # Episode 0, 1, 2, etc.
                    # Construct the correct task log directory using logger's run_dir
                    task_log_dir = (
                        logger.run_dir
                        / task_context.task_id
                        / f"{task_context.task_id}.1-of-1.{logger.timestamp}"
                    )
                    episode_task_logger = TaskLogger(
                        logger.run_id, task_context.task_id, episode_num, task_log_dir
                    )
                    # But keep using the global task_logger for panes/sessions logging

                    # Log the prompt for this episode (full conversation history)
                    # Log the current episode prompt
                    episode_task_logger.log_prompt(current_prompt)

                # Calculate elapsed time since task start
                elapsed_time = (datetime.now() - start_time).total_seconds()
                elapsed_minutes = elapsed_time / 60

                # Check token limits and trim conversation if needed
                self.llm.manage_conversation_tokens(
                    logger, episode_num, elapsed_minutes
                )

                # Call LLM with conversation history (like the old _get_hybrid_agent_response)
                # The LLM.call() method now uses conversation_history internally
                response_content = self.llm.call(current_prompt)

                # Add to conversation history for next episode
                self.llm.add_to_conversation(current_prompt, response_content)

                # Estimate token usage for tracking
                total_tokens = self.llm.count_tokens(
                    [
                        {"role": "user", "content": current_prompt},
                        {"role": "assistant", "content": response_content},
                    ]
                )
                input_tokens = self.llm.count_tokens(
                    [{"role": "user", "content": current_prompt}]
                )
                output_tokens = self.llm.count_tokens(
                    [{"role": "assistant", "content": response_content}]
                )

                # Create response object for logging and tracking
                response = {
                    "content": response_content,
                    "metadata": {
                        "model": self.llm.model_name,
                        "success": True,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    },
                    "tokens_used": total_tokens,
                }

                # Log agent response and track tokens
                if episode_task_logger:
                    episode_task_logger.log_response(response)
                    episode_task_logger.log_agent_action(
                        "response",
                        {
                            "step": episode_num,  # Use episode_num to match directory naming (0-based)
                            "content_preview": response.get("content", "")[:200],
                            "tokens": response.get("tokens_used", 0),
                        },
                    )

                # Track token usage
                response_tokens = response.get("tokens_used", 0)
                if response_tokens > 0:
                    self.llm.log_token_usage(step_num - 1, response_tokens)

                    # Log token status
                    token_status = self.llm.get_conversation_status()
                    usage_percentage = (
                        token_status["current_tokens"] / token_status["limit"]
                    ) * 100
                    if logger and usage_percentage > 80:
                        logger._log(
                            f"High token usage warning: {usage_percentage:.1f}% of limit"
                        )

                # Parse and execute any tool calls
                tool_results = tool_executor.parse_and_execute_tools(
                    response.get("content", ""), logs
                )

                # Log tool executions
                if episode_task_logger and tool_results:
                    for tool_result in tool_results:
                        episode_task_logger.log_tool_execution(
                            tool_result["tool"],
                            tool_result["call"],
                            tool_result["result"],
                        )

                # Record step
                step_data = {
                    "step_number": episode_num,  # Use 0-based numbering to match episode directories
                    "agent_response": response,
                    "tool_calls": tool_results,
                    "timestamp": datetime.now().isoformat(),
                }
                steps.append(step_data)

                # Finalize the task logger for this episode
                if episode_task_logger:
                    episode_task_logger.finalize()

                # Build prompt for next episode
                # Episode 0: Full prompt with template
                # Episodes 1+: Only terminal output
                terminal_content = terminal_manager.capture_current_state(tool_executor)
                todo_list_text = tool_executor.get_todo_list_text()
                current_prompt = build_episode_prompt(
                    step_num, initial_prompt, terminal_content, todo_list_text
                )

                if self.check_completion_indication(response.get("content", "")):
                    if task_logger:
                        task_logger.log_agent_action(
                            "completion",
                            {
                                "step": step_num
                                - 1,  # Use 0-based numbering to match episode directories
                                "reason": "explicit_completion",
                            },
                        )
                    break

            print("Agent completed.")

            # Capture terminal state after agent completes
            terminal_manager.capture_pane_safely(
                tool_executor, task_logger, "post-agent.txt"
            )

            # Capture agent patch before running tests
            if git_ready and task_logger:
                _capture_agent_patch(docker_manager, task_logger, repo_path="/app")

            # Calculate execution metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            max_memory = (
                self.calculate_max_memory(steps) if self.monitor_memory else None
            )

            # Format the final response properly
            final_response = {
                "content": steps[-1]["agent_response"].get("content", "")
                if steps
                else "",
                "steps": steps,
                "total_steps": len(steps),
                "conversation_history": self.llm.conversation_history,
            }

            # Create initial execution result
            execution = TaskExecution(
                trial_number=trial_number,
                status=ExecutionStatus.COMPLETED,
                agent_response=final_response,
                execution_time=execution_time,
                memory_used=max_memory,
                logs=logs,
                metadata={
                    "working_dir": str(working_dir),
                    "setup_metadata": setup_metadata.model_dump(),
                    "multi_step": True,
                    "max_steps": self.max_steps,
                    "steps_taken": len(steps),
                },
                started_at=start_time,
                completed_at=datetime.now(),
            )

            # Run evaluation
            if logger:
                logger._log("Running task evaluation")
            evaluator = TaskEvaluator()
            evaluation_result = evaluator.evaluate_execution(
                execution,
                task_dir,
                working_dir,
                max_test_timeout=task_context.max_test_timeout_sec,
                docker_manager=docker_manager,
            )

            # Persist test_results.json for integration parity
            if task_logger:
                test_results_path = task_logger.log_dir / "test_results.json"
                try:
                    test_results_payload = {
                        "task_id": task_context.task_id,
                        "trial_number": trial_number,
                        "passed": evaluation_result.get("passed", False),
                        "status": (
                            "passed"
                            if evaluation_result.get("passed", False)
                            else "failed"
                        ),
                        "test_output": evaluation_result.get("test_output"),
                        "evaluation": evaluation_result,
                        "timestamp": datetime.now().isoformat(),
                    }
                    test_results_path.write_text(
                        json.dumps(test_results_payload, indent=2, default=str)
                    )
                    task_logger._log(f"Saved test_results.json to {test_results_path}")
                except Exception as e:
                    task_logger._log(f"Failed to write test_results.json: {e}")

            # Log evaluation result
            if task_logger:
                task_logger.log_evaluation_result(evaluation_result)

            # Capture final terminal state and copy session logs
            terminal_manager.capture_pane_safely(
                tool_executor, task_logger, "post-test.txt"
            )
            terminal_manager.copy_session_logs_safely(
                tool_executor, task_logger, "agent"
            )

            # Update execution with evaluation results
            execution.metadata["evaluation"] = evaluation_result
            execution.metadata["test_passed"] = evaluation_result.get("passed", False)

            # Update status based on test results
            if not evaluation_result.get("passed", False):
                execution.status = ExecutionStatus.FAILED
                execution.error_message = (
                    f"Tests failed: {evaluation_result.get('test_output', 'No output')}"
                )

            # Log evaluation results
            test_passed = evaluation_result.get("passed", False)
            status_text = "PASS" if test_passed else "FAIL"
            evaluation_message = (
                f"[{status_text}] Evaluation complete. Tests passed: {test_passed}"
            )

            print(evaluation_message)
            if logger:
                logger._log(evaluation_message)

            # Finalize task logger
            if task_logger:
                task_logger.end_episode(execution.status.value)

            return execution

        except ContextLengthExceededError as e:
            # Handle context length exceeded errors specifically
            return TaskExecution(
                trial_number=trial_number,
                status=ExecutionStatus.FAILED,
                error_message=f"Context length exceeded: {str(e)}",
                execution_time=(datetime.now() - start_time).total_seconds(),
                logs=logs,
                metadata={
                    "working_dir": str(working_dir),
                    "error_type": "ContextLengthExceededError",
                    "failure_mode": "context_length_exceeded",
                    "steps_completed": len(steps) if "steps" in locals() else 0,
                    "token_usage": self.llm.get_conversation_status(),
                },
                started_at=start_time,
                completed_at=datetime.now(),
            )

        except ConversationTooLongError as e:
            # Handle conversation too long errors
            return TaskExecution(
                trial_number=trial_number,
                status=ExecutionStatus.FAILED,
                error_message=f"Conversation too long: {str(e)}",
                execution_time=(datetime.now() - start_time).total_seconds(),
                logs=logs,
                metadata={
                    "working_dir": str(working_dir),
                    "error_type": "ConversationTooLongError",
                    "failure_mode": "conversation_too_long",
                    "steps_completed": len(steps) if "steps" in locals() else 0,
                    "token_usage": self.llm.get_conversation_status(),
                },
                started_at=start_time,
                completed_at=datetime.now(),
            )

        except Exception as e:
            return TaskExecution(
                trial_number=trial_number,
                status=ExecutionStatus.FAILED,
                error_message=str(e),
                execution_time=(datetime.now() - start_time).total_seconds(),
                logs=logs,
                metadata={
                    "working_dir": str(working_dir),
                    "error_type": type(e).__name__,
                    "failure_mode": "unknown_error",
                    "steps_completed": len(steps) if "steps" in locals() else 0,
                    "token_usage": self.llm.get_conversation_status(),
                },
                started_at=start_time,
                completed_at=datetime.now(),
            )
        finally:
            # Clean up tool executor (e.g., tmux sessions)
            if "tool_executor" in locals() and tool_executor:
                try:
                    tool_executor.cleanup()
                except Exception as e:
                    if "logs" in locals() and logs:
                        logs.append(f"Error cleaning up tools: {e}")

            # Clean up Docker if used
            if "docker_ctx" in locals() and docker_ctx:
                try:
                    # Small delay to allow any pending log operations to complete
                    time.sleep(0.5)
                    # Small delay to allow any pending log operations to complete
                    time.sleep(0.5)
                    docker_ctx.__exit__(None, None, None)
                    if "logger" in locals() and logger:
                        container_name = "unknown"
                        if (
                            "docker_manager" in locals()
                            and docker_manager
                            and hasattr(docker_manager, "_container_name")
                        ):
                            container_name = docker_manager._container_name
                        logger.log_docker_compose("docker compose down", container_name)
                except Exception as e:
                    if "logs" in locals() and logs:
                        logs.append(f"Error cleaning up Docker: {e}")

            # Clean up environment (this handles file cleanup)
            cleanup_environment(working_dir, setup_metadata, preserve_logs=True)
