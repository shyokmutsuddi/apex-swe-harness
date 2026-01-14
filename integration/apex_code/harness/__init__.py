"""Evaluation harness for apex-code."""

from ..llms import (
    BaseLLM,
    ContextLengthExceededError,
    LiteLLM,
    MockLLM,
    OutputLengthExceededError,
    ParseError,
    create_llm,
)
from .docker_manager import (
    DockerComposeManager,
    DockerEnvironmentVars,
    DockerSetupMetadata,
    TaskSetupMetadata,
    check_docker,
    check_docker_availability,
    docker_environment,
    get_docker_info,
    spin_up_docker_environment,
)

# Conversation management is now integrated into BaseLLM
from .evaluator import TaskEvaluator
from .executor import EvaluationExecutor
from .models import (
    EvaluationConfig,
    ExecutionStatus,
    ModelResponse,
    ModelType,
    RunResult,
    TaskContext,
    TaskExecution,
)
from .multi_step_runner import MultiStepRunner
from .terminal_manager import (
    TerminalCommand,
    TerminalConstants,
    TerminalOutput,
    TerminalSessionManager,
    TerminalSessionMetadata,
    TmuxSession,
    check_tmux_availability,
    install_tmux_in_container,
    terminal_environment,
)
from .utils import (
    calculate_metrics,
    cleanup_environment,
    format_results,
    setup_task_environment,
    validate_agent_response,
)

__all__ = [
    "EvaluationConfig",
    "TaskExecution",
    "RunResult",
    "ModelResponse",
    "TaskContext",
    "ExecutionStatus",
    "ModelType",
    "MultiStepRunner",
    "EvaluationExecutor",
    "setup_task_environment",
    "validate_agent_response",
    "calculate_metrics",
    "cleanup_environment",
    "format_results",
    "DockerComposeManager",
    "DockerEnvironmentVars",
    "DockerSetupMetadata",
    "TaskSetupMetadata",
    "docker_environment",
    "check_docker",
    "get_docker_info",
    "spin_up_docker_environment",
    "check_docker_availability",
    "TerminalConstants",
    "TerminalCommand",
    "TerminalOutput",
    "TerminalSessionMetadata",
    "TmuxSession",
    "TerminalSessionManager",
    "check_tmux_availability",
    "install_tmux_in_container",
    "terminal_environment",
    "BaseLLM",
    "LiteLLM",
    "MockLLM",
    "create_llm",
    "ContextLengthExceededError",
    "OutputLengthExceededError",
    "ParseError",
    "TaskEvaluator",
]
