"""Apex-Code evaluation system - simple pass/fail based on test scripts."""

import base64
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from .logging_utils import get_logger
from .models import ExecutionStatus, TaskExecution


class TaskEvaluator:
    """Apex-Code task evaluation using test scripts."""

    def __init__(self):
        """Initialize evaluator."""
        pass

    def evaluate_execution(
        self,
        execution: TaskExecution,
        task_dir: Path,
        working_dir: Path,
        max_test_timeout: float | None = None,
        docker_manager=None,
    ) -> dict[str, Any]:
        """
        Evaluate task execution using Apex-Code test scripts.

        Args:
            execution: The task execution to evaluate
            task_dir: Directory containing the task files
            working_dir: Working directory where the task was executed

        Returns:
            Evaluation results with pass/fail status
        """
        evaluation = {
            "execution_id": execution.trial_number,
            "timestamp": datetime.now().isoformat(),
            "passed": False,
            "test_output": "",
            "test_exit_code": -1,
            "execution_time": execution.execution_time,
            "memory_used": execution.memory_used,
        }

        # Check if execution completed successfully
        if execution.status != ExecutionStatus.COMPLETED:
            evaluation["test_output"] = (
                f"Execution failed with status: {execution.status}"
            )
            if execution.error_message:
                evaluation["test_output"] += f"\nError: {execution.error_message}"
            return evaluation

        # Look for test script in task directory
        test_scripts = [
            "run-tests.sh",
            "test.sh",
            "tests.sh",
            "run_tests.sh",
            "verify.sh",
        ]

        test_script = None
        # ONLY check in task directory for official test scripts
        # Do NOT use agent-created tests in working directory
        for script in test_scripts:
            script_path = task_dir / script
            if script_path.exists():
                test_script = script_path
                break

        if not test_script:
            evaluation["test_output"] = (
                f"EVALUATION FAILED: No test script found in working directory ({working_dir}) or task directory ({task_dir}). Expected one of: {', '.join(test_scripts)}"
            )
            evaluation["passed"] = False
            return evaluation

        # Run the test script
        logger = get_logger()
        if logger:
            logger._log(f"EVALUATION: Using test script: {test_script}")
            logger._log(
                "EVALUATION: Looking for test parser output ('PASSED' or 'FAILED')"
            )

        try:
            test_timeout = max_test_timeout if max_test_timeout else 60

            if docker_manager:
                # First, copy test files into the container
                test_files_to_copy = []

                # Copy the test script
                test_files_to_copy.append(test_script)

                # Copy the tests directory if it exists
                tests_dir = task_dir / "tests"
                if tests_dir.exists():
                    test_files_to_copy.append(tests_dir)

                # Copy files to container at /tests location
                # First, ensure /tests directory exists
                docker_manager.exec_command("mkdir -p /tests")

                for file_or_dir in test_files_to_copy:
                    if file_or_dir.is_file():
                        # For files, read and write to container
                        content = file_or_dir.read_text()
                        container_path = f"/tests/{file_or_dir.name}"
                        # Write file content using base64 encoding to avoid heredoc nesting issues

                        encoded_content = base64.b64encode(
                            content.encode("utf-8")
                        ).decode("ascii")
                        docker_manager.exec_command(
                            f"echo '{encoded_content}' | base64 -d > {container_path}"
                        )
                        # Make executable if it's a shell script
                        if file_or_dir.suffix == ".sh":
                            docker_manager.exec_command(f"chmod +x {container_path}")
                    elif file_or_dir.is_dir():
                        # For directories, copy all files maintaining structure
                        for test_file in file_or_dir.rglob("*"):
                            if test_file.is_file():
                                # Get relative path from the task directory
                                relative_path = test_file.relative_to(task_dir)
                                container_path = f"/{relative_path}"
                                # Create parent directories
                                docker_manager.exec_command(
                                    f"mkdir -p $(dirname {container_path})"
                                )
                                # Write file content
                                try:
                                    content = test_file.read_text()
                                    # Use base64 encoding to avoid heredoc nesting issues

                                    encoded_content = base64.b64encode(
                                        content.encode("utf-8")
                                    ).decode("ascii")
                                    docker_manager.exec_command(
                                        f"echo '{encoded_content}' | base64 -d > {container_path}"
                                    )
                                    if test_file.suffix == ".sh":
                                        docker_manager.exec_command(
                                            f"chmod +x {container_path}"
                                        )
                                except Exception:
                                    # Handle binary files or read errors
                                    pass

                # Run the test script inside the container from /tests to avoid clashing with /app pyproject
                test_command = (
                    f"cd /tests && chmod +x /tests/{test_script.name} "
                    f"&& TEST_DIR=/tests bash /tests/{test_script.name}"
                )
                result = docker_manager.exec_command(test_command, timeout=test_timeout)

                stdout = result.get("stdout", "")
                stderr = result.get("stderr", "")
                exit_code = result.get("exit_code", -1)

                # Parse test output for real pass/fail determination
                test_passed = self._parse_test_output(stdout)

                evaluation[
                    "test_output"
                ] = f"""OFFICIAL TEST EXECUTION: {test_script.name}
Command: {test_command}
Timeout: {test_timeout}s

=== STDOUT ===
{stdout}

=== STDERR ===
{stderr}

=== RESULT ===
Exit Code: {exit_code}
Test Result: {"PASSED" if test_passed else "FAILED" if test_passed is False else "NO_PARSER_OUTPUT"}"""

                evaluation["test_exit_code"] = exit_code
                evaluation["passed"] = test_passed if test_passed is not None else False

                # Log detailed results
                if test_passed is True:
                    evaluation["test_output"] += (
                        "\n\nEVALUATION RESULT: PASSED - Test parser found 'PASSED'"
                    )
                    success_message = f"EVALUATION PASSED: {test_script.name} - Test parser returned PASSED"
                    if logger:
                        logger._log(success_message)
                elif test_passed is False:
                    evaluation["test_output"] += (
                        "\n\nEVALUATION RESULT: FAILED - Test parser found 'FAILED'"
                    )
                    failure_message = f"EVALUATION FAILED: {test_script.name} - Test parser returned FAILED"
                    if logger:
                        logger._log(failure_message)
                else:
                    evaluation["test_output"] += (
                        f"\n\nEVALUATION RESULT: FAILED - No test parser output found (exit code: {exit_code})"
                    )
                    no_parser_message = f"EVALUATION FAILED: {test_script.name} - No test parser output detected"
                    if logger:
                        logger._log(no_parser_message)

            else:
                # Fallback: Run on host (original behavior)
                # Make script executable
                test_script.chmod(0o755)

                # Set up environment variables for test execution
                env = os.environ.copy()
                env["TEST_DIR"] = str(task_dir / "tests")
                env["HOME"] = os.path.expanduser("~")

                # Execute test script in working directory
                result = subprocess.run(
                    [str(test_script)],
                    cwd=working_dir,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=test_timeout,
                )

                # Parse test output for real pass/fail determination
                test_passed = self._parse_test_output(result.stdout)

                evaluation[
                    "test_output"
                ] = f"""OFFICIAL TEST EXECUTION: {test_script.name}
Command: {str(test_script)}
Timeout: {test_timeout}s

=== STDOUT ===
{result.stdout}

=== STDERR ===
{result.stderr}

=== RESULT ===
Exit Code: {result.returncode}
Test Result: {"PASSED" if test_passed else "FAILED" if test_passed is False else "NO_PARSER_OUTPUT"}"""

                evaluation["test_exit_code"] = result.returncode
                evaluation["passed"] = test_passed if test_passed is not None else False

                # Log detailed results with test parser output
                if test_passed is True:
                    evaluation["test_output"] += (
                        "\n\nEVALUATION RESULT: PASSED - Test parser found 'PASSED'"
                    )
                    success_message = f"EVALUATION PASSED: {test_script.name} - Test parser returned PASSED"
                    if logger:
                        logger._log(success_message)
                elif test_passed is False:
                    evaluation["test_output"] += (
                        "\n\nEVALUATION RESULT: FAILED - Test parser found 'FAILED'"
                    )
                    failure_message = f"EVALUATION FAILED: {test_script.name} - Test parser returned FAILED"
                    if logger:
                        logger._log(failure_message)
                else:
                    evaluation["test_output"] += (
                        f"\n\nEVALUATION RESULT: FAILED - No test parser output found (exit code: {result.returncode})"
                    )
                    no_parser_message = f"EVALUATION FAILED: {test_script.name} - No test parser output detected"
                    if logger:
                        logger._log(no_parser_message)

        except subprocess.TimeoutExpired:
            evaluation["test_output"] = (
                f"EVALUATION FAILED: Test script timed out after {test_timeout} seconds"
            )
            evaluation["test_exit_code"] = -1
            evaluation["passed"] = False

        except Exception as e:
            error_msg = f"EVALUATION FAILED: Error running test script {test_script.name}: {str(e)}"
            evaluation["test_output"] = error_msg
            evaluation["test_exit_code"] = -1
            evaluation["passed"] = False

        return evaluation

    def _parse_test_output(self, stdout: str) -> bool | None:
        """
        Parse test output to determine actual pass/fail status.

        Supports multiple test result formats:
        1. Structured format:
           - "results starts here" or "SWEBench results starts here"
           - "PASSED" or "FAILED"
           - "results ends here" or "SWEBench results ends here"
        2. Pytest native output:
           - "=== N passed in X.XXs ===" (all passed)
           - "=== N failed, M passed in X.XXs ===" (some failed)
           - "=== N failed in X.XXs ===" (all failed)
        3. Exit code based on script output

        Returns:
            True if PASSED found
            False if FAILED found
            None if no test parser output found
        """
        if not stdout:
            return None

        # Look for test parser output pattern (structured format)
        lines = stdout.split("\n")
        in_results_section = False

        for line in lines:
            line_stripped = line.strip()

            # Start of test results section
            if "results starts here" in line_stripped.lower():
                in_results_section = True
                continue

            # End of test results section
            if "results ends here" in line_stripped.lower():
                break

            # Parse result within the results section
            if in_results_section:
                if line_stripped == "PASSED":
                    return True
                elif line_stripped == "FAILED":
                    return False

        # Check for pytest output patterns
        import re

        # Look for pytest summary lines (appears at the end)
        # Examples:
        # "====== 6 passed in 2.34s ======"
        # "====== 5 passed, 2 warnings in 2.34s ======"
        # "====== 1 failed, 5 passed in 2.34s ======"
        # "====== 0 failed, 6 passed in 2.34s ======"  <- This should PASS (0 failures)
        # "====== 6 failed in 2.34s ======"

        for line in reversed(lines):
            line_stripped = line.strip()

            # Check for failure count - extract the number of failures
            failed_match = re.search(r"(\d+)\s+failed", line_stripped)
            if failed_match:
                num_failed = int(failed_match.group(1))
                # If there are actual failures (> 0), it's a failure
                if num_failed > 0:
                    return False
                # If 0 failed, continue checking for passed tests

            # Check for passed tests
            passed_match = re.search(r"(\d+)\s+passed", line_stripped)
            # Allow for optional human-readable time format like (MM:SS) between seconds and equals
            if passed_match and re.search(r"in\s+[\d.]+s", line_stripped):
                num_passed = int(passed_match.group(1))
                # If we have passed tests and haven't found failures > 0, it's a pass
                if num_passed > 0:
                    return True

            # Also check for pytest's FAILED marker
            if re.match(r"=+\s*FAILED\s*=+", line_stripped):
                return False

        # No test parser output found
        return None

    def evaluate_run(
        self, executions: list[TaskExecution], task_dir: Path, working_dirs: list[Path]
    ) -> dict[str, Any]:
        """
        Evaluate a complete run with multiple trials.

        Args:
            executions: List of task executions
            task_dir: Directory containing the task files
            working_dirs: List of working directories for each execution

        Returns:
            Aggregated evaluation results
        """
        evaluations = []

        # Ensure we have matching working directories
        if len(working_dirs) < len(executions):
            # Pad with None if needed
            working_dirs = working_dirs + [None] * (len(executions) - len(working_dirs))

        for i, execution in enumerate(executions):
            working_dir = (
                working_dirs[i] if working_dirs[i] else Path(f"/tmp/trial_{i + 1}")
            )
            eval_result = self.evaluate_execution(execution, task_dir, working_dir)
            evaluations.append(eval_result)

        # Calculate aggregate metrics
        passed_count = sum(1 for e in evaluations if e["passed"])
        total_count = len(evaluations)

        return {
            "evaluations": evaluations,
            "summary": {
                "total_trials": total_count,
                "passed_trials": passed_count,
                "failed_trials": total_count - passed_count,
                "success_rate": passed_count / total_count if total_count > 0 else 0.0,
                "average_execution_time": sum(e["execution_time"] for e in evaluations)
                / total_count
                if total_count > 0
                else 0.0,
                "average_memory_used": sum(
                    e["memory_used"] for e in evaluations if e["memory_used"]
                )
                / total_count
                if total_count > 0
                else 0.0,
            },
        }

    def format_evaluation_report(self, evaluation_results: dict[str, Any]) -> str:
        """
        Format evaluation results for display.

        Args:
            evaluation_results: Results from evaluate_run

        Returns:
            Formatted report string
        """
        summary = evaluation_results["summary"]
        evaluations = evaluation_results["evaluations"]

        report = []
        report.append("=" * 60)
        report.append("EVALUATION REPORT")
        report.append("=" * 60)
        report.append(f"Total Trials: {summary['total_trials']}")
        report.append(f"Passed: {summary['passed_trials']}")
        report.append(f"Failed: {summary['failed_trials']}")
        report.append(f"Success Rate: {summary['success_rate']:.1%}")
        report.append(
            f"Average Execution Time: {summary['average_execution_time']:.2f}s"
        )
        report.append(f"Average Memory Used: {summary['average_memory_used']:.1f}MB")
        report.append("")

        # Individual trial results
        report.append("TRIAL RESULTS:")
        report.append("-" * 60)

        for i, eval in enumerate(evaluations):
            status = "✅ PASSED" if eval["passed"] else "❌ FAILED"
            report.append(f"Trial {i + 1}: {status}")

            if eval["test_output"]:
                # Show first few lines of test output
                output_lines = eval["test_output"].strip().split("\n")[:5]
                for line in output_lines:
                    report.append(f"  {line}")
                if len(eval["test_output"].strip().split("\n")) > 5:
                    report.append("  ...")

            report.append("")

        return "\n".join(report)
