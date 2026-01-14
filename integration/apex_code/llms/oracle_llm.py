"""Oracle LLM implementation that executes pre-written solution files."""

import logging
import subprocess
from pathlib import Path

from .base_llm import BaseLLM

logger = logging.getLogger(__name__)


class OracleLLM(BaseLLM):
    """Oracle LLM that executes pre-written solution files instead of calling an LLM."""

    def __init__(
        self,
        task_dir: Path,
        container_name: str,
        **kwargs,
    ):
        super().__init__(model_name="oracle", **kwargs)
        self.task_dir = task_dir
        self.container_name = container_name
        self.solution_path = self._find_solution_file()

        if not self.solution_path:
            raise FileNotFoundError(
                f"No solution.sh or solution.py found in {self.task_dir}"
            )

        logger.info(f"Oracle LLM initialized with solution: {self.solution_path}")

    def _find_solution_file(self) -> Path | None:
        """Find solution file in task directory."""
        solution_sh = self.task_dir / "solution.sh"
        solution_py = self.task_dir / "solution.py"

        if solution_sh.exists():
            return solution_sh
        elif solution_py.exists():
            return solution_py

        return None

    def call(self, prompt: str, **kwargs) -> str:
        """Execute the solution file instead of calling an LLM."""
        logger.info("Oracle LLM executing solution file")

        try:
            if self.solution_path.suffix == ".sh":
                return self._execute_shell_solution()
            elif self.solution_path.suffix == ".py":
                return self._execute_python_solution()
            else:
                raise ValueError(
                    f"Unsupported solution file type: {self.solution_path}"
                )

        except subprocess.CalledProcessError as e:
            error_msg = f"Solution execution failed with exit code {e.returncode}"
            if e.stdout:
                error_msg += f"\nStdout: {e.stdout}"
            if e.stderr:
                error_msg += f"\nStderr: {e.stderr}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        except Exception as e:
            logger.error(f"Oracle execution failed: {e}")
            raise

    def _execute_shell_solution(self) -> str:
        """Execute shell script solution."""
        logger.info(f"Executing shell solution: {self.solution_path}")

        copy_cmd = [
            "docker",
            "cp",
            str(self.solution_path),
            f"{self.container_name}:/app/solution.sh",
        ]
        subprocess.run(copy_cmd, check=True, capture_output=True)

        result = subprocess.run(
            [
                "docker",
                "exec",
                self.container_name,
                "bash",
                "/app/solution.sh",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
        )

        logger.info("Shell solution executed successfully")
        return f"Oracle executed solution.sh:\n{result.stdout}"

    def _execute_python_solution(self) -> str:
        """Execute Python solution."""
        logger.info(f"Executing Python solution: {self.solution_path}")

        copy_cmd = [
            "docker",
            "cp",
            str(self.solution_path),
            f"{self.container_name}:/app/solution.py",
        ]
        subprocess.run(copy_cmd, check=True, capture_output=True)

        result = subprocess.run(
            [
                "docker",
                "exec",
                self.container_name,
                "python3",
                "/app/solution.py",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
        )

        logger.info("Python solution executed successfully")
        return f"Oracle executed solution.py:\n{result.stdout}"

    def count_tokens(self, messages: list[dict]) -> int:
        """Oracle uses 0 tokens."""
        return 0
