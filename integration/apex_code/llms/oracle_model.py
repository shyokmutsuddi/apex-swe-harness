"""Oracle Model implementation that executes pre-written solution files."""

import logging
import re
import subprocess
from pathlib import Path

from .base_llm import BaseLLM

logger = logging.getLogger(__name__)


class OracleModel(BaseLLM):
    """Oracle Model that executes pre-written solution files instead of calling an LLM."""

    def __init__(
        self,
        task_dir: Path,
        container_name: str | None = None,
        **kwargs,
    ):
        super().__init__(model_name="oracle", **kwargs)
        self.task_dir = task_dir
        self._container_name = container_name
        self.solution_path = self._find_solution_file()

        if not self.solution_path:
            raise FileNotFoundError(
                f"No solution.sh or solution.py found in {self.task_dir}"
            )

        logger.info(f"Oracle Model initialized with solution: {self.solution_path}")

    def _get_container_name(self) -> str:
        """Get the container name, discovering it if not provided."""
        if self._container_name:
            return self._container_name

        import subprocess

        task_id_safe = self.task_dir.name.replace("-", "_")
        result = subprocess.run(
            [
                "docker",
                "ps",
                "--filter",
                f"name=apex_task_{task_id_safe}",
                "--format",
                "{{.Names}}",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        container_names = [name for name in result.stdout.strip().split("\n") if name]
        if not container_names:
            raise RuntimeError(
                f"No running container found for task {self.task_dir.name}"
            )

        # Prefer the dashless client container for THIS task id only.
        client_container = None
        prefix = f"apex_task_{task_id_safe}_"
        for name in container_names:
            if name and name.startswith(prefix) and "-" not in name:
                client_container = name
                break

        # If not found on first pass, refresh once (client should always be present).
        if not client_container:
            retry = subprocess.run(
                [
                    "docker",
                    "ps",
                    "--filter",
                    f"name=apex_task_{task_id_safe}",
                    "--format",
                    "{{.Names}}",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            retry_names = [n for n in retry.stdout.strip().split("\n") if n]
            for name in retry_names:
                if name and name.startswith(prefix) and "-" not in name:
                    client_container = name
                    break

        # Last resort: fall back to the first listed container (should be rare).
        if not client_container:
            client_container = container_names[0]
            logger.warning(
                "Could not find dashless client container; falling back to first listed "
                f"container: {client_container}"
            )

        self._container_name = client_container
        logger.info(f"Discovered client container name: {self._container_name}")
        return self._container_name

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
        logger.info("Oracle Model executing solution file")
        print("=" * 80, flush=True)
        print("[ORACLE] Starting oracle agent execution", flush=True)
        print(f"[ORACLE] Solution file: {self.solution_path}", flush=True)
        print(f"[ORACLE] Solution type: {self.solution_path.suffix}", flush=True)
        print("=" * 80, flush=True)

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
            print(f"[ORACLE ERROR] Exit code: {e.returncode}", flush=True)
            if e.stdout:
                error_msg += f"\nStdout: {e.stdout}"
                print(f"[ORACLE ERROR] Stdout: {e.stdout[:1000]}", flush=True)
            if e.stderr:
                error_msg += f"\nStderr: {e.stderr}"
                print(f"[ORACLE ERROR] Stderr: {e.stderr[:1000]}", flush=True)
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        except Exception as e:
            logger.error(f"Oracle execution failed: {e}")
            raise

    def _execute_shell_solution(self) -> str:
        """Execute shell script solution."""
        logger.info(f"Executing shell solution: {self.solution_path}")
        print(f"[ORACLE] Executing shell solution: {self.solution_path}", flush=True)

        container_name = self._get_container_name()
        print(f"[ORACLE] Container: {container_name}", flush=True)

        copy_cmd = [
            "docker",
            "cp",
            str(self.solution_path),
            f"{container_name}:/app/solution.sh",
        ]
        print(
            f"[ORACLE] Copying solution to container: {' '.join(copy_cmd)}", flush=True
        )
        subprocess.run(copy_cmd, check=True, capture_output=True)

        exec_cmd = [
            "docker",
            "exec",
            container_name,
            "bash",
            "/app/solution.sh",
        ]
        print(f"[ORACLE] Executing: {' '.join(exec_cmd)}", flush=True)

        result = subprocess.run(
            exec_cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
        )

        # Parse and display useful output info
        stdout_lines = result.stdout.strip().split("\n") if result.stdout else []
        stderr_lines = result.stderr.strip().split("\n") if result.stderr else []

        print(f"[ORACLE] ✅ Exit code: {result.returncode}", flush=True)
        print(
            f"[ORACLE] 📊 Stdout: {len(stdout_lines)} lines, {len(result.stdout)} chars",
            flush=True,
        )

        # Show first and last few lines of stdout
        if stdout_lines:
            print("[ORACLE] 📄 First lines of stdout:", flush=True)
            for line in stdout_lines[:5]:
                print(f"[ORACLE]   | {line}", flush=True)
            if len(stdout_lines) > 10:
                print(
                    f"[ORACLE]   | ... ({len(stdout_lines) - 10} lines omitted) ...",
                    flush=True,
                )
            if len(stdout_lines) > 5:
                print("[ORACLE] 📄 Last lines of stdout:", flush=True)
                for line in stdout_lines[-5:]:
                    print(f"[ORACLE]   | {line}", flush=True)

        if stderr_lines:
            print(f"[ORACLE] ⚠️  Stderr: {len(stderr_lines)} lines", flush=True)
            for line in stderr_lines[:10]:
                print(f"[ORACLE]   | {line}", flush=True)

        print("=" * 80, flush=True)

        logger.info("Shell solution executed successfully")
        return f"Oracle executed solution.sh:\n{result.stdout}\n\n<task_complete>true</task_complete>"

    def _execute_python_solution(self) -> str:
        """Execute Python solution."""
        logger.info(f"Executing Python solution: {self.solution_path}")
        print(f"[ORACLE] Executing Python solution: {self.solution_path}", flush=True)

        container_name = self._get_container_name()
        print(f"[ORACLE] Container: {container_name}", flush=True)

        copy_cmd = [
            "docker",
            "cp",
            str(self.solution_path),
            f"{container_name}:/app/solution.py",
        ]
        print(
            f"[ORACLE] Copying solution to container: {' '.join(copy_cmd)}", flush=True
        )
        subprocess.run(copy_cmd, check=True, capture_output=True)

        exec_cmd = [
            "docker",
            "exec",
            container_name,
            "python3",
            "/app/solution.py",
        ]
        print(f"[ORACLE] Executing: {' '.join(exec_cmd)}", flush=True)

        result = subprocess.run(
            exec_cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
        )

        # Parse and display useful output info
        stdout_lines = result.stdout.strip().split("\n") if result.stdout else []
        stderr_lines = result.stderr.strip().split("\n") if result.stderr else []

        print(f"[ORACLE] ✅ Exit code: {result.returncode}", flush=True)
        print(
            f"[ORACLE] 📊 Stdout: {len(stdout_lines)} lines, {len(result.stdout)} chars",
            flush=True,
        )

        # Show first and last few lines of stdout
        if stdout_lines:
            print("[ORACLE] 📄 First lines of stdout:", flush=True)
            for line in stdout_lines[:5]:
                print(f"[ORACLE]   | {line}", flush=True)
            if len(stdout_lines) > 10:
                print(
                    f"[ORACLE]   | ... ({len(stdout_lines) - 10} lines omitted) ...",
                    flush=True,
                )
            if len(stdout_lines) > 5:
                print("[ORACLE] 📄 Last lines of stdout:", flush=True)
                for line in stdout_lines[-5:]:
                    print(f"[ORACLE]   | {line}", flush=True)

        if stderr_lines:
            print(f"[ORACLE] ⚠️  Stderr: {len(stderr_lines)} lines", flush=True)
            for line in stderr_lines[:10]:
                print(f"[ORACLE]   | {line}", flush=True)

        print("=" * 80, flush=True)

        logger.info("Python solution executed successfully")
        return f"Oracle executed solution.py:\n{result.stdout}\n\n<task_complete>true</task_complete>"

    def count_tokens(self, messages: list[dict]) -> int:
        """Oracle uses 0 tokens."""
        return 0
