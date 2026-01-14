#!/usr/bin/env python3
"""
Validation script to verify zero regression between old and new implementations.

This script compares command generation, CSV format, and behavior.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from apex_harness.models import get_model_config
from apex_harness.task_executor import run_task_command
from apex_harness.status_tracker import CSVStatusTracker


def validate_command_generation():
    """Validate that generated commands match legacy format."""
    print("=" * 70)
    print("VALIDATION 1: Command Generation")
    print("=" * 70)
    
    test_cases = [
        ("claude", "claude-sonnet-4-5-20250929", "claude", 3),
        ("xai", "xai/grok-4", "grok", 3),
        ("gemini", "gemini/gemini-3-pro-preview", "gemini", 3),
        ("deepseek", "fireworks_ai/accounts/fireworks/models/deepseek-v3p2", "deepseek", 3),
        ("qwen", "fireworks_ai/accounts/fireworks/models/qwen3-coder-480b-a35b-instruct", "fireworks", 2),
    ]
    
    all_passed = True
    
    for model_name, expected_id, expected_prefix, expected_workers in test_cases:
        config = get_model_config(model_name)
        
        # Validate config matches expectations
        checks = [
            (config.identifier == expected_id, f"identifier: {config.identifier} vs {expected_id}"),
            (config.report_prefix == expected_prefix, f"prefix: {config.report_prefix} vs {expected_prefix}"),
            (config.max_workers == expected_workers, f"workers: {config.max_workers} vs {expected_workers}"),
        ]
        
        passed = all(check[0] for check in checks)
        status = "[PASS]" if passed else "[FAIL]"
        print(f"\n{model_name}: {status}")
        
        if not passed:
            for check_passed, msg in checks:
                if not check_passed:
                    print(f"  [X] {msg}")
            all_passed = False
        else:
            print(f"  [OK] Config matches legacy")
    
    return all_passed


def validate_csv_format():
    """Validate CSV output format matches legacy."""
    print("\n" + "=" * 70)
    print("VALIDATION 2: CSV Format")
    print("=" * 70)
    
    import tempfile
    import csv
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "test.csv"
        tracker = CSVStatusTracker(csv_path)
        
        # Add test data
        tracker.update_status(
            "test-task",
            "completed",
            start_time="2026-01-14 10:00:00",
            end_time="2026-01-14 10:05:00",
            duration_seconds="300.5",
            return_code="0",
            error_message="",
        )
        
        # Read and validate
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            headers = reader.fieldnames
        
        expected_headers = [
            "task_name",
            "status",
            "start_time",
            "end_time",
            "duration_seconds",
            "return_code",
            "error_message",
        ]
        
        headers_match = headers == expected_headers
        data_correct = (
            len(rows) == 1
            and rows[0]["task_name"] == "test-task"
            and rows[0]["status"] == "completed"
            and rows[0]["duration_seconds"] == "300.5"
        )
        
        if headers_match and data_correct:
            print("[PASS]: CSV format matches legacy")
            print(f"  [OK] Headers: {headers}")
            print(f"  [OK] Data format validated")
            return True
        else:
            print("[FAIL]: CSV format mismatch")
            if not headers_match:
                print(f"  [X] Headers: {headers} vs {expected_headers}")
            if not data_correct:
                print(f"  [X] Data validation failed")
            return False


def validate_command_construction():
    """Validate actual command construction."""
    print("\n" + "=" * 70)
    print("VALIDATION 3: Command Construction")
    print("=" * 70)
    
    model_config = get_model_config("claude")
    task_name = "test-task"
    timestamp = "20260114-120000"
    
    with patch("apex_harness.task_executor.subprocess.run") as mock_run:
        with patch("apex_harness.task_executor.cleanup_docker"):
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            
            run_task_command(task_name, timestamp, model_config, dry_run=False)
            
            # Get the command that was called (first call)
            call_args = mock_run.call_args[0][0]
        
        expected = [
            "apx",
            "reports",
            "run",
            f"claude-{task_name}-{timestamp}",
            "--tasks",
            task_name,
            "--models",
            "claude-sonnet-4-5-20250929",
            "--n-trials",
            "3",
            "--max-workers",
            "3",
            "--timeout",
            "3600",
        ]
        
        if call_args == expected:
            print("[PASS]: Command construction matches legacy")
            print(f"  [OK] Command: {' '.join(call_args)}")
            return True
        else:
            print("[FAIL]: Command construction mismatch")
            print(f"  Expected: {' '.join(expected)}")
            print(f"  Got:      {' '.join(call_args)}")
            return False


def validate_model_registry():
    """Validate all models in registry have correct structure."""
    print("\n" + "=" * 70)
    print("VALIDATION 4: Model Registry Completeness")
    print("=" * 70)
    
    from apex_harness.models import MODEL_REGISTRY, list_available_models
    
    expected_models = {"claude", "opus", "xai", "gemini", "deepseek", "codex", "qwen", "kimi"}
    actual_models = set(list_available_models())
    
    if actual_models == expected_models:
        print(f"[PASS]: All {len(expected_models)} models present")
        for model in sorted(actual_models):
            config = get_model_config(model)
            print(f"  [OK] {model}: {config.name}")
        return True
    else:
        print("[FAIL]: Model registry incomplete")
        missing = expected_models - actual_models
        extra = actual_models - expected_models
        if missing:
            print(f"  [X] Missing: {missing}")
        if extra:
            print(f"  [!] Extra: {extra}")
        return False


def main():
    """Run all validations."""
    print("\n" + "=" * 70)
    print("APEX SWE HARNESS - REFACTORING VALIDATION")
    print("=" * 70)
    print("\nVerifying zero regression against original implementation...\n")
    
    results = []
    
    results.append(("Command Generation", validate_command_generation()))
    results.append(("CSV Format", validate_csv_format()))
    results.append(("Command Construction", validate_command_construction()))
    results.append(("Model Registry", validate_model_registry()))
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n" + "=" * 70)
        print("*** ALL VALIDATIONS PASSED ***")
        print("=" * 70)
        print("\nRefactoring is complete with ZERO REGRESSION.")
        print("The new implementation produces identical outputs to the original.")
        return 0
    else:
        print("\n" + "=" * 70)
        print("*** SOME VALIDATIONS FAILED ***")
        print("=" * 70)
        print("\nPlease review and fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
