#!/usr/bin/env python3
"""
Example: Using APEX SWE Harness programmatically from Python
"""

from pathlib import Path
from datetime import datetime

from apex_harness.models import get_model_config, list_available_models
from apex_harness.task_discovery import get_all_tasks
from apex_harness.runner import run_sequential, run_parallel
from apex_harness.status_tracker import CSVStatusTracker
from apex_harness.reporting import print_summary


def example_1_list_models():
    """Example 1: List all available models"""
    print("Available models:")
    for model in list_available_models():
        config = get_model_config(model)
        print(f"  - {model}: {config.name}")


def example_2_get_model_config():
    """Example 2: Get configuration for a specific model"""
    config = get_model_config("claude")
    print(f"\nClaude configuration:")
    print(f"  Name: {config.name}")
    print(f"  Identifier: {config.identifier}")
    print(f"  Report prefix: {config.report_prefix}")
    print(f"  Max workers: {config.max_workers}")
    print(f"  Timeout: {config.timeout}s")


def example_3_discover_tasks():
    """Example 3: Discover tasks in a directory"""
    tasks_dir = Path("tasks")
    if tasks_dir.exists():
        tasks = get_all_tasks(tasks_dir)
        print(f"\nFound {len(tasks)} tasks:")
        for task in tasks[:5]:  # Show first 5
            print(f"  - {task}")


def example_4_run_tasks_programmatically():
    """Example 4: Run tasks programmatically"""
    # Setup
    model_config = get_model_config("claude")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tasks = ["task1", "task2"]  # Your tasks
    
    # Create tracker
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    csv_path = artifacts_dir / f"claude_status_{timestamp}.csv"
    tracker = CSVStatusTracker(csv_path)
    
    # Run tasks sequentially
    results = run_sequential(
        tasks=tasks,
        timestamp=timestamp,
        model_config=model_config,
        tracker=tracker,
        dry_run=True  # Set to False for actual execution
    )
    
    # Print summary
    print_summary(results)


def example_5_run_parallel():
    """Example 5: Run tasks in parallel"""
    model_config = get_model_config("gemini")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tasks = ["task1", "task2", "task3"]
    
    # Run with custom worker count
    results = run_parallel(
        tasks=tasks,
        timestamp=timestamp,
        model_config=model_config,
        max_workers=3,
        tracker=None,  # Optional
        dry_run=True
    )
    
    print_summary(results)


def example_6_custom_model_config():
    """Example 6: Working with model configurations"""
    # Get all model configs
    from apex_harness.models import MODEL_REGISTRY
    
    print("\nAll model configurations:")
    for name, config in MODEL_REGISTRY.items():
        print(f"\n{name}:")
        print(f"  API Identifier: {config.identifier}")
        print(f"  Requires API key: {config.requires_env_var or 'No'}")
        print(f"  Max workers: {config.max_workers}")


if __name__ == "__main__":
    print("APEX SWE Harness Python API Examples")
    print("=" * 60)
    
    example_1_list_models()
    example_2_get_model_config()
    example_3_discover_tasks()
    
    print("\n" + "=" * 60)
    print("For actual task execution, use:")
    print("  - example_4_run_tasks_programmatically()")
    print("  - example_5_run_parallel()")
