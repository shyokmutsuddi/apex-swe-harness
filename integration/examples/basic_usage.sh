#!/bin/bash
#
# Example: Basic usage of apex-runner
#

# Run all tasks with Claude sequentially
echo "Example 1: Sequential execution with Claude"
apex-runner --model claude

# Run specific tasks with Gemini in parallel
echo -e "\nExample 2: Parallel execution with Gemini"
apex-runner --model gemini \
  --tasks task1 task2 task3 \
  --parallel \
  --max-workers 5

# Dry run to preview commands
echo -e "\nExample 3: Dry run with XAI"
apex-runner --model xai --dry-run

# Custom output directory
echo -e "\nExample 4: Custom artifacts directory"
apex-runner --model opus \
  --artifacts-dir ./my-results \
  --parallel

# Specific tasks with custom CSV
echo -e "\nExample 5: Custom status CSV location"
apex-runner --model deepseek \
  --tasks important-task \
  --status-csv ./status/deepseek-status.csv

# Using environment variable for tasks directory
echo -e "\nExample 6: Custom tasks directory"
apex-runner --model qwen \
  --tasks-dir ./custom-tasks \
  --parallel \
  --max-workers 3
