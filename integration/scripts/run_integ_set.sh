#!/bin/bash
#
# Wrapper script for running task sets with apex-runner
#
# This script provides backward compatibility with the old run_integ_set.sh interface
# while delegating to the new unified apex-runner CLI.
#
# Usage:
#   ./run_integ_set.sh --model <model> <task-set-file> [additional-options]
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    echo "Usage: $0 --model <model> <task-set-file> [additional-options]"
    echo ""
    echo "Models: claude, opus, xai, gemini, deepseek, codex, qwen, kimi"
    echo ""
    echo "Examples:"
    echo "  $0 --model gemini integ-set-1.txt"
    echo "  $0 --model claude integ-set-2.txt --parallel --max-workers 3"
    echo "  $0 --model opus integ-set-1.txt --dry-run"
    exit 1
}

# Parse --model argument
MODEL=""
TASK_FILE=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            if [ -z "$TASK_FILE" ]; then
                TASK_FILE="$1"
            else
                EXTRA_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

# Validate model
if [ -z "$MODEL" ]; then
    echo "Error: --model is required"
    echo ""
    usage
fi

# Validate task file
if [ -z "$TASK_FILE" ]; then
    echo "Error: Task file is required"
    echo ""
    usage
fi

if [ ! -f "$TASK_FILE" ]; then
    # Try relative to script directory
    if [ -f "$SCRIPT_DIR/$TASK_FILE" ]; then
        TASK_FILE="$SCRIPT_DIR/$TASK_FILE"
    else
        echo "Error: Task file not found: $TASK_FILE"
        exit 1
    fi
fi

# Read tasks from file, ignoring comments and empty lines
TASKS=()
while IFS= read -r line || [ -n "$line" ]; do
    # Skip empty lines and comments
    line=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    if [ -n "$line" ] && [[ ! "$line" =~ ^# ]]; then
        TASKS+=("$line")
    fi
done < "$TASK_FILE"

if [ ${#TASKS[@]} -eq 0 ]; then
    echo "Error: No tasks found in $TASK_FILE"
    exit 1
fi

echo "==================================================================="
echo "Model: $MODEL"
echo "Running ${#TASKS[@]} tasks from: $TASK_FILE"
echo "==================================================================="
echo ""
echo "Tasks to run:"
for task in "${TASKS[@]}"; do
    echo "  - $task"
done
echo ""

# Run the unified apex-runner
apex-runner --model "$MODEL" --tasks "${TASKS[@]}" "${EXTRA_ARGS[@]}"
