# APEX Code Harness

Core evaluation framework for running AI model assessments on software engineering tasks in isolated Docker environments.

## Overview

The APEX Code Harness is the engine that:
- Executes tasks in isolated Docker containers
- Manages AI model interactions and tool usage
- Evaluates task completion and correctness
- Generates detailed evaluation reports

## Installation

### From Source

```bash
# Clone repository
git clone https://github.com/your-org/apex-swe-harness.git
cd apex-swe-harness

# Install APEX harness
cd apex_code
pip install -e .

# Verify installation
apx --help
```

### With Development Dependencies

```bash
cd apex_code
pip install -e ".[dev]"
```

## Quick Start

### Run a Task Evaluation

```bash
apx reports run my-evaluation \
  --tasks 1-aws-s3-snapshots \
  --models claude-sonnet-4-5-20250929 \
  --n-trials 3 \
  --max-workers 3 \
  --timeout 3600
```

### List Available Tasks

```bash
apx tasks list
```

### Manage Evaluation Runs

```bash
# List all runs
apx runs list

# Show specific run details
apx runs show RUN_ID

# Delete a run
apx runs delete RUN_ID
```

### Manage Datasets

```bash
# List datasets
apx datasets list

# Show dataset details
apx datasets show DATASET_NAME
```

## Command Reference

### `apx reports`

Run evaluations and generate reports:

```bash
apx reports run REPORT_NAME \
  --tasks TASK1 TASK2 ... \
  --models MODEL_ID \
  --n-trials 3 \
  --max-workers 3 \
  --timeout 3600
```

**Options:**
- `--tasks`: Task names to evaluate (required)
- `--models`: Model identifier (required)
- `--n-trials`: Number of trials per task (default: 3)
- `--max-workers`: Parallel workers (default: 3)
- `--timeout`: Timeout in seconds per task (default: 3600)

### `apx tasks`

Manage tasks:

```bash
apx tasks list              # List all tasks
apx tasks show TASK_NAME    # Show task details
apx tasks validate TASK_NAME # Validate task configuration
```

### `apx runs`

Manage evaluation runs:

```bash
apx runs list               # List all runs
apx runs show RUN_ID        # Show run details
apx runs delete RUN_ID      # Delete a run
apx runs export RUN_ID      # Export run results
```

### `apx datasets`

Manage datasets:

```bash
apx datasets list           # List datasets
apx datasets show NAME      # Show dataset details
apx datasets create NAME    # Create new dataset
```

## Configuration

### Environment Variables

```bash
# Model API Keys
export ANTHROPIC_API_KEY='your-key'      # For Claude models
export GOOGLE_API_KEY='your-key'         # For Gemini models
export XAI_API_KEY='your-key'            # For XAI models
export FIREWORKS_API_KEY='your-key'      # For Fireworks models
export OPENAI_API_KEY='your-key'         # For OpenAI models

# Docker Configuration
export DOCKER_HOST='unix:///var/run/docker.sock'  # Default

# Harness Configuration
export APEX_CONFIG_PATH='/path/to/config.yaml'
```

### Configuration File

Create a `config.yaml`:

```yaml
models:
  claude-sonnet-4-5:
    provider: anthropic
    max_tokens: 4096
    temperature: 0.0

tasks:
  base_path: ./tasks
  timeout: 3600
  
docker:
  network: apex-network
  cleanup: true
```

## Task Structure

Tasks are defined in YAML with the following structure:

```
tasks/
└── my-task/
    ├── task.yaml           # Task configuration
    ├── task-spec.md        # Task specification
    ├── Dockerfile          # Task environment
    ├── docker-compose.yaml # Services configuration
    ├── solution.py         # Reference solution
    ├── rubric/
    │   └── rubric.json    # Evaluation rubric
    └── tests/
        └── test_*.py      # Test suite
```

## Architecture

The harness consists of several key components:

- **CLI** (`cli/`) - Command-line interface and commands
- **Harness** (`harness/`) - Core evaluation engine
  - Task execution
  - Docker management
  - Model interaction
  - Result evaluation
- **LLMs** (`llms/`) - Model adapters and interfaces
- **Tools** (`tools/`) - Tool execution and management
- **Utils** (`utils/`) - Utility functions

## Integration with Test Runner

For running evaluations across multiple models, use the integration test runner:

```bash
# Install integration runner
cd ../integration
pip install -e .

# Run with multiple models
apex-runner --model claude --parallel --max-workers 3
```

See [../integration/README.md](../integration/README.md) for details.

## Development

### Running Tests

```bash
cd apex_code
pytest tests/ -v
```

### Code Formatting

```bash
black apex_code/
ruff check apex_code/
```

### Type Checking

```bash
mypy apex_code/
```

## Troubleshooting

### Docker Issues

```bash
# Verify Docker is running
docker ps

# Check Docker permissions
docker run hello-world

# Clean up Docker resources
docker system prune -a
```

### Task Execution Issues

```bash
# Validate task configuration
apx tasks validate TASK_NAME

# Check task logs
docker logs CONTAINER_ID

# Run with verbose output
apx --verbose reports run ...
```

### Model API Issues

```bash
# Verify API keys are set
env | grep API_KEY

# Test model connection
apx models test MODEL_ID
```

## Contributing

See [../CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guidelines.

## License

MIT License - see [../LICENSE](../LICENSE) for details.

## Related Documentation

- **Integration Runner**: [../integration/README.md](../integration/README.md)
- **Task Definitions**: [../tasks/](../tasks/)
- **Architecture**: [../integration/docs/ARCHITECTURE.md](../integration/docs/ARCHITECTURE.md)
