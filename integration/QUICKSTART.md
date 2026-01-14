# Quick Start Guide

Get up and running with APEX SWE Harness in 5 minutes.

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/apex-swe-harness.git
cd apex-swe-harness

# Install the package
pip install -e .

# Verify installation
apex-runner --help
```

## Your First Run

```bash
# List available models
apex-runner --model claude --dry-run

# Run a single task
apex-runner --model claude --tasks your-task-name

# Run all tasks
apex-runner --model claude
```

## Common Use Cases

### Sequential Execution (Default)
```bash
apex-runner --model gemini
```

### Parallel Execution
```bash
apex-runner --model gemini --parallel --max-workers 5
```

### Specific Tasks
```bash
apex-runner --model xai --tasks task1 task2 task3
```

### Custom Output Directory
```bash
apex-runner --model opus --artifacts-dir ./my-results
```

### Dry Run (Preview Commands)
```bash
apex-runner --model deepseek --dry-run
```

## Environment Setup

Some models require API keys:

```bash
# For Fireworks-based models (DeepSeek, Qwen)
export FIREWORKS_API_KEY='your-api-key'
```

## Output Files

Results are stored in `artifacts/` (by default):

```
artifacts/
├── claude_tasks_status_20260114-120000.csv
├── gemini_tasks_status_20260114-120500.csv
└── ...
```

## Next Steps

1. **Read the docs**: Check [README.md](README.md) for detailed usage
2. **Run tests**: `pytest tests/` to verify your setup
3. **Explore examples**: See [examples/](examples/) for more usage patterns
4. **Contribute**: Read [CONTRIBUTING.md](CONTRIBUTING.md) to get involved

## Troubleshooting

### Command not found
```bash
# Make sure package is installed
pip install -e .

# Or use module syntax
python -m apex_harness.cli --model claude
```

### Tasks not found
```bash
# Specify tasks directory explicitly
apex-runner --model claude --tasks-dir ./tasks
```

### API key missing
```bash
# Set required environment variable
export FIREWORKS_API_KEY='your-key'
```

## Getting Help

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/your-org/apex-swe-harness/issues)
- **Examples**: [examples/](examples/)

Happy testing! 🚀
