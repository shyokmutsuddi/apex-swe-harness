# APEX SWE Harness

A comprehensive framework for evaluating AI models on software engineering tasks.

## 🎯 Overview

This repository provides a complete evaluation framework with two main components:

1. **APEX Code Harness** (`apex_code/`) - Core evaluation engine
   - Executes tasks in isolated Docker containers
   - Manages AI model interactions
   - Generates detailed evaluation reports
   - CLI: `apx` command

2. **Integration Test Runner** (`integration/`) - Multi-model orchestration
   - Unified interface for running multiple models
   - Parallel and sequential execution
   - Progress tracking and CSV reports
   - CLI: `apex-runner` command

3. **Tasks** (`tasks/`) - Software engineering task definitions
   - Real-world coding scenarios
   - Docker-based isolated environments
   - Automated evaluation rubrics

## 🚀 Quick Start

### One-Command Installation

```bash
# Clone repository
git clone https://github.com/your-org/apex-swe-harness.git
cd apex-swe-harness

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1

# Install both components
pip install -e apex_code
pip install -e integration

# Verify installation
apx --help
apex-runner --help
```

### Run Your First Evaluation

```bash
# Single task with APEX harness
apx reports run my-eval \
  --tasks 1-aws-s3-snapshots \
  --models claude-sonnet-4-5-20250929 \
  --n-trials 3

# Multiple tasks with integration runner
apex-runner --model claude \
  --tasks 1-aws-s3-snapshots 2-localstack-s3-snapshots \
  --parallel --max-workers 3
```

## 📦 Installation Options

### Option 1: Both Components (Recommended)

```bash
# Install APEX harness (required)
cd apex_code
pip install -e .

# Install integration runner
cd ../integration
pip install -e .
```

### Option 2: APEX Harness Only

```bash
cd apex_code
pip install -e .
apx --help
```

### Option 3: Integration Runner Only

```bash
cd integration
pip install -e .
apex-runner --help
```

**Note**: Integration runner requires APEX harness (`apx` command) to be installed.

## 🎮 Usage Examples

### Run Single Model Evaluation

```bash
apex-runner --model claude \
  --tasks 1-aws-s3-snapshots 2-localstack-s3-snapshots 3-localstack-s3-snapshots \
  --parallel --max-workers 3
```

### Run All 8 Models

```bash
for model in claude opus xai gemini deepseek qwen codex kimi; do
  apex-runner --model $model \
    --tasks 1-aws-s3-snapshots 2-localstack-s3-snapshots \
    --parallel --max-workers 3
done
```

### Dry Run (Preview Commands)

```bash
apex-runner --model claude --dry-run
```

## 🔑 Environment Setup

Some models require API keys:

```bash
# Anthropic models (Claude, Opus)
export ANTHROPIC_API_KEY='your-key'

# Google models (Gemini)
export GOOGLE_API_KEY='your-key'

# XAI models (Grok)
export XAI_API_KEY='your-key'

# Fireworks models (DeepSeek, Qwen)
export FIREWORKS_API_KEY='your-key'

# OpenAI models (Codex)
export OPENAI_API_KEY='your-key'
```

## 📊 Results and Artifacts

Results are automatically saved to `integration/artifacts/`:

```
integration/artifacts/
├── claude_tasks_status_20260114-120000.csv
├── gemini_tasks_status_20260114-120500.csv
└── ...
```

## 🏗️ Repository Structure

```
apex-swe-harness/
├── apex_code/              # Core evaluation harness
│   ├── cli/                # Command-line interface
│   ├── harness/            # Evaluation engine
│   ├── llms/               # Model adapters
│   ├── tools/              # Tool execution
│   └── utils/              # Utilities
├── integration/            # Multi-model test runner
│   ├── src/apex_harness/   # Runner implementation
│   ├── tests/              # Test suite (40 tests)
│   ├── docs/               # Documentation
│   └── examples/           # Usage examples
├── tasks/                  # Task definitions
│   ├── 1-aws-s3-snapshots/
│   ├── 2-localstack-s3-snapshots/
│   └── ...
└── temp/                   # Legacy code (to be removed)
```

## 📚 Documentation

- **APEX Harness**: [apex_code/README.md](apex_code/README.md)
- **Integration Runner**: [integration/README.md](integration/README.md)
- **Quick Start**: [integration/QUICKSTART.md](integration/QUICKSTART.md)
- **Architecture**: [integration/docs/ARCHITECTURE.md](integration/docs/ARCHITECTURE.md)
- **Testing Guide**: [integration/docs/TESTING.md](integration/docs/TESTING.md)
- **Migration Guide**: [integration/docs/MIGRATION.md](integration/docs/MIGRATION.md)
- **Installation Guide**: [INSTALL.md](INSTALL.md)
- **Contributing**: [integration/CONTRIBUTING.md](integration/CONTRIBUTING.md)

## 🧪 Supported Models

| Model | Provider | Identifier |
|-------|----------|------------|
| Claude Sonnet 4.5 | Anthropic | `claude` |
| Claude Opus 4 | Anthropic | `opus` |
| Gemini 3 Pro | Google | `gemini` |
| Grok 4 | xAI | `xai` |
| DeepSeek V3.2 | Fireworks | `deepseek` |
| Qwen 3 Coder | Fireworks | `qwen` |
| Codex | OpenAI | `codex` |
| Kimi | Moonshot | `kimi` |

## 🧑‍💻 Development

### Running Tests

```bash
# APEX harness tests
cd apex_code
pytest tests/ -v

# Integration runner tests (40 tests)
cd integration
pytest tests/ -v
python scripts/validate_refactoring.py
```

### Code Quality

```bash
# Format code
black apex_code/ integration/src/

# Lint
ruff check apex_code/ integration/src/

# Type check
mypy apex_code/ integration/src/
```

## 🤝 Contributing

We welcome contributions! See [integration/CONTRIBUTING.md](integration/CONTRIBUTING.md) for guidelines.

### Quick Contribution Steps

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run quality checks
5. Submit pull request

## 📝 License

MIT License - see [integration/LICENSE](integration/LICENSE) for details.

## 🆘 Troubleshooting

### `apx: command not found`

```bash
# Make sure APEX harness is installed
cd apex_code
pip install -e .
which apx
```

### `apex-runner: command not found`

```bash
# Make sure integration runner is installed
cd integration
pip install -e .
which apex-runner
```

### Docker Issues

```bash
# Verify Docker is running
docker ps

# Check permissions
docker run hello-world
```

### API Key Issues

```bash
# Verify keys are set
env | grep API_KEY

# Export required keys
export ANTHROPIC_API_KEY='your-key'
```

## 🔗 Related Projects

- APEX Code Framework
- Docker SDK for Python
- AI Model Evaluation Tools

## 📬 Support

- **Issues**: [GitHub Issues](https://github.com/your-org/apex-swe-harness/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/apex-swe-harness/discussions)
- **Documentation**: [docs/](integration/docs/)

---

Built with modern Python best practices for production-grade AI model evaluation.

## Documentation

- **Integration Harness**: [integration/README.md](integration/README.md)
- **Quick Start**: [integration/QUICKSTART.md](integration/QUICKSTART.md)
- **Architecture**: [integration/docs/ARCHITECTURE.md](integration/docs/ARCHITECTURE.md)
- **Contributing**: [integration/CONTRIBUTING.md](integration/CONTRIBUTING.md)

## Project Organization

```
apex-swe-harness/
├── integration/           # Integration testing harness
│   ├── src/              # Source code
│   ├── tests/            # Test suite
│   ├── docs/             # Documentation
│   ├── examples/         # Usage examples
│   └── scripts/          # Helper scripts
├── tasks/                # Task definitions
├── temp/                 # Legacy code (temporary)
└── README.md            # This file
```

## License

MIT License - see [integration/LICENSE](integration/LICENSE) for details.
