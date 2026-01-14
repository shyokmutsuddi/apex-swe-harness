# Repository Structure

This document describes the complete structure of the APEX SWE Harness repository.

## Overview

The repository is organized into three main components:

1. **APEX Code Harness** (`apex_code/`) - Core evaluation engine
2. **Integration Test Runner** (`integration/`) - Multi-model orchestration
3. **Tasks** (`tasks/`) - Software engineering task definitions

## Directory Tree

```
apex-swe-harness/
│
├── apex_code/                   # Core APEX harness (apx command)
│   ├── cli/                     # Command-line interface
│   │   ├── main.py              # Main CLI entry point
│   │   ├── datasets/            # Dataset management commands
│   │   ├── reports/             # Report generation commands
│   │   ├── runs/                # Run management commands
│   │   ├── tasks/               # Task management commands
│   │   └── utils/               # CLI utilities
│   ├── harness/                 # Core evaluation engine
│   │   ├── docker_manager.py    # Docker container management
│   │   ├── evaluator.py         # Task evaluation logic
│   │   ├── executor.py          # Task execution
│   │   ├── multi_step_runner.py # Multi-step task orchestration
│   │   └── terminal_manager.py  # Terminal interaction
│   ├── llms/                    # AI model adapters
│   │   ├── base_llm.py          # Base LLM interface
│   │   ├── llm.py               # Main LLM implementation
│   │   ├── mock_llm.py          # Mock for testing
│   │   └── oracle_llm.py        # Oracle model
│   ├── tools/                   # Tool execution framework
│   │   ├── file_tool.py         # File manipulation tools
│   │   ├── terminal_tool.py     # Terminal command tools
│   │   ├── todo_tool.py         # Todo tracking tools
│   │   └── tool_executor.py     # Tool execution engine
│   ├── utils/                   # Utilities
│   │   ├── logging_utils.py     # Logging configuration
│   │   └── prompt_utils.py      # Prompt templating
│   ├── config.py                # Configuration management
│   ├── pyproject.toml           # Package configuration
│   ├── setup.py                 # Setup script
│   └── README.md                # APEX harness documentation
│
├── integration/                 # Integration test runner (apex-runner command)
│   ├── src/apex_harness/        # Runner implementation
│   │   ├── __init__.py          # Package initialization
│   │   ├── cli.py               # Unified CLI entry point
│   │   ├── models.py            # Model registry and configs (Strategy pattern)
│   │   ├── status_tracker.py   # CSV status tracking
│   │   ├── task_discovery.py   # Task discovery logic
│   │   ├── task_executor.py    # Task execution (calls apx)
│   │   ├── runner.py            # Sequential/parallel orchestration
│   │   ├── reporting.py         # Result reporting
│   │   └── docker_utils.py      # Docker cleanup utilities
│   ├── tests/                   # Comprehensive test suite (40 tests)
│   │   ├── __init__.py
│   │   ├── test_models.py       # Model config tests
│   │   ├── test_status_tracker.py # Status tracking tests
│   │   ├── test_task_discovery.py # Task discovery tests
│   │   ├── test_cli.py          # CLI tests
│   │   ├── test_reporting.py    # Reporting tests
│   │   └── test_integration.py  # End-to-end integration tests
│   ├── scripts/                 # Utility scripts
│   │   ├── run_integ_set.sh     # Legacy wrapper (backward compatibility)
│   │   └── validate_refactoring.py # Regression validation script
│   ├── docs/                    # Documentation
│   │   ├── ARCHITECTURE.md      # Architecture decisions
│   │   ├── MIGRATION.md         # Migration guide
│   │   └── TESTING.md           # Testing guide
│   ├── examples/                # Usage examples
│   │   ├── basic_usage.sh       # Shell examples
│   │   └── python_usage.py      # Python API examples
│   ├── artifacts/               # Generated artifacts (gitignored)
│   │   └── .gitkeep
│   ├── pyproject.toml           # Package configuration
│   ├── pytest.ini               # Pytest configuration
│   ├── setup.py                 # Setup script
│   ├── CONTRIBUTING.md          # Contribution guidelines
│   ├── CODE_OF_CONDUCT.md       # Code of conduct
│   ├── SECURITY.md              # Security policy
│   ├── LICENSE                  # MIT License
│   ├── QUICKSTART.md            # Quick start guide
│   ├── REFACTORING_SUMMARY.md   # Refactoring summary
│   └── README.md                # Integration runner documentation
│
├── tasks/                       # Task definitions
│   ├── 1-aws-s3-snapshots/      # Example task
│   │   ├── task.yaml            # Task configuration
│   │   ├── task-spec.md         # Task specification
│   │   ├── Dockerfile           # Task environment
│   │   ├── docker-compose.yaml  # Services configuration
│   │   ├── solution.py          # Reference solution
│   │   ├── rubric/
│   │   │   └── rubric.json      # Evaluation rubric
│   │   └── tests/
│   │       └── test_*.py        # Test suite
│   ├── 2-localstack-s3-snapshots/
│   ├── 3-localstack-s3-snapshots/
│   ├── 4-localstack-notifications/
│   └── 5-localstack-s3-notifications/
│
├── temp/                        # Legacy code (to be removed after verification)
│   ├── apex_code/               # Original APEX harness
│   ├── tasks/                   # Original tasks
│   └── *_run_all_tasks.py       # Legacy task runner scripts
│
├── .github/                     # GitHub configuration
│   └── workflows/
│       └── ci.yml               # CI/CD pipeline
│
├── .editorconfig                # Editor configuration
├── .gitignore                   # Git ignore rules
├── README.md                    # Main documentation
├── INSTALL.md                   # Installation guide
├── STRUCTURE.md                 # This file
└── LICENSE                      # MIT License (duplicated from integration/)

```

## Component Relationships

```
┌─────────────────────────────────────────────────────────────────┐
│                     User / CI/CD Pipeline                        │
└──────────────┬──────────────────────────────────────────────────┘
               │
               ├──► apex-runner (integration/src/apex_harness/cli.py)
               │    │
               │    ├──► Model Registry (models.py)
               │    ├──► Task Discovery (task_discovery.py)
               │    ├──► Task Executor (task_executor.py)
               │    │    │
               │    │    └──► Calls: apx reports run [OPTIONS]
               │    │         │
               │    └──────────┘
               │
               └──► apx (apex_code/cli/main.py)
                    │
                    ├──► tasks - List/validate tasks
                    ├──► reports - Run evaluations
                    │    │
                    │    ├──► Harness (harness/executor.py)
                    │    │    │
                    │    │    ├──► Docker Manager
                    │    │    ├──► LLM Adapters
                    │    │    └──► Tool Executors
                    │    │
                    │    └──► Evaluator (harness/evaluator.py)
                    │
                    ├──► runs - Manage evaluation runs
                    └──► datasets - Manage datasets
```

## Data Flow

```
1. User runs: apex-runner --model claude --tasks task1 --parallel

2. Integration runner:
   ├─ Discovers tasks from tasks/ directory
   ├─ Loads model config from MODEL_REGISTRY
   ├─ Creates status CSV tracker
   └─ For each task:
      ├─ Constructs apx command:
      │  apx reports run <report-name> \
      │    --tasks <task> \
      │    --models <model-id> \
      │    --n-trials 3 \
      │    --max-workers 3 \
      │    --timeout 3600
      │
      └─ Executes command → APEX harness

3. APEX harness (apx):
   ├─ Parses task from tasks/ directory
   ├─ Spins up Docker container
   ├─ Initializes LLM adapter
   ├─ Runs task with AI model
   ├─ Evaluates results against rubric
   └─ Generates report

4. Integration runner:
   ├─ Tracks status in CSV
   └─ Prints summary
```

## Installation Flow

```bash
# 1. Install APEX harness (provides apx command)
cd apex_code
pip install -e .

# 2. Install integration runner (provides apex-runner command)
cd ../integration
pip install -e .

# Both are now available:
apx --help
apex-runner --help
```

## Key Files

### Configuration
- `apex_code/pyproject.toml` - APEX harness package config
- `integration/pyproject.toml` - Integration runner package config
- `integration/src/apex_harness/models.py` - Model registry

### Entry Points
- `apex_code/cli/main.py` - `apx` command entry point
- `integration/src/apex_harness/cli.py` - `apex-runner` command entry point

### Core Logic
- `apex_code/harness/executor.py` - Task execution engine
- `integration/src/apex_harness/task_executor.py` - Command construction

### Testing
- `integration/tests/` - Unit and integration tests (40 tests)
- `integration/scripts/validate_refactoring.py` - Regression validation

### Documentation
- `README.md` - Main overview
- `apex_code/README.md` - APEX harness docs
- `integration/README.md` - Integration runner docs
- `INSTALL.md` - Installation guide
- `integration/docs/ARCHITECTURE.md` - Architecture decisions
- `integration/docs/MIGRATION.md` - Migration from legacy scripts

## Generated Artifacts

All generated files go to `integration/artifacts/` (gitignored):

```
integration/artifacts/
├── claude_tasks_status_20260114-120000.csv
├── gemini_tasks_status_20260114-120500.csv
├── deepseek_tasks_status_20260114-121000.csv
└── ... (other model CSVs and reports)
```

## Legacy Code

The `temp/` directory contains the original cluttered codebase:
- Used for regression validation
- To be removed after final verification
- **Do not modify or use for new development**

## Development Workflow

```bash
# 1. Make changes to code

# 2. Run tests
cd integration
pytest tests/ -v

# 3. Validate refactoring
python scripts/validate_refactoring.py

# 4. Format code
black src/ tests/

# 5. Lint
ruff check src/ tests/

# 6. Commit changes
git add .
git commit -m "feat: your changes"
```

## CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/ci.yml`):

1. **Lint** - Run code quality checks
2. **Test** - Run 40 unit/integration tests
3. **Build** - Build both packages
4. **Deploy** (future) - Publish to PyPI

## Design Patterns

### Strategy Pattern
- Location: `integration/src/apex_harness/models.py`
- Purpose: Model-specific configurations
- Benefit: Easy to add new models

### Factory Pattern
- Location: `apex_code/harness/executor.py`
- Purpose: Create appropriate executors
- Benefit: Flexible execution strategies

### Dependency Injection
- Throughout codebase
- Purpose: Testable, modular components
- Benefit: Easy mocking and testing

## Version Compatibility

- **Python**: 3.10+
- **Docker**: Any recent version
- **OS**: Linux, macOS, Windows (with WSL2)

## Next Steps

1. ✅ Setup complete
2. ✅ Tests passing (40/40)
3. ✅ Zero regression validated
4. 🔄 Ready for testing on EC2
5. 📦 Ready for production use
6. ⏳ Remove `temp/` after final verification

---

For more details, see:
- [README.md](README.md) - Main overview
- [INSTALL.md](INSTALL.md) - Installation
- [integration/docs/ARCHITECTURE.md](integration/docs/ARCHITECTURE.md) - Architecture
