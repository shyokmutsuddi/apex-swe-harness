# Repository Structure

## Overview

This repository is organized for integration testing of AI models on software engineering tasks.

## Directory Layout

```
apex-swe-harness/
├── integration/              # Integration testing harness (main component)
│   ├── src/apex_harness/     # Source code for test harness
│   │   ├── cli.py            # Command-line interface
│   │   ├── models.py         # Model registry (8 AI models)
│   │   ├── runner.py         # Task execution orchestration
│   │   ├── task_executor.py  # Individual task execution
│   │   ├── status_tracker.py # Progress tracking (CSV)
│   │   ├── task_discovery.py # Task scanning
│   │   ├── docker_utils.py   # Docker resource management
│   │   └── reporting.py      # Result reporting
│   ├── tests/                # Test suite (40 tests)
│   │   ├── test_cli.py       # CLI tests
│   │   ├── test_models.py    # Model registry tests
│   │   ├── test_integration.py # Integration tests
│   │   └── ...
│   ├── docs/                 # Documentation
│   │   ├── ARCHITECTURE.md   # Design patterns and architecture
│   │   ├── MIGRATION.md      # Migration from legacy scripts
│   │   └── TESTING.md        # Testing guide
│   ├── examples/             # Usage examples
│   │   ├── basic_usage.sh    # Shell script examples
│   │   └── python_usage.py   # Python API examples
│   ├── scripts/              # Helper scripts
│   │   ├── run_integ_set.sh  # Task set runner
│   │   └── validate_refactoring.py # Regression validation
│   ├── tasks/                # Task definitions (can also use ../tasks/)
│   ├── artifacts/            # Generated outputs (gitignored)
│   ├── README.md             # Integration harness documentation
│   ├── QUICKSTART.md         # Quick start guide
│   ├── pyproject.toml        # Package configuration
│   ├── setup.py              # Setup script
│   ├── pytest.ini            # Test configuration
│   ├── LICENSE               # MIT License
│   ├── CONTRIBUTING.md       # Contribution guidelines
│   ├── CODE_OF_CONDUCT.md    # Code of conduct
│   └── SECURITY.md           # Security policy
│
├── tasks/                    # Shared task definitions (optional)
│   ├── 1-aws-s3-snapshots/
│   ├── 2-localstack-s3-snapshots/
│   └── ...
│
├── temp/                     # Legacy scripts (read-only, temporary)
│   ├── *_run_all_tasks.py    # Original duplicated scripts
│   ├── apex_code/            # Original codebase
│   └── tasks/                # Original task definitions
│
├── observability/            # Observability features (if any)
│
├── .github/                  # GitHub configuration
│   └── workflows/
│       └── ci.yml            # CI/CD pipeline
│
├── .gitignore                # Git ignore rules
├── README.md                 # Repository overview
└── STRUCTURE.md              # This file
```

## Key Components

### Integration Harness (`integration/`)

The main component - a production-grade harness for running integration tests:

- **Unified CLI**: Single `apex-runner` command replaces 8+ duplicate scripts
- **Model Registry**: Easy configuration for 8 AI models
- **Parallel Execution**: Thread-safe concurrent task execution
- **Progress Tracking**: Real-time CSV status tracking
- **Docker Management**: Automatic cleanup of Docker resources
- **40+ Tests**: Comprehensive unit and integration tests
- **Zero Regression**: Validated against original implementation

### Tasks (`tasks/` or `integration/tasks/`)

Task definitions for integration tests. Can be placed either:
- At repository root: `tasks/`
- Within integration: `integration/tasks/`

The harness will look in both locations (root takes precedence).

### Legacy Code (`temp/`)

Original cluttered implementation preserved for:
- Regression validation
- Output comparison
- Migration reference

**Will be removed** once migration is complete and validated.

## Usage

### Installation

```bash
cd integration
pip install -e .
```

### Running Tests

```bash
# From integration directory
apex-runner --model claude --parallel

# Or specify path
apex-runner --model claude --tasks-dir ../tasks
```

### Development

```bash
cd integration

# Run tests
pytest tests/

# Validate refactoring
python scripts/validate_refactoring.py
```

## Migration Status

✅ **Complete**
- All code moved to `integration/`
- All tests passing (40/40)
- Zero regression validated
- Documentation updated
- CI/CD updated

## Design Philosophy

### Why `integration/`?

The harness is specifically for **integration testing** of AI models:
1. Tests run in isolated Docker environments
2. Evaluates end-to-end task completion
3. Measures AI model performance on real tasks
4. Not unit/functional testing - true integration tests

### Separation of Concerns

- `integration/`: Test harness and tooling
- `tasks/`: Test definitions and specifications
- `temp/`: Legacy code (temporary)
- Root: Repository organization and documentation

## Documentation

- **Main README**: [integration/README.md](integration/README.md)
- **Quick Start**: [integration/QUICKSTART.md](integration/QUICKSTART.md)
- **Architecture**: [integration/docs/ARCHITECTURE.md](integration/docs/ARCHITECTURE.md)
- **Migration**: [integration/docs/MIGRATION.md](integration/docs/MIGRATION.md)
- **Testing**: [integration/docs/TESTING.md](integration/docs/TESTING.md)
- **Contributing**: [integration/CONTRIBUTING.md](integration/CONTRIBUTING.md)
- **This File**: [STRUCTURE.md](STRUCTURE.md) (repository overview)

## Maintenance

### Adding New Models

1. Edit `integration/src/apex_harness/models.py`
2. Add test in `integration/tests/test_models.py`
3. Update documentation

### Adding New Tasks

1. Create task directory in `tasks/` or `integration/tasks/`
2. Follow existing task structure
3. Test with `apex-runner --tasks your-new-task`

### Running Validation

```bash
cd integration
python scripts/validate_refactoring.py
```

Expected: All validations pass, confirming zero regression.
