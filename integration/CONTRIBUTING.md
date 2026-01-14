# Contributing to APEX SWE Harness

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct (see CODE_OF_CONDUCT.md).

## Getting Started

### Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/apex-swe-harness.git
   cd apex-swe-harness
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Verify installation**
   ```bash
   pytest tests/
   ```

## Development Workflow

### Making Changes

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clear, readable code
   - Follow existing code style
   - Add tests for new functionality
   - Update documentation as needed

3. **Format and lint your code**
   ```bash
   black src/ tests/
   ruff check src/ tests/ --fix
   mypy src/
   ```

4. **Run tests**
   ```bash
   pytest tests/ -v
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

   Use conventional commit format:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation changes
   - `test:` for test additions/changes
   - `refactor:` for code refactoring
   - `chore:` for maintenance tasks

6. **Push and create a pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

### Pull Request Guidelines

- **Title**: Clear, descriptive title following conventional commit format
- **Description**: Explain what changed and why
- **Tests**: Include tests for new functionality
- **Documentation**: Update relevant documentation
- **Commits**: Keep commits atomic and well-described
- **Review**: Address review feedback promptly

## Code Style

### Python Style Guide

- Follow PEP 8
- Use Black for formatting (line length: 100)
- Use type hints for function signatures
- Write docstrings for public functions and classes

Example:
```python
def process_task(
    task_name: str,
    config: ModelConfig,
    timeout: int = 3600
) -> TaskResult:
    """
    Process a single task with the given configuration.

    Args:
        task_name: Name of the task to process
        config: Model configuration
        timeout: Maximum execution time in seconds

    Returns:
        TaskResult containing execution details

    Raises:
        TaskExecutionError: If task execution fails
    """
    ...
```

### Testing Guidelines

- Write unit tests for all new functionality
- Maintain or improve code coverage
- Use descriptive test names: `test_<what>_<condition>_<expected>`
- Use pytest fixtures for common setup
- Mark integration tests with `@pytest.mark.integration`

Example:
```python
def test_model_config_returns_valid_config():
    """Test that get_model_config returns valid configuration."""
    config = get_model_config("claude")
    assert config.name == "Claude Sonnet 4.5"
```

## Project Structure

```
apex-swe-harness/
├── src/apex_harness/      # Main package code
│   ├── __init__.py
│   ├── cli.py             # Command-line interface
│   ├── models.py          # Model registry
│   ├── runner.py          # Execution orchestration
│   ├── task_executor.py   # Task execution
│   ├── status_tracker.py  # Progress tracking
│   └── ...
├── tests/                 # Test suite
│   ├── test_models.py
│   ├── test_cli.py
│   └── ...
├── docs/                  # Documentation
├── examples/              # Usage examples
├── scripts/               # Helper scripts
└── artifacts/             # Generated outputs (gitignored)
```

## Adding a New Model

To add support for a new AI model:

1. **Add model configuration** in `src/apex_harness/models.py`:
   ```python
   "new-model": ModelConfig(
       name="New Model Name",
       identifier="api-model-identifier",
       report_prefix="newmodel",
       status_csv_prefix="newmodel",
       n_trials=3,
       max_workers=3,
       timeout=3600,
       requires_env_var="NEW_MODEL_API_KEY",  # Optional
   ),
   ```

2. **Add tests** in `tests/test_models.py`:
   ```python
   def test_new_model_configuration():
       config = get_model_config("new-model")
       assert config.identifier == "api-model-identifier"
   ```

3. **Update documentation** in README.md and relevant docs

4. **Test thoroughly**:
   ```bash
   apex-runner --model new-model --tasks test-task --dry-run
   ```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest tests/ --cov=apex_harness --cov-report=term

# Run only unit tests
pytest tests/ -m "not integration"

# Run only integration tests
pytest tests/ -m integration

# Run with verbose output
pytest tests/ -v
```

## Documentation

- Update README.md for user-facing changes
- Update ARCHITECTURE.md for design changes
- Add docstrings to all public functions/classes
- Include examples in documentation

## Questions or Issues?

- Check existing issues before creating new ones
- Provide minimal reproducible examples for bugs
- Include system info (OS, Python version, package version)
- Tag issues appropriately (bug, feature, documentation, etc.)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
