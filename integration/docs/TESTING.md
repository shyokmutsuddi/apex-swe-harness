# Testing Guide

## Test Suite Overview

The APEX SWE Harness includes comprehensive test coverage with 40+ tests covering unit, integration, and regression scenarios.

## Running Tests

### All Tests
```bash
pytest tests/ -v
```

### Unit Tests Only
```bash
pytest tests/ -m "not integration" -v
```

### Integration Tests Only
```bash
pytest tests/ -m integration -v
```

### With Coverage
```bash
pytest tests/ --cov=apex_harness --cov-report=term --cov-report=html
```

### Specific Test File
```bash
pytest tests/test_models.py -v
```

### Specific Test
```bash
pytest tests/test_models.py::test_model_registry_exists -v
```

## Test Organization

### Unit Tests

#### `test_models.py`
Tests for model registry and configuration.

**Coverage**:
- Model registry population
- Model configuration structure
- Model lookup (valid/invalid)
- Model enumeration
- Configuration consistency
- Legacy compatibility

#### `test_status_tracker.py`
Tests for CSV status tracking.

**Coverage**:
- Tracker initialization
- New task status updates
- Existing task updates
- Multiple task tracking
- Thread safety

#### `test_task_discovery.py`
Tests for task discovery logic.

**Coverage**:
- Empty directory handling
- Valid task discovery
- Exclusion logic (shared, hidden, files)
- Sorting
- Error handling

#### `test_cli.py`
Tests for CLI interface.

**Coverage**:
- Argument parser creation
- Required arguments
- Argument validation
- Flag parsing
- Environment validation
- Basic execution flow

#### `test_reporting.py`
Tests for result reporting.

**Coverage**:
- Summary printing (all success)
- Summary with failures
- Exit code determination
- Empty results

### Integration Tests

#### `test_integration.py`
End-to-end integration tests.

**Coverage**:
- Command generation matches legacy
- Model-specific configurations (XAI, Qwen)
- CSV output format
- Docker cleanup invocation
- Timeout handling

## Test Markers

Use pytest markers to categorize tests:

- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.unit`: Unit tests (default)

## Writing Tests

### Test Naming Convention
```python
def test_<what>_<condition>_<expected>():
    """Test that <description>."""
```

**Examples**:
```python
def test_model_config_returns_valid_config():
    """Test that get_model_config returns valid configuration."""

def test_status_tracker_update_new_task():
    """Test adding a new task status."""

def test_get_all_tasks_excludes_shared():
    """Test that 'shared' directory is excluded."""
```

### Test Structure

Follow the **Arrange-Act-Assert** pattern:

```python
def test_example():
    """Test example function."""
    # Arrange
    config = get_model_config("claude")
    
    # Act
    result = some_function(config)
    
    # Assert
    assert result.success is True
```

### Using Fixtures

Create reusable test fixtures:

```python
import pytest
from pathlib import Path

@pytest.fixture
def temp_tasks_dir(tmp_path):
    """Create temporary tasks directory."""
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    (tasks_dir / "task1").mkdir()
    (tasks_dir / "task2").mkdir()
    return tasks_dir

def test_with_fixture(temp_tasks_dir):
    """Test using fixture."""
    tasks = get_all_tasks(temp_tasks_dir)
    assert len(tasks) == 2
```

### Mocking External Dependencies

Use `unittest.mock` for external dependencies:

```python
from unittest.mock import MagicMock, patch

def test_command_execution():
    """Test command execution with mock."""
    with patch("apex_harness.task_executor.subprocess.run") as mock_run:
        with patch("apex_harness.task_executor.cleanup_docker"):
            mock_run.return_value = MagicMock(
                returncode=0, 
                stdout="output", 
                stderr=""
            )
            
            result = run_task_command("task1", "timestamp", config)
            
            assert result[1] == 0  # return code
            mock_run.assert_called_once()
```

## Continuous Integration

Tests run automatically on:
- Push to main/develop
- Pull requests to main
- Multiple OS (Linux, Windows, macOS)
- Multiple Python versions (3.10, 3.11, 3.12)

## Regression Testing

### Validation Script

Run comprehensive regression validation:

```bash
python scripts/validate_refactoring.py
```

This validates:
1. Command generation matches legacy
2. CSV format matches legacy
3. Command construction matches legacy
4. Model registry completeness

Expected output:
```
======================================================================
*** ALL VALIDATIONS PASSED ***
======================================================================

Refactoring is complete with ZERO REGRESSION.
```

### Manual Regression Testing

1. **Command Generation**:
   ```bash
   apex-runner --model claude --dry-run
   # Verify command matches: apx reports run claude-{task}-{timestamp} ...
   ```

2. **CSV Format**:
   ```bash
   apex-runner --model claude --tasks test-task
   # Check artifacts/claude_tasks_status_*.csv format
   ```

3. **Parallel Execution**:
   ```bash
   apex-runner --model gemini --parallel --max-workers 3
   # Verify concurrent execution works
   ```

## Test Coverage

Current coverage targets:

| Module | Coverage | Status |
|--------|----------|--------|
| models.py | 100% | ✅ |
| status_tracker.py | 95%+ | ✅ |
| task_discovery.py | 100% | ✅ |
| cli.py | 85%+ | ✅ |
| runner.py | 90%+ | ✅ |
| reporting.py | 100% | ✅ |
| task_executor.py | 90%+ | ✅ |
| docker_utils.py | 80%+ | ✅ |

## Adding New Tests

When adding new functionality:

1. **Write tests first** (TDD approach)
2. **Test edge cases** (empty inputs, errors, boundaries)
3. **Test normal flow** (happy path)
4. **Mock external dependencies** (subprocess, file I/O)
5. **Use descriptive names** and docstrings
6. **Run tests locally** before committing
7. **Verify CI passes** after push

## Test Data

Use temporary directories for test data:

```python
def test_with_temp_dir(tmp_path):
    """Test with temporary directory."""
    test_file = tmp_path / "test.csv"
    # Create test file
    # Run test
    # tmp_path is automatically cleaned up
```

## Debugging Tests

### Verbose Output
```bash
pytest tests/ -v
```

### Show Print Statements
```bash
pytest tests/ -s
```

### Drop into Debugger on Failure
```bash
pytest tests/ --pdb
```

### Run Specific Test with Output
```bash
pytest tests/test_models.py::test_specific_test -v -s
```

## Performance Testing

For performance-critical code, use benchmarking:

```python
import time

def test_performance():
    """Test execution performance."""
    start = time.time()
    
    # Run operation
    result = expensive_operation()
    
    duration = time.time() - start
    assert duration < 1.0  # Should complete in <1 second
```

## Best Practices

1. ✅ Keep tests independent (no shared state)
2. ✅ Use meaningful test names
3. ✅ Test one thing per test
4. ✅ Use fixtures for common setup
5. ✅ Mock external dependencies
6. ✅ Assert specific values, not just truthiness
7. ✅ Use pytest's built-in assertions
8. ✅ Add docstrings to tests
9. ✅ Clean up resources (use context managers)
10. ✅ Run tests before committing

## Troubleshooting

### Import Errors
```bash
# Reinstall package in development mode
pip install -e .
```

### Test Failures
```bash
# Run with verbose output
pytest tests/test_file.py::test_name -v -s
```

### Coverage Not Updated
```bash
# Clear cache and re-run
pytest --cache-clear --cov=apex_harness tests/
```

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [unittest.mock Documentation](https://docs.python.org/3/library/unittest.mock.html)
