# APEX SWE Harness Architecture

## Overview

The APEX SWE Harness is designed as a modular, extensible system for evaluating AI models on software engineering tasks. The architecture emphasizes:

- **High Cohesion**: Related functionality grouped together
- **Low Coupling**: Minimal dependencies between modules
- **Extensibility**: Easy addition of new models and features
- **Testability**: Clean interfaces enable comprehensive testing
- **Maintainability**: Clear separation of concerns

## Design Patterns

### Strategy Pattern

Used for model selection and execution strategies.

```
┌─────────────────┐
│  ModelConfig    │  ← Strategy Interface
├─────────────────┤
│ - name          │
│ - identifier    │
│ - report_prefix │
│ - n_trials      │
│ - max_workers   │
│ - timeout       │
└─────────────────┘
         △
         │ implements
         │
┌────────┴────────┬──────────┬──────────┐
│   Claude        │  Gemini  │   XAI    │ ...
│   Config        │  Config  │  Config  │
└─────────────────┴──────────┴──────────┘
```

**Benefits**:
- Add new models without modifying existing code
- Centralized configuration management
- Consistent behavior across models

### Factory Pattern

Used for creating executors and trackers.

```python
def create_tracker(csv_path: Path) -> CSVStatusTracker:
    """Factory for status tracker creation."""
    return CSVStatusTracker(csv_path)
```

**Benefits**:
- Flexible object creation
- Easy to mock in tests
- Encapsulates construction logic

### Dependency Injection

Used throughout for configurability and testability.

```python
def run_task_command(
    task_name: str,
    timestamp: str,
    model_config: ModelConfig,  # Injected
    tracker: Optional[CSVStatusTracker] = None,  # Injected
    dry_run: bool = False
) -> Tuple[str, int, str, str]:
    ...
```

**Benefits**:
- Easy to test with mocks
- Flexible configuration
- Clear dependencies

## Module Structure

### Core Modules

#### `models.py` - Model Registry

**Responsibility**: Model configuration management

**Key Components**:
- `ModelConfig`: Dataclass for model configuration
- `MODEL_REGISTRY`: Dictionary of all available models
- `get_model_config()`: Model lookup function
- `list_available_models()`: Model enumeration

**Design Decision**: Centralized registry eliminates duplication across 8+ scripts.

#### `cli.py` - Command-Line Interface

**Responsibility**: User interaction and command parsing

**Key Components**:
- `create_parser()`: Argument parser configuration
- `validate_environment()`: API key validation
- `main()`: Entry point orchestration

**Design Decision**: Single unified CLI replaces model-specific scripts.

#### `runner.py` - Execution Orchestration

**Responsibility**: Task execution strategy (sequential/parallel)

**Key Components**:
- `run_sequential()`: Sequential task execution
- `run_parallel()`: Parallel task execution with ThreadPoolExecutor

**Design Decision**: Separate strategies allow optimization for different use cases.

#### `task_executor.py` - Task Execution

**Responsibility**: Individual task execution logic

**Key Components**:
- `run_task_command()`: Execute single task
- Timeout handling
- Error tracking
- Docker cleanup integration

**Design Decision**: Single point of command execution ensures consistency.

#### `status_tracker.py` - Progress Tracking

**Responsibility**: Thread-safe status persistence

**Key Components**:
- `CSVStatusTracker`: Thread-safe CSV writer
- Atomic row updates
- Status state management

**Design Decision**: Thread-safe design enables parallel execution without data races.

#### `task_discovery.py` - Task Scanning

**Responsibility**: Task directory scanning

**Key Components**:
- `get_all_tasks()`: Directory enumeration
- Exclusion logic (shared, hidden)

**Design Decision**: Separate module for clear responsibility boundary.

#### `docker_utils.py` - Docker Management

**Responsibility**: Docker resource cleanup

**Key Components**:
- `cleanup_docker_networks()`: Network pruning
- `cleanup_docker_containers()`: Container pruning
- `cleanup_docker()`: Combined cleanup

**Design Decision**: Isolated utility prevents subnet exhaustion.

#### `reporting.py` - Result Reporting

**Responsibility**: Result aggregation and display

**Key Components**:
- `print_summary()`: Formatted result display
- `get_exit_code()`: Exit code determination

**Design Decision**: Separate reporting from execution for clarity.

## Data Flow

### Sequential Execution Flow

```
┌─────────┐
│   CLI   │
└────┬────┘
     │ parse args
     ▼
┌─────────────┐
│ get_model_  │
│  config()   │
└─────┬───────┘
      │
      ▼
┌─────────────────┐
│ get_all_tasks() │
└────┬────────────┘
     │
     ▼
┌──────────────────┐
│ run_sequential() │
└────┬─────────────┘
     │
     ├──► ┌────────────────────┐
     │    │ run_task_command() │──► Docker cleanup
     │    └────────────────────┘
     │           │
     │           ▼
     │    ┌─────────────────┐
     │    │ CSVStatusTracker│
     │    └─────────────────┘
     │
     ▼
┌──────────────┐
│print_summary │
└──────────────┘
```

### Parallel Execution Flow

```
┌─────────┐
│   CLI   │
└────┬────┘
     │
     ▼
┌────────────────┐
│ run_parallel() │
└────┬───────────┘
     │
     ├──► ThreadPoolExecutor
     │    ┌────────┬────────┬────────┐
     │    │ Task 1 │ Task 2 │ Task 3 │ ...
     │    └───┬────┴───┬────┴───┬────┘
     │        │        │        │
     │        ▼        ▼        ▼
     │    ┌────────────────────────┐
     │    │  run_task_command()    │ (parallel)
     │    └────────────────────────┘
     │               │
     │               ▼
     │    ┌─────────────────────┐
     │    │ CSVStatusTracker    │ (thread-safe)
     │    └─────────────────────┘
     │
     ▼
┌──────────────┐
│print_summary │
└──────────────┘
```

## Key Design Decisions

### 1. Model Registry over Script Duplication

**Problem**: 8 nearly identical scripts with only model configuration differences

**Solution**: Single registry with model configurations

**Trade-offs**:
- ✅ Eliminates 3500+ lines of duplication
- ✅ Single point of truth for model configs
- ✅ Easy to add new models
- ⚠️ Requires registry update for new models (acceptable)

### 2. Unified CLI over Per-Model Scripts

**Problem**: Multiple entry points, maintenance burden

**Solution**: Single CLI with `--model` flag

**Trade-offs**:
- ✅ Consistent interface across models
- ✅ Single codebase to maintain
- ✅ Better user experience
- ⚠️ Slightly longer command (mitigated by defaults)

### 3. Strategy Pattern for Execution

**Problem**: Different execution needs (sequential/parallel)

**Solution**: Separate functions for each strategy

**Trade-offs**:
- ✅ Clear separation of concerns
- ✅ Easy to add new strategies
- ✅ Optimized for each use case
- ⚠️ Two code paths to maintain (acceptable, well-tested)

### 4. Thread-Safe Status Tracking

**Problem**: Parallel execution needs coordinated status updates

**Solution**: Lock-based CSV writer

**Trade-offs**:
- ✅ Correct concurrent updates
- ✅ No data races
- ✅ Simple implementation
- ⚠️ Lock contention under high load (unlikely with typical workloads)

### 5. Artifacts Directory

**Problem**: Generated files scattered across working directory

**Solution**: Dedicated `artifacts/` directory

**Trade-offs**:
- ✅ Clean working directory
- ✅ Easy to .gitignore
- ✅ Clear separation of generated vs source files
- ✅ Configurable location

## Testing Strategy

### Unit Tests

Test individual modules in isolation with mocked dependencies.

**Coverage**:
- Model registry operations
- Status tracker operations
- Task discovery logic
- CLI argument parsing
- Result reporting

### Integration Tests

Test end-to-end flows with mocked external dependencies (subprocess, Docker).

**Coverage**:
- Command generation matches legacy
- CSV format compatibility
- Docker cleanup invocation
- Error handling flows

### Regression Tests

Verify refactored code produces identical outputs to original.

**Coverage**:
- Command construction
- CSV output format
- Exit codes
- Error messages

## Extensibility Points

### Adding a New Model

1. Add entry to `MODEL_REGISTRY` in `models.py`
2. Add tests to `test_models.py`
3. Update documentation

**No other code changes required!**

### Adding a New Execution Strategy

1. Implement new function in `runner.py`
2. Add CLI flag in `cli.py`
3. Add tests

**Example**: `run_batch()` for batched execution

### Adding New Output Format

1. Implement new reporter in `reporting.py`
2. Add CLI flag for format selection
3. Add tests

**Example**: JSON output format

## Performance Considerations

### Parallel Execution

- Uses `ThreadPoolExecutor` for I/O-bound tasks (subprocess calls)
- Thread-safe status updates with minimal lock contention
- Worker count configurable per model

### Docker Cleanup

- Runs after each task to prevent resource exhaustion
- Non-blocking (failures logged but don't stop execution)
- Timeout protection (30 seconds)

### CSV Updates

- Atomic writes to prevent corruption
- Read-modify-write pattern with lock
- Acceptable performance for typical task counts (<1000)

## Security Considerations

### API Keys

- Never hardcoded
- Environment variable-based
- Validated before execution
- Not logged or persisted

### Docker Security

- Cleanup prevents resource exhaustion
- Container isolation (handled by Docker)
- No privileged operations

### File System

- All writes to configured artifacts directory
- No writes outside designated paths
- Path validation in place

## Future Enhancements

### Potential Improvements

1. **Database Backend**: Replace CSV with SQLite for better concurrency
2. **Remote Execution**: Support for distributed task execution
3. **Real-time Dashboard**: Web UI for monitoring
4. **Retry Logic**: Configurable retry for transient failures
5. **Result Caching**: Skip redundant task executions
6. **Plugin System**: External model providers

### Backward Compatibility

Any future changes must maintain:
- CLI interface compatibility
- CSV output format
- Command generation logic
- Exit code behavior

## Conclusion

The refactored architecture achieves:
- ✅ **Zero Regression**: Identical behavior to original
- ✅ **Reduced Complexity**: 3500+ lines of duplication eliminated
- ✅ **Improved Maintainability**: Clear module boundaries
- ✅ **Enhanced Testability**: Comprehensive test coverage
- ✅ **Better Extensibility**: Easy to add models and features
- ✅ **Production Quality**: Proper error handling, logging, cleanup
