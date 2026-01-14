# Migration Guide

This guide helps you migrate from the legacy `*_run_all_tasks.py` scripts to the new unified `apex-runner` CLI.

## Quick Migration

### Before (Legacy)

```bash
# Old way - separate script per model
python claude_run_all_tasks.py --parallel --max-workers 5
python xai_run_all_tasks.py --tasks task1 task2
python gemini_run_all_tasks.py --dry-run
```

### After (Unified)

```bash
# New way - single CLI with model selection
apex-runner --model claude --parallel --max-workers 5
apex-runner --model xai --tasks task1 task2
apex-runner --model gemini --dry-run
```

## Command Mapping

### Claude

```bash
# Old
python claude_run_all_tasks.py [OPTIONS]

# New
apex-runner --model claude [OPTIONS]
```

### XAI/Grok

```bash
# Old
python xai_run_all_tasks.py [OPTIONS]

# New
apex-runner --model xai [OPTIONS]
```

### Gemini

```bash
# Old
python gemini_run_all_tasks.py [OPTIONS]

# New
apex-runner --model gemini [OPTIONS]
```

### DeepSeek

```bash
# Old
python deepseek_run_all_tasks.py [OPTIONS]

# New
apex-runner --model deepseek [OPTIONS]
```

### Qwen

```bash
# Old
python qwen_run_all_tasks.py [OPTIONS]

# New
apex-runner --model qwen [OPTIONS]
```

### Opus

```bash
# Old
python opus_run_all_tasks.py [OPTIONS]

# New
apex-runner --model opus [OPTIONS]
```

### Codex

```bash
# Old
python codex_run_all_tasks.py [OPTIONS]

# New
apex-runner --model codex [OPTIONS]
```

### Kimi

```bash
# Old
python kimi_all_all_tasks.py [OPTIONS]

# New
apex-runner --model kimi [OPTIONS]
```

## Option Mapping

All options remain the same:

| Legacy Option | New Option | Notes |
|--------------|------------|-------|
| `--parallel` | `--parallel` | ✅ Identical |
| `--max-workers N` | `--max-workers N` | ✅ Identical |
| `--tasks T1 T2...` | `--tasks T1 T2...` | ✅ Identical |
| `--tasks-dir PATH` | `--tasks-dir PATH` | ✅ Identical |
| `--dry-run` | `--dry-run` | ✅ Identical |
| `--status-csv PATH` | `--status-csv PATH` | ✅ Identical |

## Output Changes

### CSV Status Files

**Before**: Files created in current directory
```
claude_tasks_status_20260114-120000.csv
xai_tasks_status_20260114-120000.csv
```

**After**: Files created in `artifacts/` directory (configurable)
```
artifacts/claude_tasks_status_20260114-120000.csv
artifacts/xai_tasks_status_20260114-120000.csv
```

**Migration**: Update any scripts that reference these files.

### CSV Format

The CSV format is **unchanged**:
- Same columns
- Same field names
- Same value formats

✅ **No changes needed** to any CSV parsing code.

## Script Migration

### Shell Scripts

If you have shell scripts calling the old scripts:

```bash
# Before
#!/bin/bash
python claude_run_all_tasks.py --parallel --max-workers 3
```

```bash
# After
#!/bin/bash
apex-runner --model claude --parallel --max-workers 3
```

### Python Scripts

If you're importing from the old scripts:

```python
# Before
from claude_run_all_tasks import run_sequential, CSVStatusTracker
```

```python
# After
from apex_harness.runner import run_sequential
from apex_harness.status_tracker import CSVStatusTracker
```

## Environment Variables

No changes - all environment variables work the same:

```bash
# Fireworks-based models still need:
export FIREWORKS_API_KEY='your-key'
```

## Backward Compatibility

### Wrapper Script

For maximum compatibility, use the provided wrapper:

```bash
# Old interface still works via wrapper
./scripts/run_integ_set.sh --model claude integ-set-1.txt --parallel
```

The wrapper delegates to the new CLI internally.

### Legacy Script Location

If you need the old scripts temporarily:

1. They're preserved in `temp/` directory (read-only snapshot)
2. Use only for validation and comparison
3. Will be removed once migration is complete

## Validation Steps

### 1. Test Dry Run

```bash
# Old
python claude_run_all_tasks.py --dry-run

# New
apex-runner --model claude --dry-run

# Compare outputs - should be identical
```

### 2. Test Single Task

```bash
# Old
python claude_run_all_tasks.py --tasks test-task

# New
apex-runner --model claude --tasks test-task

# Verify CSV output format matches
```

### 3. Test Parallel Execution

```bash
# Old
python claude_run_all_tasks.py --parallel --max-workers 2

# New
apex-runner --model claude --parallel --max-workers 2

# Check for any behavioral differences
```

### 4. Compare CSV Outputs

```python
# Validation script
import csv

def compare_csvs(old_path, new_path):
    with open(old_path) as f:
        old_data = list(csv.DictReader(f))
    with open(new_path) as f:
        new_data = list(csv.DictReader(f))
    
    assert old_data == new_data, "CSV outputs differ!"
    print("✅ CSV outputs match exactly")

compare_csvs(
    "temp/claude_tasks_status_OLD.csv",
    "artifacts/claude_tasks_status_NEW.csv"
)
```

## Troubleshooting

### Issue: Command not found

**Problem**: `apex-runner: command not found`

**Solution**:
```bash
# Reinstall package
pip install -e .

# Or use python module syntax
python -m apex_harness.cli --model claude
```

### Issue: Tasks directory not found

**Problem**: `Error: Tasks directory not found`

**Solution**:
```bash
# Specify tasks directory explicitly
apex-runner --model claude --tasks-dir ./tasks

# Or create symlink/copy tasks to expected location
ln -s temp/tasks tasks
```

### Issue: Import errors

**Problem**: `ModuleNotFoundError: No module named 'apex_harness'`

**Solution**:
```bash
# Install package in development mode
pip install -e .

# Verify installation
python -c "import apex_harness; print(apex_harness.__version__)"
```

### Issue: Different output location

**Problem**: Can't find CSV status files

**Solution**:
```bash
# Check artifacts directory
ls artifacts/

# Or specify custom location
apex-runner --model claude --artifacts-dir ./outputs
```

## Rollback Plan

If you need to rollback temporarily:

1. **Use legacy scripts** from `temp/`:
   ```bash
   cd temp
   python claude_run_all_tasks.py [OPTIONS]
   ```

2. **Preserve old outputs** before migration:
   ```bash
   mkdir backup
   cp *_tasks_status_*.csv backup/
   ```

3. **Document issues** for investigation

## Complete Migration Checklist

- [ ] Install new package: `pip install -e .`
- [ ] Test dry run for each model
- [ ] Update shell scripts to use `apex-runner`
- [ ] Update Python imports if any
- [ ] Update CI/CD pipelines
- [ ] Update documentation/runbooks
- [ ] Validate CSV output format
- [ ] Test parallel execution
- [ ] Update artifact paths in downstream tools
- [ ] Remove or archive legacy scripts
- [ ] Update team documentation

## Benefits After Migration

- ✅ Single command to remember
- ✅ Consistent interface across all models
- ✅ Better error messages
- ✅ Improved documentation
- ✅ Comprehensive test coverage
- ✅ Easier to add new models
- ✅ Better maintainability

## Questions?

See [CONTRIBUTING.md](../CONTRIBUTING.md) or open an issue.
