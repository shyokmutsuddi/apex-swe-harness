# Quick Command Reference for EC2 Testing

## Setup Commands (One-Time)

```bash
# 1. Navigate to repository
cd apex-swe-harness

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install APEX harness (apx command)
cd apex_code
pip install -e .
cd ..

# 4. Install integration runner (apex-runner command)
cd integration
pip install -e .
cd ..

# 5. Verify installation
apx --help
apex-runner --help

# 6. Set API keys (required for models you're testing)
export ANTHROPIC_API_KEY='your-key-here'
export FIREWORKS_API_KEY='your-key-here'
export GOOGLE_API_KEY='your-key-here'
export XAI_API_KEY='your-key-here'
export OPENAI_API_KEY='your-key-here'
export MOONSHOT_API_KEY='your-key-here'
```

## Test Commands

### Test 1: Dry Run (Preview Commands)

```bash
cd integration
apex-runner --model claude \
  --tasks 1-aws-s3-snapshots 2-localstack-s3-snapshots \
  --tasks-dir ../tasks \
  --dry-run
```

### Test 2: Single Task

```bash
cd integration
apex-runner --model claude \
  --tasks 1-aws-s3-snapshots \
  --tasks-dir ../tasks \
  --max-workers 1
```

### Test 3: Run 5 Tasks with Claude (as requested)

```bash
cd integration

# Define the 5 tasks
export TASKS="1-aws-s3-snapshots 2-localstack-s3-snapshots 3-localstack-s3-snapshots 4-localstack-notifications 5-localstack-s3-notifications"

# Run with Claude in parallel
apex-runner --model claude \
  --tasks $TASKS \
  --tasks-dir ../tasks \
  --parallel --max-workers 3
```

### Test 4: Run All 8 Models Sequentially

```bash
cd integration

export TASKS="1-aws-s3-snapshots 2-localstack-s3-snapshots 3-localstack-s3-snapshots 4-localstack-notifications 5-localstack-s3-notifications"

# Run each model one by one
for model in claude opus gemini xai deepseek qwen codex kimi; do
  echo "========================================"
  echo "Running model: $model"
  echo "========================================"
  
  apex-runner --model $model \
    --tasks $TASKS \
    --tasks-dir ../tasks \
    --parallel --max-workers 3
  
  echo "Completed: $model"
  echo ""
done

echo "All models completed!"
```

### Test 5: Check Results

```bash
cd integration

# List all generated CSV files
ls -lh artifacts/*.csv

# View Claude results
cat artifacts/claude_tasks_status_*.csv

# Count total tasks completed
grep ",completed," artifacts/*.csv | wc -l

# Count total tasks failed
grep ",failed," artifacts/*.csv | wc -l
```

## Complete One-Liner Setup + Test

```bash
cd apex-swe-harness && \
python3 -m venv venv && \
source venv/bin/activate && \
cd apex_code && pip install -e . && cd .. && \
cd integration && pip install -e . && cd .. && \
export ANTHROPIC_API_KEY='your-key' && \
cd integration && \
export TASKS="1-aws-s3-snapshots 2-localstack-s3-snapshots 3-localstack-s3-snapshots 4-localstack-notifications 5-localstack-s3-notifications" && \
apex-runner --model claude --tasks $TASKS --tasks-dir ../tasks --parallel --max-workers 3
```

## Troubleshooting Commands

```bash
# Check if apx is installed
which apx
pip list | grep apex-code

# Check if apex-runner is installed
which apex-runner
pip list | grep apex-swe

# Check Docker
docker ps
docker version

# Check API keys
env | grep API_KEY

# Clean Docker resources
docker system prune -a -f

# Reinstall if needed
cd apex_code && pip install -e . --force-reinstall && cd ..
cd integration && pip install -e . --force-reinstall && cd ..
```

## Expected Output

### Successful Execution

```
[1/5] Running: 1-aws-s3-snapshots
✓ Completed: 1-aws-s3-snapshots (return code: 0)

[2/5] Running: 2-localstack-s3-snapshots
✓ Completed: 2-localstack-s3-snapshots (return code: 0)

[3/5] Running: 3-localstack-s3-snapshots
✓ Completed: 3-localstack-s3-snapshots (return code: 0)

[4/5] Running: 4-localstack-notifications
✓ Completed: 4-localstack-notifications (return code: 0)

[5/5] Running: 5-localstack-s3-notifications
✓ Completed: 5-localstack-s3-notifications (return code: 0)

================================================================================
SUMMARY
================================================================================
Total tasks: 5
Successful: 5
Failed: 0

Status CSV saved to: artifacts/claude_tasks_status_20260114-120000.csv
```

### Generated Files

```
integration/artifacts/
├── claude_tasks_status_20260114-120000.csv
├── opus_tasks_status_20260114-120500.csv
├── gemini_tasks_status_20260114-121000.csv
├── xai_tasks_status_20260114-121500.csv
├── deepseek_tasks_status_20260114-122000.csv
├── qwen_tasks_status_20260114-122500.csv
├── codex_tasks_status_20260114-123000.csv
└── kimi_tasks_status_20260114-123500.csv
```

## CSV Format

```csv
task_name,status,start_time,end_time,duration_seconds,return_code,error_message
1-aws-s3-snapshots,completed,2026-01-14 12:00:00,2026-01-14 12:15:30,930.5,0,
2-localstack-s3-snapshots,completed,2026-01-14 12:15:31,2026-01-14 12:28:45,794.2,0,
3-localstack-s3-snapshots,failed,2026-01-14 12:28:46,2026-01-14 12:35:12,386.1,1,Timeout exceeded
```

## Performance Estimates

- **Single task**: 5-15 minutes
- **5 tasks parallel (3 workers)**: 15-30 minutes
- **5 tasks sequential**: 30-60 minutes
- **All 8 models (5 tasks each)**: 2-4 hours

## Quick Reference

| Command | Purpose |
|---------|---------|
| `apx --help` | Show APEX harness help |
| `apex-runner --help` | Show integration runner help |
| `apex-runner --model claude --dry-run` | Preview commands |
| `apex-runner --model claude --tasks task1` | Run single task |
| `apex-runner --model claude --parallel` | Run all tasks in parallel |
| `ls artifacts/*.csv` | List result files |

---

For detailed guide, see [EC2_TESTING.md](EC2_TESTING.md)
