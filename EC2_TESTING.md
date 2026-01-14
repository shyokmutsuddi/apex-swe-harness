# EC2 Testing Guide

Complete guide for testing the APEX SWE Harness on EC2.

## Prerequisites

- EC2 instance with Docker installed
- Python 3.10 or higher
- Git
- Sufficient disk space (10GB+)
- Required API keys

## Step 1: Initial Setup

```bash
# Clone repository
git clone https://github.com/your-org/apex-swe-harness.git
cd apex-swe-harness

# Or if already cloned, pull latest
git pull origin main
```

## Step 2: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

## Step 3: Install Packages

```bash
# Install APEX Code Harness (apx command)
cd apex_code
pip install -e .
cd ..

# Install Integration Test Runner (apex-runner command)
cd integration
pip install -e .
cd ..

# Verify installations
which apx
which apex-runner

# Test commands
apx --help
apex-runner --help
```

**Expected output:**
```
✓ apx - Located at /path/to/venv/bin/apx
✓ apex-runner - Located at /path/to/venv/bin/apex-runner
```

## Step 4: Set Environment Variables

Set API keys for the models you want to test:

```bash
# Anthropic (Claude, Opus)
export ANTHROPIC_API_KEY='sk-ant-...'

# Google (Gemini)
export GOOGLE_API_KEY='AIza...'

# XAI (Grok)
export XAI_API_KEY='xai-...'

# Fireworks (DeepSeek, Qwen)
export FIREWORKS_API_KEY='fw_...'

# OpenAI (Codex)
export OPENAI_API_KEY='sk-...'

# Moonshot (Kimi)
export MOONSHOT_API_KEY='mk-...'

# Verify keys are set
env | grep API_KEY
```

## Step 5: Test APEX Harness (apx)

```bash
# Test basic command
apx --help

# List available tasks
apx tasks list

# Expected output:
# Available tasks:
#   - 1-aws-s3-snapshots
#   - 2-localstack-s3-snapshots
#   - 3-localstack-s3-snapshots
#   - 4-localstack-notifications
#   - 5-localstack-s3-notifications
```

## Step 6: Test Integration Runner (apex-runner)

### Option A: Dry Run (Preview Commands)

```bash
cd integration

# Test with Claude - dry run
apex-runner --model claude \
  --tasks 1-aws-s3-snapshots 2-localstack-s3-snapshots \
  --parallel --max-workers 3 \
  --dry-run

# Expected output:
# [DRY RUN] Would execute: apx reports run claude-1-aws-s3-snapshots-... --tasks 1-aws-s3-snapshots ...
# [DRY RUN] Would execute: apx reports run claude-2-localstack-s3-snapshots-... --tasks 2-localstack-s3-snapshots ...
```

### Option B: Run Single Task

```bash
cd integration

# Run single task with Claude
apex-runner --model claude \
  --tasks 1-aws-s3-snapshots \
  --tasks-dir ../tasks \
  --max-workers 1

# Expected output:
# [1/1] Running: 1-aws-s3-snapshots
# ✓ Completed: 1-aws-s3-snapshots (return code: 0)
# 
# ================================================================================
# SUMMARY
# ================================================================================
# Total tasks: 1
# Successful: 1
# Failed: 0
```

### Option C: Run Multiple Tasks (5 tasks specified)

```bash
cd integration

# Define tasks
export TASKS="1-aws-s3-snapshots 2-localstack-s3-snapshots 3-localstack-s3-snapshots 4-localstack-notifications 5-localstack-s3-notifications"

# Run with Claude
apex-runner --model claude \
  --tasks $TASKS \
  --tasks-dir ../tasks \
  --parallel --max-workers 3
```

## Step 7: Run All 8 Models

### Sequential Execution

```bash
cd integration

export TASKS="1-aws-s3-snapshots 2-localstack-s3-snapshots 3-localstack-s3-snapshots 4-localstack-notifications 5-localstack-s3-notifications"

# Run each model sequentially
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
```

### Parallel Execution (Advanced)

```bash
cd integration

export TASKS="1-aws-s3-snapshots 2-localstack-s3-snapshots 3-localstack-s3-snapshots 4-localstack-notifications 5-localstack-s3-notifications"

# Run all models in parallel (use with caution - resource intensive)
for model in claude opus gemini xai deepseek qwen codex kimi; do
  apex-runner --model $model \
    --tasks $TASKS \
    --tasks-dir ../tasks \
    --parallel --max-workers 3 &
done

# Wait for all to complete
wait

echo "All models completed!"
```

## Step 8: Check Results

```bash
cd integration

# View generated artifacts
ls -lh artifacts/

# Expected files:
# claude_tasks_status_TIMESTAMP.csv
# opus_tasks_status_TIMESTAMP.csv
# gemini_tasks_status_TIMESTAMP.csv
# xai_tasks_status_TIMESTAMP.csv
# deepseek_tasks_status_TIMESTAMP.csv
# qwen_tasks_status_TIMESTAMP.csv
# codex_tasks_status_TIMESTAMP.csv
# kimi_tasks_status_TIMESTAMP.csv

# View a specific CSV
cat artifacts/claude_tasks_status_*.csv

# Count successful tasks
grep ",completed," artifacts/*.csv | wc -l

# Count failed tasks
grep ",failed," artifacts/*.csv | wc -l
```

## Complete Test Script

Save this as `test_all_models.sh`:

```bash
#!/bin/bash

# Exit on error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "APEX SWE Harness - EC2 Test Script"
echo "========================================"
echo ""

# Check virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${RED}ERROR: Virtual environment not activated${NC}"
    echo "Run: source venv/bin/activate"
    exit 1
fi

# Check API keys
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo -e "${YELLOW}WARNING: ANTHROPIC_API_KEY not set (needed for claude, opus)${NC}"
fi

# Navigate to integration directory
cd integration

# Define tasks
export TASKS="1-aws-s3-snapshots 2-localstack-s3-snapshots 3-localstack-s3-snapshots 4-localstack-notifications 5-localstack-s3-notifications"

# Models to test
MODELS=("claude" "opus" "gemini" "xai" "deepseek" "qwen" "codex" "kimi")

# Track results
declare -A results

# Run each model
for model in "${MODELS[@]}"; do
    echo ""
    echo "========================================"
    echo -e "Testing model: ${GREEN}$model${NC}"
    echo "========================================"
    
    if apex-runner --model $model \
        --tasks $TASKS \
        --tasks-dir ../tasks \
        --parallel --max-workers 3; then
        results[$model]="SUCCESS"
        echo -e "${GREEN}✓ $model completed successfully${NC}"
    else
        results[$model]="FAILED"
        echo -e "${RED}✗ $model failed${NC}"
    fi
    
    echo ""
done

# Print summary
echo ""
echo "========================================"
echo "FINAL SUMMARY"
echo "========================================"

for model in "${MODELS[@]}"; do
    status="${results[$model]}"
    if [ "$status" == "SUCCESS" ]; then
        echo -e "${GREEN}✓${NC} $model: $status"
    else
        echo -e "${RED}✗${NC} $model: $status"
    fi
done

echo ""
echo "Results saved to: integration/artifacts/"
ls -lh artifacts/*.csv

echo ""
echo "========================================"
echo "Testing complete!"
echo "========================================"
```

Make it executable and run:

```bash
chmod +x test_all_models.sh
./test_all_models.sh
```

## Troubleshooting

### Issue: `apx: command not found`

```bash
# Check installation
pip list | grep apex-code

# Reinstall
cd apex_code
pip install -e .

# Check PATH
echo $PATH
which apx

# Run directly if needed
python -m apex_code.cli --help
```

### Issue: `apex-runner: command not found`

```bash
# Check installation
pip list | grep apex-swe

# Reinstall
cd integration
pip install -e .

# Check PATH
which apex-runner

# Run directly if needed
python -m apex_harness.cli --help
```

### Issue: Docker permission denied

```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Reload group membership
newgrp docker

# Verify
docker ps
```

### Issue: Out of disk space

```bash
# Check disk space
df -h

# Clean Docker
docker system prune -a -f

# Clean Python cache
find . -type d -name __pycache__ -exec rm -r {} +
find . -type f -name '*.pyc' -delete
```

### Issue: API rate limits

```bash
# Add delays between models
for model in claude opus gemini; do
  apex-runner --model $model --tasks $TASKS --tasks-dir ../tasks
  sleep 60  # Wait 60 seconds between models
done
```

## Performance Monitoring

### Monitor Resource Usage

```bash
# In separate terminal
watch -n 1 'echo "=== CPU ===" && top -bn1 | head -n 20 && echo "" && echo "=== Memory ===" && free -h && echo "" && echo "=== Docker ===" && docker ps'
```

### Monitor Progress

```bash
# Watch artifacts directory
watch -n 5 'ls -lh integration/artifacts/ && echo "" && tail -n 10 integration/artifacts/*.csv'
```

## Cleanup

```bash
# Remove artifacts
rm -rf integration/artifacts/*.csv

# Clean Docker
docker system prune -a -f

# Deactivate virtual environment
deactivate
```

## Expected Timeline

For 5 tasks × 8 models = 40 evaluations:

- **Sequential**: ~2-4 hours (depending on model speed)
- **Parallel (3 workers)**: ~1-2 hours
- **Parallel (all models at once)**: ~30-60 minutes (high resource usage)

## Success Criteria

✅ All commands execute without errors
✅ CSV files generated for each model
✅ Tasks show "completed" status
✅ Return codes are 0 for successful tasks
✅ Docker containers clean up properly

## Next Steps

1. Analyze results in `integration/artifacts/`
2. Compare model performance
3. Generate reports
4. Share findings

---

For more information, see:
- [INSTALL.md](../INSTALL.md) - Installation guide
- [README.md](../README.md) - Main documentation
- [integration/README.md](../integration/README.md) - Integration runner docs
