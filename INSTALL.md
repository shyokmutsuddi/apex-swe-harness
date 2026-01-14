# Complete Installation Guide

This guide provides detailed installation instructions for the APEX SWE Harness framework.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Installation](#quick-installation)
3. [Detailed Installation Steps](#detailed-installation-steps)
4. [Environment Configuration](#environment-configuration)
5. [Verification](#verification)
6. [Troubleshooting](#troubleshooting)
7. [Uninstallation](#uninstallation)

## Prerequisites

### Required Software

- **Python 3.10 or higher**
  ```bash
  python --version  # Should be 3.10+
  ```

- **Docker** (for running tasks)
  ```bash
  docker --version
  docker ps  # Verify Docker is running
  ```

- **Git**
  ```bash
  git --version
  ```

### System Requirements

- **OS**: Linux, macOS, or Windows (with WSL2 for best Docker support)
- **RAM**: 8GB minimum (16GB recommended)
- **Disk**: 10GB free space
- **Internet**: Required for downloading dependencies and models

## Quick Installation

### One-Command Setup

```bash
# Clone repository
git clone https://github.com/your-org/apex-swe-harness.git
cd apex-swe-harness

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1

# Install both components
pip install --upgrade pip
pip install -e apex_code
pip install -e integration

# Verify installation
apx --help
apex-runner --help

echo "✓ Installation complete!"
```

## Detailed Installation Steps

### Step 1: Clone Repository

```bash
# Clone the repository
git clone https://github.com/your-org/apex-swe-harness.git
cd apex-swe-harness

# Or if you have SSH access
git clone git@github.com:your-org/apex-swe-harness.git
cd apex-swe-harness
```

### Step 2: Create Virtual Environment

**Linux/macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
python -m venv venv
.\venv\Scripts\activate.bat
```

**Verify activation:**
```bash
which python  # Should point to venv/bin/python or venv\Scripts\python.exe
```

### Step 3: Upgrade pip

```bash
pip install --upgrade pip
```

### Step 4: Install APEX Code Harness

```bash
# Navigate to apex_code directory
cd apex_code

# Install with dependencies
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"

# Verify installation
apx --help

# Go back to root
cd ..
```

**Expected output:**
```
Usage: apx [OPTIONS] COMMAND [ARGS]...

Commands:
  datasets  Manage datasets.
  reports   Run experiments and generate reports.
  run       Run the apex-code harness
  runs      Manage runs.
  tasks     Manage tasks.
```

### Step 5: Install Integration Test Runner

```bash
# Navigate to integration directory
cd integration

# Install with dependencies
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"

# Verify installation
apex-runner --help

# Go back to root
cd ..
```

**Expected output:**
```
Usage: apex-runner [OPTIONS]

Options:
  --model TEXT        AI model to use for evaluation  [required]
  --parallel          Run tasks in parallel
  --max-workers INT   Maximum number of parallel workers
  --tasks TEXT        Specific tasks to run
  ...
```

### Step 6: Install Additional Dependencies (Optional)

For development and testing:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Install code quality tools
pip install black ruff mypy

# Install all development tools
cd apex_code && pip install -e ".[dev]" && cd ..
cd integration && pip install -e ".[dev]" && cd ..
```

## Environment Configuration

### Set API Keys

Different models require different API keys:

```bash
# Anthropic (Claude, Opus)
export ANTHROPIC_API_KEY='your-anthropic-key'

# Google (Gemini)
export GOOGLE_API_KEY='your-google-key'

# XAI (Grok)
export XAI_API_KEY='your-xai-key'

# Fireworks (DeepSeek, Qwen)
export FIREWORKS_API_KEY='your-fireworks-key'

# OpenAI (Codex)
export OPENAI_API_KEY='your-openai-key'

# Moonshot (Kimi)
export MOONSHOT_API_KEY='your-moonshot-key'
```

### Persistent Environment Variables

**Linux/macOS** - Add to `~/.bashrc` or `~/.zshrc`:

```bash
echo 'export ANTHROPIC_API_KEY="your-key"' >> ~/.bashrc
echo 'export FIREWORKS_API_KEY="your-key"' >> ~/.bashrc
source ~/.bashrc
```

**Windows** - Set system environment variables:

```powershell
# PowerShell (requires admin)
[System.Environment]::SetEnvironmentVariable('ANTHROPIC_API_KEY', 'your-key', 'User')
[System.Environment]::SetEnvironmentVariable('FIREWORKS_API_KEY', 'your-key', 'User')
```

### Docker Configuration

Ensure Docker is running and accessible:

```bash
# Test Docker
docker run hello-world

# If permission issues on Linux
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker ps
```

## Verification

### Verify Installation

```bash
# 1. Check Python packages
pip list | grep apex

# Expected output:
# apex-code            1.0.0
# apex-swe-harness     1.0.0

# 2. Check commands
which apx
which apex-runner

# 3. Test APEX harness
apx --help
apx tasks list

# 4. Test integration runner
apex-runner --help

# 5. Run validation tests
cd integration
python scripts/validate_refactoring.py

# Expected output:
# ======================================================================
# *** ALL VALIDATIONS PASSED ***
# ======================================================================

# 6. Run unit tests
pytest tests/ -v

# Expected output:
# ============================= 40 passed in 0.XX s ==============================
```

### Test Run (Dry Run)

```bash
# Test without executing
apex-runner --model claude \
  --tasks 1-aws-s3-snapshots \
  --dry-run

# Should show the command that would be executed
```

### Full Test Run (Optional)

```bash
# Run a single task with one model
apex-runner --model claude \
  --tasks 1-aws-s3-snapshots \
  --max-workers 1

# Check results
ls -la integration/artifacts/
cat integration/artifacts/claude_tasks_status_*.csv
```

## Troubleshooting

### Issue: `apx: command not found`

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # or .\venv\Scripts\Activate.ps1

# Reinstall APEX harness
cd apex_code
pip install -e .

# Check installation
pip list | grep apex-code

# Try running directly
python -m apex_code.cli --help
```

### Issue: `apex-runner: command not found`

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall integration runner
cd integration
pip install -e .

# Check installation
pip list | grep apex-swe

# Try running directly
python -m apex_harness.cli --help
```

### Issue: Import Errors

**Solution:**
```bash
# Ensure both packages are installed
cd apex_code && pip install -e . && cd ..
cd integration && pip install -e . && cd ..

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Verify imports work
python -c "import apex_code; print(apex_code.__file__)"
python -c "import apex_harness; print(apex_harness.__version__)"
```

### Issue: Docker Permission Denied

**Solution (Linux):**
```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Reload group membership
newgrp docker

# Verify
docker ps
```

**Solution (Windows):**
```
1. Ensure Docker Desktop is running
2. Check Docker Desktop settings
3. Restart Docker Desktop
```

### Issue: Missing Dependencies

**Solution:**
```bash
# Reinstall with all dependencies
cd apex_code
pip install -e ".[dev]"

cd ../integration
pip install -e ".[dev]"

# If specific package is missing
pip install package-name
```

### Issue: Tests Failing

**Solution:**
```bash
# Clean and reinstall
pip uninstall apex-code apex-swe-harness -y
cd apex_code && pip install -e . && cd ..
cd integration && pip install -e . && cd ..

# Run validation
cd integration
python scripts/validate_refactoring.py

# Run tests with verbose output
pytest tests/ -v --tb=short
```

## Uninstallation

### Complete Uninstallation

```bash
# Deactivate virtual environment
deactivate

# Remove packages
pip uninstall apex-code apex-swe-harness -y

# Remove virtual environment
rm -rf venv/  # On Windows: Remove-Item -Recurse -Force venv

# Remove repository (optional)
cd ..
rm -rf apex-swe-harness/
```

### Reinstallation

After uninstallation, follow the [Quick Installation](#quick-installation) steps again.

## Post-Installation

### Next Steps

1. **Read Documentation**: See [README.md](README.md) for usage examples
2. **Explore Tasks**: Check [tasks/](tasks/) directory
3. **Run Tests**: Execute `pytest integration/tests/`
4. **Try Examples**: See [integration/examples/](integration/examples/)

### Getting Help

- **Documentation**: [integration/docs/](integration/docs/)
- **Issues**: [GitHub Issues](https://github.com/your-org/apex-swe-harness/issues)
- **Community**: [GitHub Discussions](https://github.com/your-org/apex-swe-harness/discussions)

## Development Setup

For contributors, additional setup:

```bash
# Install development tools
pip install black ruff mypy pytest pytest-cov

# Install pre-commit hooks (if available)
pre-commit install

# Verify development setup
black --check apex_code/ integration/src/
ruff check apex_code/ integration/src/
mypy apex_code/ integration/src/
pytest integration/tests/ -v
```

---

**Installation complete!** 🎉 You're ready to evaluate AI models on software engineering tasks.

For usage instructions, see [README.md](README.md) or [integration/README.md](integration/README.md).
