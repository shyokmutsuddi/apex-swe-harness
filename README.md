# APEX SWE Harness

A comprehensive harness for software engineering tasks and AI model evaluation.

## Structure

This repository contains:

- **`integration/`** - Integration testing harness for AI model evaluation
  - Complete test harness with unified CLI
  - Support for multiple AI models (Claude, Gemini, XAI, etc.)
  - Parallel and sequential execution modes
  - See [integration/README.md](integration/README.md) for details

- **`temp/`** - Legacy scripts (read-only, for regression validation)
  - Original implementation preserved for comparison
  - Will be removed after migration is complete

- **`tasks/`** - Task definitions for integration tests
  - Shared task specifications
  - Docker-based test environments

## Quick Start

### Integration Testing

```bash
# Navigate to integration directory
cd integration

# Install the harness
pip install -e .

# Run integration tests
apex-runner --model claude --parallel
```

See [integration/README.md](integration/README.md) for complete documentation.

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
