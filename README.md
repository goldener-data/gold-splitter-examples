# gold-split-examples

Different examples using Goldener GoldSplitter during training Machine Learning models.

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management. Install dependencies using:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync project dependencies (creates/updates virtual environment)
uv sync

# Or sync with development dependencies
uv sync --extra dev
```

## Development Setup

This project uses pre-commit hooks with ruff and mypy for code quality:

```bash
# Install dev dependencies
uv sync --extra dev

# Install pre-commit hooks
uv run pre-commit install

# Run hooks manually on all files
uv run pre-commit run --all-files
```

The pre-commit hooks will automatically run:
- **ruff**: Linting and formatting
- **mypy**: Type checking
- Additional checks: trailing whitespace, YAML validation, etc.

## Examples

### 1. Image Classification: CIFAR-10 with CNN

**Location**: [`image_classification_cifar10_cnn/`](image_classification_cifar10_cnn/)

This experiment compares two data splitting strategies for training a CNN on CIFAR-10:
- **Random Split**: Traditional random split using scikit-learn
- **Smart Split**: Intelligent split using GoldSplitter from the Goldener library

**Key Features**:
- Simple CNN model (3 conv layers + 2 FC layers)
- PyTorch Lightning for training
- Hydra for configuration management
- MLFlow for experiment tracking
- AUROC-based model selection

**Quick Start**:
```bash
# Install dependencies (from repo root)
uv sync --extra vision

# Run experiment
cd image_classification_cifar10_cnn
uv run python cifar10_experiment.py
```

See the [detailed README](image_classification_cifar10_cnn/README.md) for more information.

## About Goldener

Goldener provides intelligent data splitting strategies that aim to create more balanced and representative train/validation splits compared to traditional random splitting.
