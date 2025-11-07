# CIFAR-10 Split Comparison Experiment

This experiment compares two data splitting strategies for training a CNN on CIFAR-10:

1. **Random Split**: Traditional random split using scikit-learn
2. **Smart Split**: Intelligent split using GoldSplitter from the Goldener library

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Technical Details](#technical-details)
- [Model Architecture](#model-architecture)
- [Training Configuration](#training-configuration)
- [Split Strategies](#split-strategies)
- [Viewing Results](#viewing-results)
- [Output Interpretation](#output-interpretation)
- [Extending the Experiment](#extending-the-experiment)
- [References](#references)

## Overview

This experiment systematically compares two data splitting strategies for training machine learning models.
Data splitting is a critical step that can significantly impact model performance and generalization.
While random splitting is the standard approach, smart splitting strategies like GoldSplitter aim
to create more balanced and representative train/validation splits, potentially leading to:

- Better model generalization
- More reliable validation metrics
- Faster convergence
- More representative performance estimation

## Features

- **Dataset**: CIFAR-10 from torchvision (50,000 training images, 10,000 test images)
- **Model**: Simple convolutional neural network (CNN) with ~650K parameters
- **Training**: PyTorch Lightning for efficient training management
- **Configuration**: Hydra for flexible configuration management
- **Logging**: MLFlow for experiment tracking and comparison
- **Metric**: AUROC (Area Under ROC Curve) on validation set for model selection
- **Reproducibility**: Fixed random seeds for reliable results

All dependencies are defined in the root `pyproject.toml` file.

## Quick Start

```bash
# Install dependencies (from repo root)
uv sync --extra vision

# Make sure you're in the experiment directory
cd image_classification_cifar10_cnn

# Run both split methods (uses default config)
uv run python cifar10_experiment.py

# View results
mlflow ui
```

Then navigate to `http://localhost:5000` in your browser.

## Usage Examples

The experiment uses [Hydra](https://hydra.cc/) for configuration management. All parameters can be configured via the `conf/config.yaml` file or overridden via command line.

### Example 1: Run both split methods (default)
```bash
uv run python cifar10_experiment.py
```

### Example 2: Run with updated config
```bash
uv run python cifar10_experiment.py plit_method=gold max_epochs=100 batch_size=256 learning_rate=0.0001 experiment_name=my-cifar10-test data_dir=/path/to/data
```

### Hydra Configuration Tips

- **Default config file**: `config/config.yaml`
- **Override any parameter**: Use `key=value` syntax
- **View config**: `python cifar10_experiment.py --cfg job`
- **View help**: `python cifar10_experiment.py --help`


## Technical Details

### Dataset: CIFAR-10

- **Classes**: 10 object categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Training samples**: 50,000 images
- **Test samples**: 10,000 images
- **Image size**: 32×32 pixels, RGB color
- **Split ratio**: 80% training, 20% validation (from the 50,000 training images)

### Data Preprocessing

**Training Set**:
```python
transforms.RandomHorizontalFlip()
transforms.RandomCrop(32, padding=4)
transforms.ToTensor()
transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2470, 0.2435, 0.2616])
```

**Validation/Test Set**:
```python
transforms.ToTensor()
transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2470, 0.2435, 0.2616])
```

## Model Architecture

The CNN model consists of:

```
Input (3×32×32)
    ↓
Conv2D(3→32, 3×3) + ReLU + MaxPool(2×2)
    ↓
Conv2D(32→64, 3×3) + ReLU + MaxPool(2×2)
    ↓
Conv2D(64→128, 3×3) + ReLU + MaxPool(2×2)
    ↓
Flatten (128×4×4 = 2048)
    ↓
Linear(2048→256) + ReLU + Dropout(0.5)
    ↓
Linear(256→10)
    ↓
Output (10 classes)
```

**Key Details**:
- 3 convolutional layers (32, 64, 128 filters)
- MaxPooling after each convolutional layer
- 2 fully connected layers (256, 10 neurons)
- Dropout (p=0.5) for regularization
- ReLU activation functions
- Total trainable parameters: ~650K
- Output: Raw logits (no softmax, handled by loss function)

## Training Configuration

- **Optimizer**: Adam
  - Learning rate: 0.001 (default, configurable)
  - Weight decay: 0 (no L2 regularization)

- **Learning Rate Scheduler**: ReduceLROnPlateau
  - Monitor: Validation AUROC
  - Mode: Maximize
  - Factor: 0.5
  - Patience: 5 epochs

- **Loss Function**: CrossEntropyLoss
  - Combines LogSoftmax and NLLLoss
  - Applied to raw logits

- **Batch Size**: 128 (default, configurable)
- **Max Epochs**: 50 (default, configurable)

### Evaluation Metrics

**Primary Metric (for model selection)**:
- **AUROC (Area Under ROC Curve)**: Measures the model's ability to distinguish between classes
  - Task: Multiclass classification
  - Used for: Model checkpoint selection (best validation AUROC)
  - Advantage: More robust than accuracy for imbalanced datasets

**Secondary Metrics**:
- **Accuracy**: Percentage of correct predictions
- **Loss**: Cross-entropy loss value

All metrics are computed and logged for both training and validation sets at each epoch.

### Model Selection

The best model is selected based on **maximum validation AUROC**:

```python
checkpoint_callback = ModelCheckpoint(
    monitor='val_auroc',
    mode='max',
    save_top_k=1
)
```

## Split Strategies

### 1. Random Split (Baseline)

```python
from sklearn.model_selection import train_test_split

train_indices, val_indices = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    shuffle=True
)
```

**Characteristics**:
- Uniform probability for each sample
- No consideration of class distribution
- Standard practice in ML
- Simple and fast

### 2. GoldSplitter (Smart Split)

```python
from image_classification_cifar10_cnn.utils import get_gold_splitter

gold_splitter = get_gold_splitter(cfg)
splits = gold_splitter.split(
    dataset=dataset,
)
train_indices = splits['train']
val_indices = splits['val']
```

**Characteristics**:
- Considers class labels for balanced splits
- Aims for optimal distribution
- May lead to more representative validation sets
- Potentially better generalization

### Expected Outcomes

GoldSplitter may provide:
1. More balanced class distribution in validation set
2. Better representation of minority classes
3. More stable validation metrics across epochs
4. Potentially higher final AUROC scores

### Evaluation Criteria

Compare the two methods on:
- **Convergence Speed**: Epochs to reach best performance
- **Stability**: Variance in validation metrics across epochs
- **Test Performance**: Final performance on held-out test set

## Viewing Results

### MLFlow UI

After running the experiment, start the MLFlow UI:
```bash
mlflow ui
```

Then open `http://localhost:5000` in your browser to compare results between split methods.

### MLFlow Tracking

All experiments are logged to MLFlow with the following information:

**Parameters**:
- Split method (random/gold)
- Batch size
- Learning rate
- Random state
- Model architecture details

**Metrics** (per epoch):
- Training loss, accuracy, AUROC
- Validation loss, accuracy, AUROC
- Learning rate

**Artifacts**:
- Best model checkpoint
- Model architecture summary
- Hyperparameters

**Tags**:
- Experiment name
- Run name (includes split method)

## Output Interpretation

After running the experiment, you will see:

1. **Console Output**: Training progress, metrics per epoch
2. **MLFlow UI**: Detailed comparison of both methods
3. **Checkpoints**: Best models saved in `./checkpoints/`
4. **Summary**: Final comparison of AUROC scores

### Example Summary Output

```
EXPERIMENT SUMMARY
============================================================

RANDOM Split Method:
  Best Validation AUROC: 0.8542

GOLD Split Method:
  Best Validation AUROC: 0.8687

AUROC Difference (Gold - Random): +0.0145
============================================================
```

## References

- **PyTorch**: https://pytorch.org/
- **PyTorch Lightning**: https://lightning.ai/
- **MLFlow**: https://mlflow.org/
- **Hydra**: https://hydra.cc/
- **Goldener**: Python library for smart data splitting
- **CIFAR-10**: https://www.cs.toronto.edu/~kriz/cifar.html

## Files

- `cifar10_experiment.py`: Main experiment script
- `config/config.yaml`: Hydra configuration file
- `README.md`: This file - complete documentation and user guide

**Note**: Dependencies are managed at the repository root level in `pyproject.toml`.
