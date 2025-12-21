# CIFAR-10 Split Comparison Experiment

This experiment compares the two data splitting strategies for training 
a Linear Probing on Dinov3 ViT-S on the CIFAR-10 dataset. 


## Table of Contents

- [Main components](#main-components)
- [Quick Start](#quick-start)
- [Technical Details](#technical-details)
- [Split Strategies](#split-strategies)
- [Viewing Results](#viewing-results)
- [Output Interpretation](#output-interpretation)
- [Extending the Experiment](#extending-the-experiment)
- [References](#references)


## Main components

- **Configuration**: The experiment is configured from a config file loaded from Hydra for flexible configuration management.
It allows to specify the hyperparameters and logging parameter for the model training/evaluation 
but as well as the data split method to use and the settings for the GoldSplitter.

- **CIFAR10DataModule**: A specific Pytorch Datamodule allowing to load data from the CIFAR-10 dataset from torchvision 
(50,000 training images, 10,000 test images). Depending on the configuration, only a subset of the 
training images is used for training/validation.

- **DinoLinearProbing**: A specific Pytorch Lightning LightningModule allowing to train and evaluate a Dinov3 ViT-S from Timm 
with a linear probing head leveraging the final class token to make image classification.

- **Trainer**: PyTorch Lightning Trainer for efficient training management allowing to handle training, validation and 
testing loops. It allows as well to checkpoint the best model based on validation AUROC metric.

- **Logging**: MLFlow for experiment tracking allowing to compare the different splitting strategies based on the logged metrics.

## Quick Start

```bash
# Install dependencies (from repo root)
uv sync --extra vision

# Make sure you're in the experiment directory
cd image_classification_cifar10

# Run both split methods (uses default config)
uv run python cifar10_experiment.py

# View results
mlflow ui
```

Then navigate to `http://localhost:5000` in your browser.


## Technical Details

### Dataset: CIFAR-10

- **Classes**: 10 object categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Training samples**: 50,000 images (5000 per class)
- **Test samples**: 10,000 images
- **Image size**: intially 32×32 pixels but resized to 224×224, RGB color

### Data Preprocessing

**Training Set with augmentation**:
```python
Compose(
  [
    RandomHorizontalFlip(),
    ToTensor(),
    Resize(224),
    Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
  ]
)
```

**Validation/Test Set**:
```python
Compose(
  [
    ToTensor(),
    Resize(224),
    Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
  ]
)
```

### Training Configuration

- **Optimizer**: Adam
  - Learning rate: 0.001 (default, configurable)
  - Weight decay: 0 (no L2 regularization)

- **Learning Rate Scheduler**: ReduceLROnPlateau
  - Monitor: Validation AUROC
  - Mode: Maximize
  - Factor: 0.5
  - Patience: 5 epochs

- **Loss Function**: CrossEntropyLoss
  - Applied to raw logits

- **Batch Size**: 128 (default, configurable)
- **Max Epochs**: 100 (default, configurable)

### Evaluation Metrics

**Primary Metric (for model selection)**:
- **AUROC (Area Under ROC Curve)**: Measures the model's ability to distinguish between classes
  - Task: Multiclass classification
  - Used for: Model checkpoint selection (best validation AUROC)

**Secondary Metrics**:
- **Accuracy**: Percentage of correct predictions

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
    shuffle=True,
    stratify=targets,
)
```

**Characteristics**:
- Uniform probability for each sample
- Selection for each class
- Standard practice in ML
- Simple and fast

### 2. GoldSplitter (Smart Split)

The smart split is done from the class token of the Dinov3 ViT-S model.

```python
from image_classification_cifar10.utils import get_gold_splitter

gold_splitter = get_gold_splitter(cfg.gold_splitter)
split_table = gold_splitter.split_in_table(dataset)
splits = gold_splitter.get_split_indices(
    split_table, selection_key="selected", idx_key="idx"
)
train_indices = np.array(list(splits["train"]))
val_indices = np.array(list(splits["val"]))
```

**Characteristics**:
- Considers class labels for balanced splits
- Aims for optimal distribution
- May lead to more representative validation sets
- Potentially better generalization


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

## Dependencies

Dependencies are managed at the repository root level in `pyproject.toml`.

- [Timm](https://huggingface.co/timm): Dinov3 ViT-S model
- [Torchvision](https://pytorch.org/vision/stable/index.html): CIFAR-10 dataset and transforms
- [Pillow](https://python-pillow.org/): Image laoding