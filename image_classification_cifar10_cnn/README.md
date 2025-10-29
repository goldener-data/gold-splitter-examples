# CIFAR-10 Split Comparison Experiment

This experiment compares two data splitting strategies for training a CNN on CIFAR-10:

1. **Random Split**: Traditional random split using scikit-learn
2. **Smart Split**: Intelligent split using GoldSplitter from the Goldener library

## Features

- **Dataset**: CIFAR-10 from torchvision
- **Model**: Simple convolutional neural network (CNN)
- **Training**: PyTorch Lightning for efficient training management
- **Configuration**: Hydra for flexible configuration management
- **Logging**: MLFlow for experiment tracking and comparison
- **Metric**: AUROC (Area Under ROC Curve) on validation set for model selection

## Installation

```bash
pip install -r requirements.txt
```

## Usage

The experiment uses [Hydra](https://hydra.cc/) for configuration management. All parameters can be configured via the `conf/config.yaml` file or overridden via command line.

### Run both split methods for comparison (default):
```bash
python cifar10_experiment.py
```

### Run only random split:
```bash
python cifar10_experiment.py split_method=random
```

### Run only GoldSplitter split:
```bash
python cifar10_experiment.py split_method=gold
```

### Customize training parameters:
```bash
python cifar10_experiment.py \
    max_epochs=100 \
    batch_size=256 \
    learning_rate=0.0001 \
    data_dir=./data \
    mlflow_tracking_uri=./mlruns \
    experiment_name=my-experiment
```

### Override multiple parameters:
```bash
python cifar10_experiment.py split_method=gold max_epochs=200 batch_size=64
```

## Configuration

The experiment configuration is located in `conf/config.yaml`. Default values:

```yaml
split_method: both        # Options: random, gold, or both
max_epochs: 50
batch_size: 128
learning_rate: 0.001
random_state: 42
data_dir: ./data
mlflow_tracking_uri: ./mlruns
experiment_name: cifar10-split-comparison
num_workers: 4
val_size: 0.2
```

You can modify the config file or override any parameter via command line using Hydra's syntax:
- `key=value` for simple overrides
- `key.nested=value` for nested configurations
- `+key=value` to add new parameters

## Viewing Results

After running the experiment, you can view the results in MLFlow:

```bash
mlflow ui
```

Then navigate to `http://localhost:5000` in your browser to compare the results of different split methods.

## Model Architecture

The CNN model consists of:
- 3 convolutional layers (32, 64, 128 filters)
- MaxPooling after each convolutional layer
- 2 fully connected layers (256, 10 neurons)
- Dropout for regularization
- ReLU activation functions

## Training Details

- **Optimizer**: Adam with learning rate scheduling
- **Scheduler**: ReduceLROnPlateau (monitors validation AUROC)
- **Data Augmentation**: Random horizontal flip and random crop for training
- **Normalization**: Standard CIFAR-10 normalization
- **Best Model**: Selected based on highest validation AUROC

## Output

The experiment will:
1. Download CIFAR-10 dataset (if not already present)
2. Train models with specified split method(s)
3. Log metrics to MLFlow
4. Save best model checkpoints to `./checkpoints`
5. Print summary of results including best validation AUROC for each method

## Files

- `cifar10_experiment.py`: Main experiment script
- `conf/config.yaml`: Hydra configuration file
- `requirements.txt`: Python dependencies
- `EXPERIMENT_DETAILS.md`: Technical documentation with architecture details and usage examples
- `README.md`: This file - user guide and quick start
