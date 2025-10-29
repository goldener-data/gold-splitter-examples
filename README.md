# gold-split-examples
Different examples using Goldener GoldSplitter during training Machine Learning models

## CIFAR-10 Split Comparison Experiment

This experiment compares two data splitting strategies for training a CNN on CIFAR-10:

1. **Random Split**: Traditional random split using scikit-learn
2. **Smart Split**: Intelligent split using GoldSplitter from the Goldener library

### Features

- **Dataset**: CIFAR-10 from torchvision
- **Model**: Simple convolutional neural network (CNN)
- **Training**: PyTorch Lightning for efficient training management
- **Logging**: MLFlow for experiment tracking and comparison
- **Metric**: AUROC (Area Under ROC Curve) on validation set for model selection

### Installation

```bash
pip install -r requirements.txt
```

### Usage

#### Run both split methods for comparison:
```bash
python cifar10_experiment.py
```

#### Run only random split:
```bash
python cifar10_experiment.py --split-method random
```

#### Run only GoldSplitter split:
```bash
python cifar10_experiment.py --split-method gold
```

#### Customize training parameters:
```bash
python cifar10_experiment.py \
    --max-epochs 100 \
    --batch-size 256 \
    --learning-rate 0.0001 \
    --data-dir ./data \
    --mlflow-tracking-uri ./mlruns \
    --experiment-name my-experiment
```

### Available Arguments

- `--split-method`: Split method to use (`random`, `gold`, or `both`) [default: `both`]
- `--max-epochs`: Maximum number of training epochs [default: `50`]
- `--batch-size`: Batch size for training [default: `128`]
- `--learning-rate`: Learning rate for optimizer [default: `0.001`]
- `--data-dir`: Directory to store CIFAR-10 data [default: `./data`]
- `--mlflow-tracking-uri`: MLFlow tracking URI [default: `./mlruns`]
- `--experiment-name`: MLFlow experiment name [default: `cifar10-split-comparison`]
- `--random-state`: Random seed for reproducibility [default: `42`]

### Viewing Results

After running the experiment, you can view the results in MLFlow:

```bash
mlflow ui
```

Then navigate to `http://localhost:5000` in your browser to compare the results of different split methods.

### Model Architecture

The CNN model consists of:
- 3 convolutional layers (32, 64, 128 filters)
- MaxPooling after each convolutional layer
- 2 fully connected layers (256, 10 neurons)
- Dropout for regularization
- ReLU activation functions

### Training Details

- **Optimizer**: Adam with learning rate scheduling
- **Scheduler**: ReduceLROnPlateau (monitors validation AUROC)
- **Data Augmentation**: Random horizontal flip and random crop for training
- **Normalization**: Standard CIFAR-10 normalization
- **Best Model**: Selected based on highest validation AUROC

### Output

The experiment will:
1. Download CIFAR-10 dataset (if not already present)
2. Train models with specified split method(s)
3. Log metrics to MLFlow
4. Save best model checkpoints to `./checkpoints`
5. Print summary of results including best validation AUROC for each method
