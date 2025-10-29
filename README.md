# gold-split-examples

Different examples using Goldener GoldSplitter during training Machine Learning models.

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
cd image_classification_cifar10_cnn
pip install -r requirements.txt
python cifar10_experiment.py
```

See the [detailed README](image_classification_cifar10_cnn/README.md) for more information.

## Contributing

To add a new example:
1. Create a new directory with a descriptive name
2. Include a README.md with usage instructions
3. Add requirements.txt with dependencies
4. Update this main README with a link to your example

## About Goldener

Goldener provides intelligent data splitting strategies that aim to create more balanced and representative train/validation splits compared to traditional random splitting.
