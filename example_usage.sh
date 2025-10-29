#!/bin/bash
# Example usage script for CIFAR-10 Split Comparison Experiment

echo "=================================================="
echo "CIFAR-10 Split Comparison Experiment Examples"
echo "=================================================="
echo ""

echo "Example 1: Run both split methods (default)"
echo "$ python cifar10_experiment.py"
echo ""

echo "Example 2: Run only random split"
echo "$ python cifar10_experiment.py --split-method random"
echo ""

echo "Example 3: Run only GoldSplitter split"
echo "$ python cifar10_experiment.py --split-method gold"
echo ""

echo "Example 4: Quick test with fewer epochs"
echo "$ python cifar10_experiment.py --max-epochs 5"
echo ""

echo "Example 5: Full training with custom parameters"
echo "$ python cifar10_experiment.py --max-epochs 100 --batch-size 256 --learning-rate 0.0001"
echo ""

echo "Example 6: Custom experiment name and data directory"
echo "$ python cifar10_experiment.py --experiment-name my-cifar10-test --data-dir /path/to/data"
echo ""

echo "=================================================="
echo "View Results with MLFlow"
echo "=================================================="
echo ""
echo "After running the experiment, start the MLFlow UI:"
echo "$ mlflow ui"
echo ""
echo "Then open http://localhost:5000 in your browser"
echo ""

echo "=================================================="
echo "For help and all available options:"
echo "$ python cifar10_experiment.py --help"
echo "=================================================="
