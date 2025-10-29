#!/bin/bash
# Example usage script for CIFAR-10 Split Comparison Experiment
# Using Hydra for configuration management

# Example 1: Run both split methods (default)
# python cifar10_experiment.py

# Example 2: Run only random split
# python cifar10_experiment.py split_method=random

# Example 3: Run only GoldSplitter split
# python cifar10_experiment.py split_method=gold

# Example 4: Quick test with fewer epochs
# python cifar10_experiment.py max_epochs=5

# Example 5: Full training with custom parameters
# python cifar10_experiment.py max_epochs=100 batch_size=256 learning_rate=0.0001

# Example 6: Custom experiment name and data directory
# python cifar10_experiment.py experiment_name=my-cifar10-test data_dir=/path/to/data

# Example 7: Override multiple parameters
# python cifar10_experiment.py split_method=gold max_epochs=200 batch_size=64 random_state=123

# View Results with MLFlow
# After running the experiment, start the MLFlow UI:
# mlflow ui
# Then open http://localhost:5000 in your browser

# Hydra Configuration
# Default config file: conf/config.yaml
# Override any parameter: key=value
# View config: python cifar10_experiment.py --cfg job
# View help: python cifar10_experiment.py --help
