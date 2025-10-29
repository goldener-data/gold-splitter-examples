#!/bin/bash
# Example usage script for CIFAR-10 Split Comparison Experiment
# Using Hydra for configuration management

echo "=================================================="
echo "CIFAR-10 Split Comparison Experiment Examples"
echo "Uses Hydra for configuration management"
echo "=================================================="
echo ""

echo "Example 1: Run both split methods (default)"
echo "$ python cifar10_experiment.py"
echo ""

echo "Example 2: Run only random split"
echo "$ python cifar10_experiment.py split_method=random"
echo ""

echo "Example 3: Run only GoldSplitter split"
echo "$ python cifar10_experiment.py split_method=gold"
echo ""

echo "Example 4: Quick test with fewer epochs"
echo "$ python cifar10_experiment.py max_epochs=5"
echo ""

echo "Example 5: Full training with custom parameters"
echo "$ python cifar10_experiment.py max_epochs=100 batch_size=256 learning_rate=0.0001"
echo ""

echo "Example 6: Custom experiment name and data directory"
echo "$ python cifar10_experiment.py experiment_name=my-cifar10-test data_dir=/path/to/data"
echo ""

echo "Example 7: Override multiple parameters"
echo "$ python cifar10_experiment.py split_method=gold max_epochs=200 batch_size=64 random_state=123"
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
echo "Hydra Configuration"
echo "=================================================="
echo ""
echo "Default config file: conf/config.yaml"
echo "Override any parameter: key=value"
echo "View config: python cifar10_experiment.py --cfg job"
echo "View help: python cifar10_experiment.py --help"
echo "=================================================="
