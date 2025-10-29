"""
CIFAR-10 Split Comparison Experiment

This script compares two data splitting strategies:
1. Random split from scikit-learn
2. Smart split using GoldSplitter from the Goldener library

The experiment uses:
- CIFAR-10 dataset from torchvision
- Simple convolutional neural network
- PyTorch Lightning for training management
- MLFlow for experiment tracking
- AUROC metric on validation set for model selection
"""

import os
from typing import Tuple

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchmetrics import AUROC
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from sklearn.model_selection import train_test_split
import numpy as np


class SimpleCNN(pl.LightningModule):
    """
    Simple Convolutional Neural Network for CIFAR-10 classification
    """
    
    def __init__(self, num_classes: int = 10, learning_rate: float = 0.001):
        super(SimpleCNN, self).__init__()
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Metrics
        self.train_auroc = AUROC(task="multiclass", num_classes=num_classes)
        self.val_auroc = AUROC(task="multiclass", num_classes=num_classes)
        
        self.save_hyperparameters()
    
    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.conv1(x)))
        # Conv block 2
        x = self.pool(F.relu(self.conv2(x)))
        # Conv block 3
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(-1, 128 * 4 * 4)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Log metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        
        # Update AUROC
        probs = F.softmax(logits, dim=1)
        self.train_auroc.update(probs, y)
        
        return loss
    
    def on_train_epoch_end(self):
        # Compute and log AUROC
        auroc = self.train_auroc.compute()
        self.log('train_auroc', auroc, prog_bar=True)
        self.train_auroc.reset()
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        
        # Update AUROC
        probs = F.softmax(logits, dim=1)
        self.val_auroc.update(probs, y)
        
        return loss
    
    def on_validation_epoch_end(self):
        # Compute and log AUROC
        auroc = self.val_auroc.compute()
        self.log('val_auroc', auroc, prog_bar=True)
        self.val_auroc.reset()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_auroc',
                'interval': 'epoch',
                'frequency': 1
            }
        }


class CIFAR10DataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for CIFAR-10
    """
    
    def __init__(
        self,
        data_dir: str = './data',
        batch_size: int = 128,
        num_workers: int = 4,
        split_method: str = 'random',
        val_size: float = 0.2,
        random_state: int = 42
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_method = split_method
        self.val_size = val_size
        self.random_state = random_state
        
        # Define transforms
        self.transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
    
    def prepare_data(self):
        # Download CIFAR-10
        torchvision.datasets.CIFAR10(root=self.data_dir, train=True, download=True)
        torchvision.datasets.CIFAR10(root=self.data_dir, train=False, download=True)
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            # Load full training set
            full_train_dataset = torchvision.datasets.CIFAR10(
                root=self.data_dir,
                train=True,
                transform=self.transform_train,
                download=False
            )
            
            # Get indices for train/val split
            train_indices, val_indices = self._split_data(full_train_dataset)
            
            # Create train and validation subsets
            self.train_dataset = Subset(full_train_dataset, train_indices)
            
            # For validation, we use the test transform (no augmentation)
            val_dataset_full = torchvision.datasets.CIFAR10(
                root=self.data_dir,
                train=True,
                transform=self.transform_test,
                download=False
            )
            self.val_dataset = Subset(val_dataset_full, val_indices)
        
        if stage == 'test' or stage is None:
            self.test_dataset = torchvision.datasets.CIFAR10(
                root=self.data_dir,
                train=False,
                transform=self.transform_test,
                download=False
            )
    
    def _split_data(self, dataset) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split dataset into train and validation indices
        """
        indices = np.arange(len(dataset))
        
        if self.split_method == 'random':
            # Random split using scikit-learn
            train_indices, val_indices = train_test_split(
                indices,
                test_size=self.val_size,
                random_state=self.random_state,
                shuffle=True
            )
        elif self.split_method == 'gold':
            # Smart split using GoldSplitter
            try:
                from goldener import GoldSplitter
                
                # Extract labels for the full training dataset
                labels = np.array([dataset[i][1] for i in indices])
                
                # Initialize GoldSplitter
                splitter = GoldSplitter(
                    split_ratio=1 - self.val_size,
                    random_state=self.random_state
                )
                
                # Perform smart split
                train_indices, val_indices = splitter.split(indices, labels)
                
            except ImportError:
                print("Warning: Goldener not installed. Falling back to random split.")
                train_indices, val_indices = train_test_split(
                    indices,
                    test_size=self.val_size,
                    random_state=self.random_state,
                    shuffle=True
                )
        else:
            raise ValueError(f"Unknown split method: {self.split_method}")
        
        return train_indices, val_indices
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True
        )


def run_experiment(
    split_method: str = 'random',
    max_epochs: int = 50,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    data_dir: str = './data',
    mlflow_tracking_uri: str = './mlruns',
    experiment_name: str = 'cifar10-split-comparison',
    random_state: int = 42
):
    """
    Run a single experiment with the specified split method
    """
    # Set random seeds for reproducibility
    pl.seed_everything(random_state)
    
    # Initialize data module
    data_module = CIFAR10DataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=4,
        split_method=split_method,
        val_size=0.2,
        random_state=random_state
    )
    
    # Initialize model
    model = SimpleCNN(num_classes=10, learning_rate=learning_rate)
    
    # Setup MLFlow logger
    mlflow_logger = MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=mlflow_tracking_uri,
        run_name=f"{split_method}_split"
    )
    
    # Log additional parameters
    mlflow_logger.log_hyperparams({
        'split_method': split_method,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'random_state': random_state
    })
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_auroc',
        dirpath=f'./checkpoints/{split_method}_split',
        filename='cifar10-{epoch:02d}-{val_auroc:.4f}',
        save_top_k=1,
        mode='max',
        verbose=True
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='auto',
        devices=1,
        logger=mlflow_logger,
        callbacks=[checkpoint_callback],
        deterministic=True,
        log_every_n_steps=50
    )
    
    # Train the model
    print(f"\n{'='*60}")
    print(f"Training with {split_method.upper()} split method")
    print(f"{'='*60}\n")
    
    trainer.fit(model, data_module)
    
    # Test the model
    test_results = trainer.test(model, data_module, ckpt_path='best')
    
    print(f"\n{'='*60}")
    print(f"Training completed for {split_method.upper()} split method")
    print(f"Best validation AUROC: {checkpoint_callback.best_model_score:.4f}")
    print(f"{'='*60}\n")
    
    return {
        'split_method': split_method,
        'best_val_auroc': checkpoint_callback.best_model_score.item(),
        'test_results': test_results
    }


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """
    Main function to run experiments with different split strategies
    Uses Hydra for configuration management
    """
    # Print configuration
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Create necessary directories
    os.makedirs(cfg.data_dir, exist_ok=True)
    os.makedirs('./checkpoints', exist_ok=True)
    
    results = []
    
    # Run experiments based on split method argument
    if cfg.split_method == 'both':
        split_methods = ['random', 'gold']
    else:
        split_methods = [cfg.split_method]
    
    for split_method in split_methods:
        result = run_experiment(
            split_method=split_method,
            max_epochs=cfg.max_epochs,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            data_dir=cfg.data_dir,
            mlflow_tracking_uri=cfg.mlflow_tracking_uri,
            experiment_name=cfg.experiment_name,
            random_state=cfg.random_state
        )
        results.append(result)
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    for result in results:
        print(f"\n{result['split_method'].upper()} Split Method:")
        print(f"  Best Validation AUROC: {result['best_val_auroc']:.4f}")
    
    if len(results) > 1:
        print(f"\nAUROC Difference (Gold - Random): "
              f"{results[1]['best_val_auroc'] - results[0]['best_val_auroc']:.4f}")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    main()
