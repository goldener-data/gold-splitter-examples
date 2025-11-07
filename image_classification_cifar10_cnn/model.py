from lightning import LightningModule
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRSchedulerConfig
from torch import nn as nn
from torch.nn import functional as F
from torchmetrics.classification import MulticlassAUROC


class SimpleCNN(LightningModule):
    def __init__(self, num_classes: int = 10, learning_rate: float = 0.001) -> None:
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
        self.train_auroc = MulticlassAUROC(task="multiclass", num_classes=num_classes)
        self.val_auroc = MulticlassAUROC(task="multiclass", num_classes=num_classes)
        self.test_auroc = MulticlassAUROC(task="multiclass", num_classes=num_classes)

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        # Log metrics
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # Update AUROC
        probs = F.softmax(logits, dim=1)
        self.train_auroc.update(probs, y)

        return loss

    def on_train_epoch_end(self) -> None:
        auroc = self.train_auroc.compute()
        self.log("train_auroc", auroc, prog_bar=True)
        self.train_auroc.reset()

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # Update AUROC
        probs = F.softmax(logits, dim=1)
        self.val_auroc.update(probs, y)

        return loss

    def on_validation_epoch_end(self) -> None:
        auroc = self.val_auroc.compute()
        self.log("val_auroc", auroc, prog_bar=True)
        self.val_auroc.reset()

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        x, y = batch
        logits = self(x)

        probs = F.softmax(logits, dim=1)
        self.test_auroc.update(probs, y)

        return probs

    def on_test_epoch_end(self) -> None:
        auroc = self.test_auroc.compute()
        self.log("test_auroc", auroc, prog_bar=True)
        self.test_auroc.reset()

    def configure_optimizers(
        self,
    ) -> OptimizerLRSchedulerConfig:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_auroc",
                "interval": "epoch",
                "frequency": 1,
            },
        }
