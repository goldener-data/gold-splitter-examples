from lightning import LightningModule
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRSchedulerConfig
from torch.nn import (
    functional as F,
    Flatten,
    Sequential,
    Conv2d,
    BatchNorm2d,
    ReLU,
    MaxPool2d,
    AdaptiveAvgPool2d,
    Linear,
    Dropout,
)
from torchmetrics.classification import MulticlassAUROC


class Cifar10CNN(LightningModule):
    def __init__(
        self,
        learning_rate: float = 0.001,
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate

        self.model = Sequential(
            Conv2d(3, 32, kernel_size=3, padding=1),
            BatchNorm2d(32),
            ReLU(),
            MaxPool2d(2),  # 32x16x16 for 32x32 inputs
            Conv2d(32, 64, kernel_size=3, padding=1),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(2),  # 64x8x8
            Conv2d(64, 128, kernel_size=3, padding=1),
            BatchNorm2d(128),
            ReLU(),
            AdaptiveAvgPool2d(1),  # 128x1x1
            Flatten(),
            Dropout(0.2),
            Linear(128, 10),
        )

        self.save_hyperparameters()

    @property
    def has_test_as_val(self) -> bool:
        if not isinstance(self.trainer.val_dataloaders, dict):
            return False
        return "test_as_val" in self.trainer.val_dataloaders

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def on_train_start(self) -> None:
        self.train_auroc = MulticlassAUROC(num_classes=10)

    def _step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        auroc_metric: MulticlassAUROC,
        prefix: str,
    ) -> torch.Tensor:
        x, y, _ = batch

        # loss
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log(f"{prefix}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log(f"{prefix}_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # AUROC
        probs = F.softmax(logits, dim=1)
        auroc_metric.update(probs, y)

        return loss

    def _compute_auroc_and_log(
        self,
        auroc_metric: MulticlassAUROC,
        prefix: str,
    ) -> None:
        auroc = auroc_metric.compute()
        self.log(f"{prefix}_auroc", auroc, prog_bar=True)
        auroc_metric.reset()

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        return self._step(batch, self.train_auroc, "train")

    def on_train_epoch_end(self) -> None:
        self._compute_auroc_and_log(self.train_auroc, "train")

    def on_validation_epoch_start(self) -> None:
        self.val_auroc = MulticlassAUROC(num_classes=10)
        if self.has_test_as_val:
            self.test_as_val_auroc = MulticlassAUROC(num_classes=10)

    def validation_step(
        self,
        batch: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> STEP_OUTPUT:
        if isinstance(batch, dict):
            val_batch = batch["val"]
            test_batch = batch["test_as_val"] if "test_as_val" in batch else None
        else:
            val_batch = batch
            test_batch = None

        loss = None
        if val_batch is not None:
            loss = self._step(val_batch, self.val_auroc, "val")
        if test_batch is not None:
            self._step(test_batch, self.test_as_val_auroc, "test_as_val")

        return loss

    def on_validation_epoch_end(self) -> None:
        self._compute_auroc_and_log(self.val_auroc, "val")

        if self.has_test_as_val:
            self._compute_auroc_and_log(self.test_as_val_auroc, "test_as_val")

    def on_test_start(self) -> None:
        self.test_auroc = MulticlassAUROC(num_classes=10)

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        return self._step(batch, self.test_auroc, "test")

    def on_test_epoch_end(self) -> None:
        test_auroc = self.test_auroc.compute()
        self.log("test_auroc", test_auroc, prog_bar=True)
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
