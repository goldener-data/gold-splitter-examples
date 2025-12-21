import timm
from lightning import LightningModule
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRSchedulerConfig
from timm.models import VisionTransformer, Eva
from torch import nn as nn
from torch.nn import functional as F
from torchmetrics.classification import MulticlassAUROC


class DinoLinearProbing(LightningModule):
    def __init__(
        self,
        num_classes: int = 10,
        learning_rate: float = 0.001,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        self.vit = timm.create_model(
            "vit_small_patch16_dinov3.lvd1689m",
            pretrained=True,
            img_size=224,
        )

        assert isinstance(self.vit, VisionTransformer) or isinstance(self.vit, Eva)

        self.head: nn.Module
        if hidden_dims is None or len(hidden_dims) == 0:
            self.head = nn.Linear(self.vit.patch_embed.proj.out_channels, num_classes)
        else:
            layers: list[nn.Module] = []
            input_dim = self.vit.patch_embed.proj.out_channels
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.ReLU())
                input_dim = hidden_dim
            layers.append(nn.Linear(input_dim, num_classes))
            self.head = nn.Sequential(*layers)

        self.train_auroc = MulticlassAUROC(num_classes=num_classes)
        self.val_auroc = MulticlassAUROC(num_classes=num_classes)
        self.test_auroc = MulticlassAUROC(num_classes=num_classes)

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(self.vit, VisionTransformer) or isinstance(self.vit, Eva)
        x = self.vit.forward_features(x)
        return self.head(x[:, 0, :])

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
