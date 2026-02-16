from lightning import LightningModule
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch.nn import (
    functional as F,
    Sequential,
    Conv2d,
    BatchNorm2d,
    ReLU,
    MaxPool2d,
    Linear,
    Dropout,
    ConvTranspose2d,
    Upsample,
)
from torchmetrics.classification import MulticlassJaccardIndex
import timm
from timm.models.eva import Eva


class VOCSegmentationLightningModule(LightningModule):
    def __init__(
        self,
        learning_rate: float = 0.001,
        model_type: str = "unet",
        num_classes: int = 21,  # Pascal VOC has 21 classes (20 objects + background)
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        self.model: torch.nn.Module
        self._setup_model(model_type)

        self.save_hyperparameters()

    def _setup_model(self, model_type: str) -> None:
        if model_type == "unet":
            # Simple U-Net style segmentation model
            self.encoder = Sequential(
                Conv2d(3, 64, kernel_size=3, padding=1),
                BatchNorm2d(64),
                ReLU(),
                Conv2d(64, 64, kernel_size=3, padding=1),
                BatchNorm2d(64),
                ReLU(),
                MaxPool2d(2),
                Conv2d(64, 128, kernel_size=3, padding=1),
                BatchNorm2d(128),
                ReLU(),
                Conv2d(128, 128, kernel_size=3, padding=1),
                BatchNorm2d(128),
                ReLU(),
                MaxPool2d(2),
            )
            self.decoder = Sequential(
                Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                Conv2d(128, 64, kernel_size=3, padding=1),
                BatchNorm2d(64),
                ReLU(),
                Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                Conv2d(64, self.num_classes, kernel_size=1),
            )
            self.model = Sequential(self.encoder, self.decoder)
        elif model_type == "vit_seg":
            # ViT-based segmentation with a simple segmentation head
            self.backbone = timm.create_model(
                model_name="vit_small_patch16_dinov3.lvd1689m",
                pretrained=True,
                num_classes=0,  # Remove classification head
                img_size=224,
            )
            # Freeze backbone
            assert isinstance(self.backbone, Eva)
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            # Simple segmentation head
            # ViT outputs features of shape (B, num_patches, embed_dim)
            # We need to reshape and upsample to get segmentation masks
            embed_dim = self.backbone.embed_dim
            self.seg_head = Sequential(
                Linear(embed_dim, 256),
                ReLU(),
                Dropout(0.1),
                Linear(256, self.num_classes),
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.model_type = model_type

    @property
    def has_test_as_val(self) -> bool:
        if not isinstance(self.trainer.val_dataloaders, dict):
            return False
        return "test_as_val" in self.trainer.val_dataloaders

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.model_type == "unet":
            return self.model(x)
        elif self.model_type == "vit_seg":
            # Get patch embeddings from ViT
            features = self.backbone.forward_features(x)
            # Remove class token
            patch_features = features[:, 1:, :]  # (B, num_patches, embed_dim)
            
            # Apply segmentation head
            logits = self.seg_head(patch_features)  # (B, num_patches, num_classes)
            
            # Reshape to spatial dimensions
            # For 224x224 input with patch_size=16, we have 14x14 patches
            B, N, C = logits.shape
            H = W = int(N ** 0.5)
            logits = logits.transpose(1, 2).reshape(B, C, H, W)
            
            # Upsample to original resolution
            logits = F.interpolate(logits, size=(224, 224), mode="bilinear", align_corners=True)
            return logits
        else:
            return self.model(x)

    def on_train_start(self) -> None:
        self.train_iou = MulticlassJaccardIndex(num_classes=self.num_classes)

    def _step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        iou_metric: MulticlassJaccardIndex,
        prefix: str,
    ) -> torch.Tensor:
        x, y, _ = batch

        # Get predictions
        logits = self(x)
        
        # Resize target to match logits if needed
        if y.shape[-2:] != logits.shape[-2:]:
            y = F.interpolate(y.float(), size=logits.shape[-2:], mode="nearest")
        
        # Convert mask to class indices (assuming grayscale mask with class values)
        y = y.squeeze(1).long()  # Remove channel dimension and convert to long
        
        # Compute loss
        loss = F.cross_entropy(logits, y, ignore_index=255)  # 255 is commonly used for void/ignore class
        self.log(f"{prefix}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Compute pixel accuracy
        preds = torch.argmax(logits, dim=1)
        valid_mask = y != 255
        acc = (preds[valid_mask] == y[valid_mask]).float().mean()
        self.log(f"{prefix}_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # Compute IoU (Jaccard Index)
        iou_metric.update(preds, y)

        return loss

    def _compute_iou_and_log(
        self,
        iou_metric: MulticlassJaccardIndex,
        prefix: str,
    ) -> None:
        iou = iou_metric.compute()
        self.log(f"{prefix}_iou", iou, prog_bar=True)
        iou_metric.reset()

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        return self._step(
            batch=batch,
            iou_metric=self.train_iou,
            prefix="train",
        )

    def on_train_epoch_end(self) -> None:
        self._compute_iou_and_log(self.train_iou, "train")

    def on_validation_epoch_start(self) -> None:
        self.val_iou = MulticlassJaccardIndex(num_classes=self.num_classes)
        if self.has_test_as_val:
            self.test_as_val_iou = MulticlassJaccardIndex(num_classes=self.num_classes)

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
            loss = self._step(
                batch=val_batch,
                iou_metric=self.val_iou,
                prefix="val",
            )

        if test_batch is not None:
            self._step(
                batch=test_batch,
                iou_metric=self.test_as_val_iou,
                prefix="test_as_val",
            )

        return loss

    def on_validation_epoch_end(self) -> None:
        self._compute_iou_and_log(self.val_iou, "val")

        if self.has_test_as_val:
            self._compute_iou_and_log(self.test_as_val_iou, "test_as_val")

    def on_test_start(self) -> None:
        self.test_iou = MulticlassJaccardIndex(num_classes=self.num_classes)

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        return self._step(
            batch=batch,
            iou_metric=self.test_iou,
            prefix="test",
        )

    def on_test_epoch_end(self) -> None:
        self._compute_iou_and_log(
            iou_metric=self.test_iou,
            prefix="test",
        )

    def configure_optimizers(
        self,
    ) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        return optimizer
