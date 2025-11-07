from typing import Tuple

from lightning import LightningDataModule
import numpy as np
from omegaconf import DictConfig
from torch.utils.data import Subset, DataLoader
import torchvision
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    ToTensor,
    RandomHorizontalFlip,
    RandomCrop,
)
from sklearn.model_selection import train_test_split

from image_classification_cifar10_cnn.utils import get_gold_splitter


class CIFAR10DataModule(LightningDataModule):
    def __init__(
        self,
        cfg: DictConfig,
        split_method: str = "random",
    ) -> None:
        super().__init__()
        self.data_dir = cfg["data_dir"]
        self.batch_size = cfg["batch_size"]
        self.num_workers = cfg["num_workers"]
        self.split_method = split_method
        self.val_ratio = cfg["val_ratio"]
        self.random_state = cfg["random_state"]
        self.gold_splitter_cfg = cfg["gold_splitter"]

        # Define transforms
        self.transform_test = Compose(
            [
                ToTensor(),
                Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
        self.transform_train = Compose(
            [
                RandomHorizontalFlip(),
                RandomCrop(32, padding=4),
            ]
            + list(self.transform_test.transforms)
        )

        self.train_dataset: Subset
        self.val_dataset: Subset
        self.test_dataset: torchvision.datasets.CIFAR10

    def prepare_data(self) -> None:
        # Download CIFAR-10
        torchvision.datasets.CIFAR10(root=self.data_dir, train=True, download=True)
        torchvision.datasets.CIFAR10(root=self.data_dir, train=False, download=True)

    def setup(self, stage: str | None = None) -> None:
        # Load full training set
        full_train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            transform=self.transform_train,
            download=False,
        )

        train_indices, val_indices = self._split_data(full_train_dataset)
        if stage == "fit" or stage is None:
            self.train_dataset = Subset(
                dataset=full_train_dataset, indices=train_indices.tolist()
            )
            self.val_dataset = Subset(
                dataset=torchvision.datasets.CIFAR10(
                    root=self.data_dir,
                    train=True,
                    transform=self.transform_test,
                    download=False,
                ),
                indices=val_indices.tolist(),
            )

        if stage == "test" or stage is None:
            self.test_dataset = torchvision.datasets.CIFAR10(
                root=self.data_dir,
                train=False,
                transform=self.transform_test,
                download=False,
            )

    def _split_data(self, dataset) -> Tuple[np.ndarray, np.ndarray]:
        if self.split_method == "random":
            train_indices, val_indices = train_test_split(
                np.arange(len(dataset)),
                test_size=int(self.val_ratio * len(dataset)),
                random_state=self.random_state,
                shuffle=True,
            )
        elif self.split_method == "gold":
            gold_splitter = get_gold_splitter(self.gold_splitter_cfg, self.val_ratio)
            splits = gold_splitter.split(dataset)
            train_indices = np.array(list(splits["train"]))
            val_indices = np.array(list(splits["val"]))
        else:
            raise ValueError(f"Unknown split method: {self.split_method}")

        return train_indices, val_indices

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True,
        )
