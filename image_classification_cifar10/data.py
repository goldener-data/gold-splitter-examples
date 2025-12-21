from pathlib import Path
from typing import Tuple, Callable

from lightning import LightningDataModule
import numpy as np
from omegaconf import DictConfig
from torch.utils.data import Subset, DataLoader
import torchvision
from torchvision.datasets import CIFAR10
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    ToTensor,
    RandomHorizontalFlip,
    Resize,
)
from sklearn.model_selection import train_test_split


from image_classification_cifar10.utils import get_gold_splitter


class GoldCifar10(CIFAR10):
    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: None | Callable = None,
        target_transform: None | Callable = None,
        download: bool = False,
        count: int | None = None,
    ) -> None:
        self.count = count
        super().__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

    def __len__(self) -> int:
        return len(self.data) if self.count is None else min(self.count, len(self.data))

    def __getitem__(self, index: int) -> Tuple:
        if self.count is not None and index >= self.count:
            raise IndexError(
                "Index out of range for GoldSplitterCifar10 with limited count."
            )
        return super().__getitem__(index)

    @property
    def targets_as_array(self) -> np.ndarray:
        return np.array(self.targets[: self.__len__()])


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
        self.train_ratio = cfg["train_ratio"]
        self.split_method = split_method
        self.val_ratio = cfg["val_ratio"]
        self.random_state = cfg["random_state"]
        self.gold_splitter_cfg = cfg["gold_splitter"]
        self.max_batches = cfg["debug_train_count"]
        self.train_count = (
            cfg["debug_train_count"] * self.batch_size
            if cfg["debug_train_count"] is not None
            else None
        )
        self.split_exists = cfg["split_exists"]

        # Define transforms
        self.transform_test = Compose(
            [
                ToTensor(),
                Resize(224),
                Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
        self.transform_train = Compose(
            [
                RandomHorizontalFlip(),
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
        full_train_dataset = GoldCifar10(
            root=self.data_dir,
            train=True,
            transform=self.transform_train,
            download=False,
            count=self.train_count,
        )

        if stage == "fit" or stage is None:
            train_indices, val_indices = self._split_data(full_train_dataset)
            self.train_dataset = Subset(
                dataset=full_train_dataset, indices=train_indices.tolist()
            )
            self.val_dataset = Subset(
                dataset=GoldCifar10(
                    root=self.data_dir,
                    train=True,
                    transform=self.transform_test,
                    download=False,
                    count=self.train_count,
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
            total_ratio = self.train_ratio + self.val_ratio
            total_length = len(dataset)
            if total_ratio >= 1.0:
                train_indices, val_indices = train_test_split(
                    np.arange(len(dataset)),
                    test_size=int(self.val_ratio * len(dataset)),
                    random_state=self.random_state,
                    shuffle=True,
                    stratify=dataset.targets_as_array,
                )
            else:
                training_indices, excluded_indices = train_test_split(
                    np.arange(len(dataset)),
                    test_size=int((1-total_ratio) * len(dataset)),
                    random_state=self.random_state,
                    shuffle=True,
                    stratify=dataset.targets_as_array,
                )
                val_ratio = self.val_ratio / (self.train_ratio + self.val_ratio)

                train_indices, val_indices = train_test_split(
                    np.arange(len(training_indices)),
                    test_size=int(val_ratio * len(training_indices)),
                    random_state=self.random_state,
                    shuffle=True,
                    stratify=dataset.targets_as_array[training_indices],
                )
        elif self.split_method == "gold":
            gold_splitter = get_gold_splitter(
                splitter_cfg=self.gold_splitter_cfg,
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                max_batches=self.max_batches
            )
            split_table = gold_splitter.split_in_table(dataset)
            splits = gold_splitter.get_split_indices(
                split_table, selection_key="selected", idx_key="idx"
            )
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
