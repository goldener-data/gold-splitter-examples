from collections import defaultdict
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Tuple, Callable, Literal

import torch
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
    ColorJitter,
    Resize,
)
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import pixeltable as pxt
from goldener.split import GoldSplitter

from image_classification_cifar10.utils import get_gold_splitter, get_gold_descriptor

logger = getLogger(__name__)


@dataclass
class Sample:
    dataset_idx: int
    features: np.ndarray
    label: str
    training_set: Literal["train", "val"] | None = None


class GoldCifar10(CIFAR10):
    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: None | Callable = None,
        target_transform: None | Callable = None,
        download: bool = False,
        count: int | None = None,
        remove_ratio: float | None = None,
        to_duplicate_clusters: int | None = None,
        cluster_count: int | None = None,
        duplicate_per_sample: int | None = None,
        random_state: int = 42,
    ) -> None:
        self.count = count
        super().__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        if count is not None:
            self.data: np.ndarray = self.data[:count]
            self.targets: list[int] = self.targets[:count]

        if remove_ratio is not None:
            training_indices, excluded = train_test_split(
                range(len(self)),
                test_size=remove_ratio,
                random_state=random_state,
                shuffle=True,
                stratify=self.targets_as_array,
            )
            self.data = self.data[training_indices]
            self.targets = [self.targets[i] for i in training_indices]

        if to_duplicate_clusters is not None and cluster_count is not None:
            gold_descriptor = get_gold_descriptor(
                table_name="gold_cifar10_descriptor",
                min_pxt_insert_size=10000,
                batch_size=128,
                num_workers=16,
                to_keep_schema={"label": pxt.String},
            )
            pxt.drop_table(gold_descriptor.table_path, if_not_exists="ignore")

            vectorized = gold_descriptor.describe_in_table(self)

            torch.cuda.empty_cache()
            features_per_label = defaultdict(list)
            indices_per_label = defaultdict(list)

            for row in vectorized.select(
                vectorized.idx, vectorized.features, vectorized.label
            ).collect():
                features_per_label[row["label"]].append(row["features"])
                indices_per_label[row["label"]].append(row["idx"])

            for label, features in features_per_label.items():
                label_indices = indices_per_label[label]
                logger.info(f"Adding duplicates for label {label}")
                kmeans = KMeans(
                    n_clusters=cluster_count,
                    random_state=random_state,
                    n_init="auto",
                ).fit(np.stack(features, axis=0))
                cluster_indices = np.random.choice(
                    range(cluster_count), size=to_duplicate_clusters, replace=False
                )
                for ci in cluster_indices:
                    for i, cluster_id in enumerate(kmeans.labels_):
                        if cluster_id == ci:
                            to_add_data = np.vstack(
                                [self.data[label_indices[i]][np.newaxis, ...]]
                                * duplicate_per_sample  # type: ignore[operator]
                            )
                            self.data = np.vstack([self.data, to_add_data])
                            self.targets.extend(
                                [self.targets[label_indices[i]]] * duplicate_per_sample  # type: ignore[operator]
                            )

    def __len__(self) -> int:
        return len(self.data) if self.count is None else min(self.count, len(self.data))

    def __getitem__(self, index: int) -> Tuple:
        if self.count is not None and index >= self.count:
            raise IndexError("Index out of range for GoldCifar10 with limited count.")
        return super().__getitem__(index) + (index,)

    @property
    def targets_as_array(self) -> np.ndarray:
        return np.array(self.targets)


class CIFAR10DataModule(LightningDataModule):
    def __init__(
        self,
        cfg: DictConfig,
    ) -> None:
        super().__init__()

        self.data_dir = cfg["data_dir"]
        self.gold_splitter_cfg = cfg["gold_splitter"]

        self.random_state = cfg["random_state"]

        self.train_ratio = cfg["train_ratio"]
        self.val_ratio = cfg["val_ratio"]

        self.random_split_state = cfg["random_split_state"]
        self.remove_ratio = cfg["remove_ratio"]
        self.to_duplicate_clusters = cfg["remove"]
        self.cluster_count = cfg["cluster_count"]
        self.duplicate_per_sample = cfg["duplicate_per_sample"]

        self.batch_size = cfg["batch_size"]
        self.num_workers = cfg["num_workers"]
        self.max_batches = cfg["debug_train_count"]
        self.train_count = (
            cfg["debug_train_count"] * self.batch_size
            if cfg["debug_train_count"] is not None
            else None
        )
        self.validate_on_test = cfg["validate_on_test"]

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
                ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ]
            + list(self.transform_test.transforms)
        )

        self.gold_splitter: GoldSplitter = get_gold_splitter(
            splitter_cfg=self.gold_splitter_cfg,
            name_prefix=(
                f"settings_{self.random_state}_{self.remove_ratio}"
                f"_{self.cluster_count}_{self.to_duplicate_clusters}"
                f"_{self.duplicate_per_sample}"
            ),
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            max_batches=self.max_batches,
        )
        if cfg["gold_splitter"]["update_selection"]:
            pxt.drop_table(
                self.gold_splitter.descriptor.table_path, if_not_exists="ignore"
            )

        self.excluded: Subset
        self.gold_train_indices: list[int]
        self.gold_val_indices: list[int]
        self.gold_train_dataset: Subset
        self.gold_val_dataset: Subset
        self.sk_train_indices: list[int]
        self.sk_val_indices: list[int]
        self.sk_train_dataset: Subset
        self.sk_val_dataset: Subset
        self.test_dataset: torchvision.datasets.CIFAR10

    def prepare_data(self) -> None:
        # Download CIFAR-10
        torchvision.datasets.CIFAR10(root=self.data_dir, train=True, download=True)
        torchvision.datasets.CIFAR10(root=self.data_dir, train=False, download=True)

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            dataset = GoldCifar10(
                root=self.data_dir,
                train=True,
                transform=self.transform_test,
                download=False,
                count=self.train_count,
                random_state=self.random_state,
                remove_ratio=0.9,
                to_duplicate_clusters=4,
                cluster_count=50,
                duplicate_per_sample=25,
            )

            # make random splitting with sklearn
            self.sk_train_indices, self.sk_val_indices = train_test_split(
                range(len(dataset)),
                test_size=int(self.val_ratio * len(dataset)),
                random_state=self.random_split_state,
                shuffle=True,
                stratify=dataset.targets_as_array,
            )
            self.sk_train_dataset = Subset(dataset, self.sk_train_indices)
            self.sk_val_dataset = Subset(dataset, self.sk_val_indices)

            # make gold splitting
            split_table = self.gold_splitter.split_in_table(dataset)
            splits = self.gold_splitter.get_split_indices(
                split_table, selection_key="selected", idx_key="idx"
            )

            self.gold_train_indices = list(splits["train"])
            self.gold_val_indices = list(splits["val"])
            self.gold_train_dataset = Subset(dataset, self.gold_train_indices)
            self.gold_val_dataset = Subset(dataset, self.gold_val_indices)

        if stage == "test" or stage is None:
            self.test_dataset = torchvision.datasets.CIFAR10(
                root=self.data_dir,
                train=False,
                transform=self.transform_test,
                download=False,
            )

    def _get_features_by_indices(
        self,
        indices: list[int],
        label: str | None = None,
    ) -> list[Sample]:
        vectorized = pxt.get_table(self.gold_splitter.descriptor.table_path)
        assert vectorized is not None
        query = vectorized.idx.isin(indices)
        if label is not None:
            query = query & (vectorized.label == label)  # type: ignore[assignment]

        return [
            row["features"]
            for row in vectorized.where(query).select(vectorized.features).collect()
        ]

    def get_gold_train_features(self, label: str | None = None) -> list[Sample]:
        return self._get_features_by_indices(
            self.gold_train_indices,
            label,
        )

    def get_gold_val_features(self, label: str | None = None) -> list[Sample]:
        return self._get_features_by_indices(
            self.gold_val_indices,
            label,
        )

    def get_sk_train_features(self, label: str | None = None) -> list[Sample]:
        return self._get_features_by_indices(
            self.sk_train_indices,
            label,
        )

    def get_sk_val_features(self, label: str | None = None) -> list[Sample]:
        return self._get_features_by_indices(
            self.sk_val_indices,
            label,
        )

    def get_training_samples(self, label: str | None = None) -> list[Sample]:
        vectorized = pxt.get_table(self.gold_splitter.descriptor.table_path)
        assert vectorized is not None

        return [
            Sample(
                dataset_idx=row["idx"],
                features=row["features"],
                label=row["label"],
                training_set=None,
            )
            for row in vectorized.where(vectorized.label == label)
            .select(vectorized.idx, vectorized.features, vectorized.label)
            .collect()
        ]

    def sk_train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.sk_train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True,
        )

    def sk_val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.sk_val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True,
        )

    def gold_train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.gold_train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True,
        )

    def gold_val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.gold_val_dataset,
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
