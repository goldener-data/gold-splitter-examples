from functools import partial
from typing import Callable

import hydra
import torch
from PIL.Image import Image
from goldener.describe import GoldDescriptor
from goldener.extract import TorchGoldFeatureExtractor, TorchGoldFeatureExtractorConfig
from goldener.select import GoldSelector

from goldener.split import GoldSet, GoldSplitter
from goldener.vectorize import GoldVectorizer
from omegaconf import DictConfig
from torchvision.models import get_model_weights


def collate_cifar10(
    batch: list[tuple[Image, int]], preprocess: Callable[[Image], torch.Tensor]
) -> dict[str, torch.Tensor]:
    images, targets = zip(*batch)
    imgs_tensor = torch.stack([preprocess(image) for image in images])
    tgts_tensor = torch.tensor(targets)
    return {"image": imgs_tensor, "class": tgts_tensor}


def get_gold_splitter(splitter_cfg: DictConfig, split_ratio: float) -> GoldSplitter:
    splitter_config = hydra.utils.instantiate(splitter_cfg)
    model_creation_func = splitter_config["model"]
    model_weights = get_model_weights(model_creation_func).__members__[
        splitter_config["weights"]
    ]
    model = model_creation_func(model_weights)
    collate_fn = partial(collate_cifar10, preprocess=model_weights.transforms())

    config = TorchGoldFeatureExtractorConfig(
        model=model, layers=[splitter_config["layers"]]
    )
    extractor = TorchGoldFeatureExtractor(config)

    descriptor = GoldDescriptor(
        table_path="cifar10_experiment.describe",
        extractor=extractor,
        collate_fn=collate_fn,
        batch_size=splitter_config["batch_size"],
        num_workers=splitter_config["num_workers"],
        device=torch.device("cpu"),
    )

    selector = GoldSelector(
        table_path="cifar10_experiment.select",
        vectorizer=GoldVectorizer(),
        reducer=splitter_config["reducer"],
        chunk=1000,
        batch_size=splitter_config["batch_size"],
        num_workers=splitter_config["num_workers"],
    )

    sets = [
        GoldSet(name="train", ratio=1 - split_ratio),
        GoldSet(name="val", ratio=split_ratio),
    ]

    return GoldSplitter(
        sets=sets, descriptor=descriptor, selector=selector, class_key="class"
    )
