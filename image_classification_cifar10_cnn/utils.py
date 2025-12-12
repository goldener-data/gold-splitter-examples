import pixeltable as pxt

import hydra
import timm
import torch
from PIL.Image import Image
from goldener.describe import GoldDescriptor
from goldener.extract import TorchGoldFeatureExtractor, TorchGoldFeatureExtractorConfig
from goldener.select import GoldSelector

from goldener.split import GoldSet, GoldSplitter
from goldener.vectorize import (
    GoldVectorizer,
    TensorVectorizer,
    Filter2DWithCount,
    FilterLocation,
)
from omegaconf import DictConfig
from torchvision.transforms.v2 import Compose, ToTensor, Normalize, CenterCrop


def collate_cifar10(
    batch: list[tuple[Image, int]],
) -> dict[str, torch.Tensor | list[str]]:
    preprocess = Compose(
        [
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            CenterCrop(28),
        ]
    )
    images, targets = zip(*batch)
    imgs_tensor = torch.stack([preprocess(image) for image in images])
    str_targets = [str(target) for target in targets]
    return {"data": imgs_tensor, "label": str_targets}


def get_gold_splitter(
    splitter_cfg: DictConfig, split_ratio: float, max_batches: int
) -> GoldSplitter:
    splitter_config = hydra.utils.instantiate(splitter_cfg)
    model = timm.create_model(
        "vit_base_patch14_reg4_dinov2.lvd142m",
        pretrained=True,
        img_size=28,
    )

    config = TorchGoldFeatureExtractorConfig(
        model=model, layers=list(splitter_config["layers"])
    )
    extractor = TorchGoldFeatureExtractor(config)

    to_keep_schema = {"label": pxt.String}
    descriptor = GoldDescriptor(
        table_path="cifar10_experiment.describe",
        extractor=extractor,
        collate_fn=collate_cifar10,
        to_keep_schema=to_keep_schema,
        batch_size=splitter_config["batch_size"],
        num_workers=splitter_config["num_workers"],
        device=torch.device("cpu")
        if not torch.cuda.is_available()
        else torch.device("cuda"),
    )
    vectorizer = GoldVectorizer(
        table_path="cifar10_experiment.vectorize",
        vectorizer=TensorVectorizer(
            keep=Filter2DWithCount(keep=True, filter_location=FilterLocation.START),
            channel_pos=2,
        ),
        to_keep_schema={"label": pxt.String},
        batch_size=splitter_config["batch_size"],
        num_workers=splitter_config["num_workers"],
    )
    selector = GoldSelector(
        table_path="cifar10_experiment.select",
        reducer=splitter_config["reducer"],
        chunk=splitter_config["chunk"],
        class_key="label",
        to_keep_schema={"label": pxt.String},
        batch_size=splitter_config["batch_size"],
        num_workers=splitter_config["num_workers"],
    )

    sets = [
        GoldSet(name="val", ratio=split_ratio),
        GoldSet(name="train", ratio=1 - split_ratio),
    ]

    return GoldSplitter(
        sets=sets,
        descriptor=descriptor,
        vectorizer=vectorizer,
        selector=selector,
        max_batches=max_batches,
    )
