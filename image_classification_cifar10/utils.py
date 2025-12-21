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
    TensorVectorizer,
    Filter2DWithCount,
    FilterLocation,
)
from omegaconf import DictConfig
from torchvision.transforms.v2 import Compose, ToTensor, Normalize, Resize


def collate_cifar10(
    batch: list[tuple[Image, int]],
) -> dict[str, torch.Tensor | list[str]]:
    preprocess = Compose(
        [
            ToTensor(),
            Resize(224),
            Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    images, targets = zip(*batch)
    imgs_tensor = torch.stack([preprocess(image) for image in images])
    str_targets = [str(target) for target in targets]
    return {"data": imgs_tensor, "label": str_targets}


def get_gold_splitter(
    splitter_cfg: DictConfig, train_ratio: float, val_ratio: float, max_batches: int
) -> GoldSplitter:
    splitter_config = hydra.utils.instantiate(splitter_cfg)
    device = (
        torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
    )
    batch_size = splitter_config["batch_size"]
    num_workers = splitter_config["num_workers"]
    min_pxt_insert_size = splitter_config["min_pxt_insert_size"]
    table_name = splitter_config["table_name"]

    extractor = TorchGoldFeatureExtractor(
        TorchGoldFeatureExtractorConfig(
            model=timm.create_model(
                model_name="vit_small_patch16_dinov3.lvd1689m",
                pretrained=True,
                img_size=224,
                device=device,
            ),
            layers=["blocks.9"],
        )
    )

    to_keep_schema = {"label": pxt.String}

    if splitter_config.update_selection:
        pxt.drop_table(table_name, if_not_exists="ignore")

    descriptor = GoldDescriptor(
        table_path=table_name,
        extractor=extractor,
        vectorizer=TensorVectorizer(
            keep=Filter2DWithCount(keep=True, filter_location=FilterLocation.START),
            channel_pos=2,
        ),
        collate_fn=collate_cifar10,
        to_keep_schema=to_keep_schema,
        min_pxt_insert_size=min_pxt_insert_size,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )

    selector = GoldSelector(
        table_path=table_name,
        vectorized_key="features",
        class_key="label",
        to_keep_schema=to_keep_schema,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    sets = [
        GoldSet(name="train", ratio=train_ratio),
        GoldSet(name="val", ratio=val_ratio),
    ]

    return GoldSplitter(
        sets=sets,
        descriptor=descriptor,
        vectorizer=None,
        selector=selector,
        max_batches=max_batches,
    )
