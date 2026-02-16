import pixeltable as pxt

import hydra
import timm
import torch
from PIL.Image import Image
from sklearn.cluster import KMeans
from goldener.clusterize import GoldSKLearnClusteringTool, GoldClusterizer
from goldener.describe import GoldDescriptor
from goldener.extract import TorchGoldFeatureExtractor, TorchGoldFeatureExtractorConfig
from goldener.select import GoldSelector, GoldGreedyClosestPointSelection
from goldener.split import GoldSet, GoldSplitter
from goldener.vectorize import (
    TensorVectorizer,
    Filter2DWithCount,
    FilterLocation,
)
from omegaconf import DictConfig
from torchvision.transforms.v2 import Compose, ToTensor, Normalize, Resize
import numpy as np

VOC_PREPROCESS = Compose(
    [
        ToTensor(),
        Resize(224),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


def collate_voc(
    batch: list[tuple[Image, Image, int]],
) -> dict[str, torch.Tensor | list[str] | list[int]]:
    """
    Collate function for VOC dataset that processes images and segmentation masks.
    For gold splitting, we need to extract features from patches corresponding to
    the segmentation mask rather than from class tokens.
    
    Note: The mask is used for creating a label identifier, but feature extraction
    is done from the image patches only.
    """
    images, masks, indices = zip(*batch)
    imgs_tensor = torch.stack([VOC_PREPROCESS(image) for image in images])
    
    # Create a label string based on the image index
    # We use a simple label since segmentation doesn't have a single class per image
    str_targets = [f"img_{i}" for i in indices]
    idx_list = [int(idx) for idx in indices]
    return {
        "data": imgs_tensor,
        "label": str_targets,
        "idx": idx_list,
    }


def get_gold_descriptor(
    table_name: str,
    min_pxt_insert_size: int,
    batch_size: int,
    num_workers: int,
    to_keep_schema: dict,
) -> GoldDescriptor:
    """
    Create a GoldDescriptor for VOC segmentation.
    
    The key difference from image classification is that we extract features
    from patches corresponding to the segmentation mask rather than using
    the class token. This is done by using Filter2DWithCount with FilterLocation.ALL
    to keep all patch embeddings, which will be filtered based on the mask.
    """
    device = (
        torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
    )

    extractor = TorchGoldFeatureExtractor(
        TorchGoldFeatureExtractorConfig(
            model=timm.create_model(
                model_name="vit_small_patch16_dinov3.lvd1689m",
                pretrained=True,
                img_size=224,
                device=device,
            ),
            layers=["blocks.11"],
        )
    )

    # For segmentation, we want to extract features from all patches, not just the class token
    # We'll use ALL filter location to get all patch embeddings (excluding class token)
    # Then we can filter based on the segmentation mask
    return GoldDescriptor(
        table_path=table_name,
        extractor=extractor,
        vectorizer=TensorVectorizer(
            keep=Filter2DWithCount(keep=True, filter_location=FilterLocation.ALL),
            channel_pos=2,
        ),
        collate_fn=collate_voc,
        to_keep_schema=to_keep_schema,
        min_pxt_insert_size=min_pxt_insert_size,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )


def get_gold_splitter(
    splitter_cfg: DictConfig,
    name_prefix: str,
    val_ratio: float,
    max_batches: int | None = None,
) -> GoldSplitter:
    splitter_config = hydra.utils.instantiate(splitter_cfg)

    batch_size = splitter_config["batch_size"]
    num_workers = splitter_config["num_workers"]
    min_pxt_insert_size = splitter_config["min_pxt_insert_size"]
    n_clusters = splitter_config["n_clusters"]

    to_keep_schema = {"label": pxt.String}

    table_name = f"{name_prefix}_{splitter_config["table_name"]}"

    clusterizer = (
        None
        if n_clusters < 2
        else GoldClusterizer(
            table_path=f"{table_name}_cluster",
            clustering_tool=GoldSKLearnClusteringTool(
                KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
            ),
            vectorized_key="features",
            min_pxt_insert_size=min_pxt_insert_size,
            batch_size=batch_size,
            num_workers=num_workers,
            to_keep_schema=to_keep_schema,
        )
    )

    descriptor = get_gold_descriptor(
        table_name=f"{table_name}_description",
        min_pxt_insert_size=min_pxt_insert_size,
        batch_size=batch_size,
        num_workers=num_workers,
        to_keep_schema=to_keep_schema,
    )

    # Splitting will be done by moving iteratively to the validation set
    # all the data with the closest distance to their neighbors
    selector = GoldSelector(
        table_path=f"{table_name}_selection",
        selection_tool=GoldGreedyClosestPointSelection(
            device="cuda:0" if torch.cuda.is_available() else "cpu"
        ),
        reducer=None,
        vectorized_key="features",
        class_key="label",
        to_keep_schema=to_keep_schema,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    sets = [
        GoldSet(name="val", size=val_ratio),
        GoldSet(name="train", size=1 - val_ratio),
    ]

    return GoldSplitter(
        sets=sets,
        descriptor=descriptor,
        vectorizer=None,
        clusterizer=clusterizer,
        selector=selector,
        max_batches=max_batches,
    )
