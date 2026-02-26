import pixeltable as pxt
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from transformers import AutoModel

from goldener import (
    GoldSKLearnClusteringTool,
    GoldClusterizer,
    GoldDescriptor,
    TorchGoldFeatureExtractor,
    TorchGoldFeatureExtractorConfig,
    GoldSelector,
    GoldGreedyKCenterSelection,
    GoldSet,
    GoldSplitter,
)

from omegaconf import DictConfig


class BertCLSExtractor(nn.Module):
    """BERT wrapper that returns the CLS token hidden state.

    Accepts a stacked tensor of shape (batch, 2, seq_len) where
    the first slice contains input_ids and the second contains attention_mask.
    The final Identity module (cls_pool) is used as the hook target for GoldDescriptor.
    """

    def __init__(self, pretrained_model: str = "bert-base-uncased") -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model)
        self.cls_pool = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_ids = x[:, 0, :].long()
        attention_mask = x[:, 1, :].long()
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.cls_pool(cls_embedding)


def collate_imdb(
    batch: list[tuple[torch.Tensor, torch.Tensor, int, int]],
) -> dict[str, torch.Tensor | list[str] | list[int]]:
    """Collate IMDb samples for use with GoldDescriptor.

    Returns a dict with:
      - ``data``: stacked tensor of shape ``(batch, 2, seq_len)``
        where ``data[:, 0]`` = input_ids and ``data[:, 1]`` = attention_mask.
      - ``label``: list of string labels ("0" or "1").
      - ``idx``: list of sample indices.
    """
    input_ids, attention_masks, labels, indices = zip(*batch)
    ids_tensor = torch.stack(list(input_ids))
    masks_tensor = torch.stack(list(attention_masks))
    stacked = torch.stack([ids_tensor, masks_tensor], dim=1)
    str_labels = [str(label) for label in labels]
    idx_list = [int(idx) for idx in indices]
    return {
        "data": stacked,
        "label": str_labels,
        "idx": idx_list,
    }


def get_gold_descriptor(
    table_name: str,
    min_pxt_insert_size: int,
    batch_size: int,
    num_workers: int,
    to_keep_schema: dict,
    pretrained_model: str = "bert-base-uncased",
) -> GoldDescriptor:
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")

    bert_cls = BertCLSExtractor(pretrained_model=pretrained_model)
    bert_cls.eval()
    bert_cls.to(device)

    extractor = TorchGoldFeatureExtractor(
        TorchGoldFeatureExtractorConfig(
            model=bert_cls,
            layers=["cls_pool"],
        )
    )

    return GoldDescriptor(
        table_path=table_name,
        extractor=extractor,
        vectorizer=None,
        collate_fn=collate_imdb,
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
    import hydra

    splitter_config = hydra.utils.instantiate(splitter_cfg)

    batch_size = splitter_config["batch_size"]
    num_workers = splitter_config["num_workers"]
    min_pxt_insert_size = splitter_config["min_pxt_insert_size"]
    n_clusters = splitter_config["n_clusters"]
    pretrained_model = splitter_config["pretrained_model"]

    to_keep_schema = {"label": pxt.String}

    table_name = f"{name_prefix}_{splitter_config['table_name']}"

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
        pretrained_model=pretrained_model,
    )

    selector = GoldSelector(
        table_path=f"{table_name}_selection",
        selection_tool=GoldGreedyKCenterSelection(
            device="cuda:0" if torch.cuda.is_available() else "cpu"
        ),
        reducer=None,
        vectorized_key="features",
        label_key="label",
        to_keep_schema=to_keep_schema,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    sets = [
        GoldSet(name="train", size=1 - val_ratio),
        GoldSet(name="val", size=val_ratio),
    ]

    return GoldSplitter(
        sets=sets,
        descriptor=descriptor,
        vectorizer=None,
        clusterizer=clusterizer,
        selector=selector,
        max_batches=max_batches,
    )
