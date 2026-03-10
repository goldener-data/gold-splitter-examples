import pytest
import torch

from image_segmentation_pascal_voc.utils import (
    transform_rgb_mask_to_class_mask,
    transform_segmentation_logits_to_rgb_preds,
    transform_rgb_mask_to_mono_mask,
    multilabel_iterative_train_test_split,
)

from image_segmentation_pascal_voc.data import GoldPascalVOC2012Segmentation


class TestGoldVOCSegmentation:
    def __init__(self):
        GoldPascalVOC2012Segmentation(".data", override=True)

    def test_simple_usage(self):
        dataset = GoldPascalVOC2012Segmentation(".data")
        assert len(dataset) > 0
        # Pascal VOC 2012 train set has 1464 images
        assert len(dataset) == 1464

    def test_with_remove_ratio(self):
        remove_ratio = 0.2
        first_dataset = GoldPascalVOC2012Segmentation(
            ".data", remove_ratio=remove_ratio
        )
        expected_length = int(1464 * (1 - remove_ratio))
        assert len(first_dataset) == expected_length

        second_dataset = GoldPascalVOC2012Segmentation(
            ".data", remove_ratio=remove_ratio
        )
        assert len(first_dataset) == len(second_dataset)

    def test_with_duplicate_failure(self):
        duplicate_table_path = "unit_test_voc_description_table_failure"
        drop_duplicate_table = True
        to_duplicate_clusters = 3
        cluster_count = 2
        duplicate_per_sample = None
        with pytest.raises(ValueError):
            GoldPascalVOC2012Segmentation(
                ".data",
                remove_ratio=0.98,
                duplicate_table_path=duplicate_table_path,
                drop_duplicate_table=drop_duplicate_table,
                to_duplicate_clusters=to_duplicate_clusters,
                cluster_count=cluster_count,
                duplicate_per_sample=duplicate_per_sample,
            )

    def test_with_duplicate(self):
        duplicate_table_path = "unit_test_voc_description_table"
        drop_duplicate_table = True
        to_duplicate_clusters = 2
        cluster_count = 2
        duplicate_per_sample = 2

        first_dataset = GoldPascalVOC2012Segmentation(
            ".data",
            remove_ratio=0.98,
            duplicate_table_path=duplicate_table_path,
            drop_duplicate_table=drop_duplicate_table,
            to_duplicate_clusters=to_duplicate_clusters,
            cluster_count=cluster_count,
            duplicate_per_sample=duplicate_per_sample,
        )

        second_dataset = GoldPascalVOC2012Segmentation(
            ".data",
            remove_ratio=0.98,
            duplicate_table_path=duplicate_table_path,
            drop_duplicate_table=drop_duplicate_table,
            to_duplicate_clusters=to_duplicate_clusters,
            cluster_count=cluster_count,
            duplicate_per_sample=duplicate_per_sample,
        )

        assert len(first_dataset) == len(second_dataset)


class TestMaskConversions:
    def test_transform_rgb_mask_to_class_mask(self):
        rgb_to_class_idx = {(255, 0, 0): 0, (0, 255, 0): 1}
        mask = torch.tensor(
            [
                [
                    [[255, 0], [0, 255]],  # R channel
                    [[0, 255], [255, 0]],  # G channel
                    [[0, 0], [0, 0]],  # B channel
                ],
                [
                    [[0, 255], [255, 0]],
                    [[255, 0], [0, 255]],
                    [[0, 0], [0, 0]],
                ],
            ],
            dtype=torch.uint8,
        )

        out = transform_rgb_mask_to_class_mask(mask, rgb_to_class_idx)

        assert out.shape == (2, 2, 2, 2)

        assert out[0, 0, 0, 0] == 1.0
        assert out[0, 0, 1, 1] == 1.0
        assert out[1, 0, 0, 1] == 1.0
        assert out[1, 0, 1, 0] == 1.0

        assert out[0, 1, 0, 1] == 1.0
        assert out[0, 1, 1, 0] == 1.0
        assert out[1, 1, 0, 0] == 1.0
        assert out[1, 1, 1, 1] == 1.0

    def test_transform_segmentation_logits_to_rgb_preds(self):
        logits = torch.tensor(
            [
                [
                    [[10.0, 0.0], [0.0, 10.0]],  # class 0
                    [[0.0, 10.0], [10.0, 0.0]],  # class 1
                ]
            ]
        )
        rgb_to_class_idx = {(255, 0, 0): 0, (0, 255, 0): 1}
        preds = transform_segmentation_logits_to_rgb_preds(logits, rgb_to_class_idx)

        assert (preds[0, :, 0, 0] == torch.tensor([255, 0, 0], dtype=torch.uint8)).all()
        assert (preds[0, :, 1, 1] == torch.tensor([255, 0, 0], dtype=torch.uint8)).all()
        assert (preds[0, 1, 0, 1] == torch.tensor([0, 255, 0], dtype=torch.uint8)).all()
        assert (preds[0, 1, 1, 0] == torch.tensor([0, 255, 0], dtype=torch.uint8)).all()

    def test_transform_rgb_mask_to_mono_mask_happy_path(self):
        rgb_to_class_idx = {(255, 0, 0): 2, (0, 255, 0): 5}
        mask = torch.tensor(
            [
                [
                    [[255, 0], [0, 255]],
                    [[0, 255], [255, 0]],
                    [[0, 0], [0, 0]],
                ]
            ],
            dtype=torch.uint8,
        )
        mono = transform_rgb_mask_to_mono_mask(mask, rgb_to_class_idx)
        assert mono.shape == (1, 1, 2, 2)
        # (0,0) is red -> class 2
        assert mono[0, 0, 0, 0] == 2
        # (0,1) is green -> class 5
        assert mono[0, 0, 0, 1] == 5

    def test_transform_rgb_mask_to_class_mask_missing_mapping_raises(self):
        rgb_to_class_idx = {(255, 0, 0): 0}
        # create a mask containing an unknown color (0,0,255)
        mask = torch.tensor(
            [
                [
                    [[0]],
                    [[0]],
                    [[255]],
                ]
            ],
            dtype=torch.uint8,
        )
        # expand to shape (B,3,H,W)
        mask = mask.expand(1, 3, 1, 1)

        with pytest.raises(ValueError):
            transform_rgb_mask_to_class_mask(mask, rgb_to_class_idx)


class TestMultilabelIterativeTrainTestSplit:
    def test_basic_split_ratio(self):
        """Test that the split returns the correct ratio of samples."""
        label_indices = {
            0: set(["a", "b", "c", "d"]),
            1: set(["b", "c", "d", "e"]),
            2: set(["c", "d", "e", "f"]),
            3: set(["a", "b", "c", "d"]),
            4: set(["b", "c", "d", "e"]),
            5: set(["c", "d", "e", "f"]),
            6: set(["a", "b", "c", "d"]),
            7: set(["b", "c", "d", "e"]),
            8: set(["c", "d", "e", "f"]),
            9: set(["a", "b", "c", "d"]),
        }
        test_size = 0.3
        train_indices, test_indices = multilabel_iterative_train_test_split(
            label_indices, test_size=test_size, random_seed=42
        )

        total_samples = 10
        expected_train_size = 7
        expected_test_size = 3

        assert len(train_indices) == expected_train_size
        assert len(test_indices) == expected_test_size
        assert len(train_indices) + len(test_indices) == total_samples

        train_set = set(train_indices)
        test_set = set(test_indices)
        assert not train_set.intersection(test_set)
        assert len(train_set) == len(train_indices)
        assert len(test_set) == len(test_indices)

    def test_reproducibility(self):
        """Test that same seed produces same results."""
        label_indices = {
            0: set(["a", "b", "c", "d"]),
            1: set(["b", "c", "d", "e"]),
            2: set(["c", "d", "e", "f"]),
            3: set(["a", "b", "c", "d"]),
            4: set(["b", "c", "d", "e"]),
            5: set(["c", "d", "e", "f"]),
            6: set(["a", "b", "c", "d"]),
            7: set(["b", "c", "d", "e"]),
            8: set(["c", "d", "e", "f"]),
            9: set(["a", "b", "c", "d"]),
        }

        train1, test1 = multilabel_iterative_train_test_split(
            label_indices, test_size=0.7, random_seed=42
        )
        train2, test2 = multilabel_iterative_train_test_split(
            label_indices, test_size=0.7, random_seed=42
        )

        assert train1 == train2
        assert test1 == test2

        train2, test2 = multilabel_iterative_train_test_split(
            label_indices, test_size=0.7, random_seed=123
        )
        assert train1 != train2
        assert test1 != test2
