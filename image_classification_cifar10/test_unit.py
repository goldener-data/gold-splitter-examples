import pytest


from image_classification_cifar10.data import GoldCifar10

GoldCifar10(".data", download=True)


class TestGoldCifar10:
    def test_simple_usage(self):
        dataset = GoldCifar10(".data")
        assert len(dataset) == 50000
        for class_name in range(10):
            class_indices = [idx for idx in dataset.targets if idx == class_name]
            assert len(class_indices) == 5000

    def test_with_remove_ratio(self):
        remove_ratio = 0.2
        first_dataset = GoldCifar10(".data", remove_ratio=remove_ratio)
        expected_length = int(50000 * (1 - remove_ratio))
        assert len(first_dataset) == expected_length
        first_indices_per_class = {}
        for class_name in range(10):
            first_indices_per_class[class_name] = [
                idx for idx in first_dataset.targets if idx == class_name
            ]
            expected_class_count = int(5000 * (1 - remove_ratio))
            assert len(first_indices_per_class[class_name]) == expected_class_count

        second_dataset = GoldCifar10(".data", remove_ratio=remove_ratio)
        second_indices_per_class = {}
        for class_name in range(10):
            second_indices_per_class[class_name] = [
                idx for idx in second_dataset.targets if idx == class_name
            ]

        for class_name in range(10):
            assert (
                first_indices_per_class[class_name]
                == second_indices_per_class[class_name]
            )

    def test_with_duplicate_failure(self):
        duplicate_table_path = "unit_test_description_table_failure"
        drop_duplicate_table = True
        to_duplicate_clusters = 3
        cluster_count = 2
        duplicate_per_sample = None
        with pytest.raises(ValueError):
            GoldCifar10(
                ".data",
                remove_ratio=0.98,
                duplicate_table_path=duplicate_table_path,
                drop_duplicate_table=drop_duplicate_table,
                to_duplicate_clusters=to_duplicate_clusters,
                cluster_count=cluster_count,
                duplicate_per_sample=duplicate_per_sample,
            )

    def test_with_duplicate(self):
        duplicate_table_path = "unit_test_description_table"
        drop_duplicate_table = True
        to_duplicate_clusters = 2
        cluster_count = 2
        duplicate_per_sample = 2

        first_dataset = GoldCifar10(
            ".data",
            remove_ratio=0.98,
            duplicate_table_path=duplicate_table_path,
            drop_duplicate_table=drop_duplicate_table,
            to_duplicate_clusters=to_duplicate_clusters,
            cluster_count=cluster_count,
            duplicate_per_sample=duplicate_per_sample,
        )
        first_indices_per_class = {}
        for class_name in range(10):
            first_indices_per_class[class_name] = [
                idx for idx in first_dataset.targets if idx == class_name
            ]

        second_dataset = GoldCifar10(
            ".data",
            remove_ratio=0.98,
            duplicate_table_path=duplicate_table_path,
            drop_duplicate_table=drop_duplicate_table,
            to_duplicate_clusters=to_duplicate_clusters,
            cluster_count=cluster_count,
            duplicate_per_sample=duplicate_per_sample,
        )
        second_indices_per_class = {}
        for class_name in range(10):
            second_indices_per_class[class_name] = [
                idx for idx in second_dataset.targets if idx == class_name
            ]

        for class_name in range(10):
            assert (
                first_indices_per_class[class_name]
                == second_indices_per_class[class_name]
            )
