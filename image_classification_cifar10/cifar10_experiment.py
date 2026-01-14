"""
CIFAR-10 Split Comparison Experiment

This script compares two data splitting strategies:
1. Random split from scikit-learn
2. Smart split using GoldSplitter from the Goldener library

The experiment uses:
- CIFAR-10 dataset from torchvision
- Simple convolutional neural network
- PyTorch Lightning for training management
- MLFlow for experiment tracking
- AUROC metric on validation set for model selection
"""

import os
from logging import getLogger, WARNING

import hydra
from omegaconf import DictConfig
import pixeltable as pxt
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger


from image_classification_cifar10.data import CIFAR10DataModule
from image_classification_cifar10.model import Cifar10DinoV3ViTSmall


logger = getLogger(__name__)

pxt.configure_logging(to_stdout=True, level=WARNING, remove="goldener")


def run_experiment(
    cfg: DictConfig,
    data_module: CIFAR10DataModule,
    split_method: str = "random",
) -> dict:
    model = Cifar10DinoV3ViTSmall(learning_rate=cfg.learning_rate)

    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.mlflow_experiment_name,
        tracking_uri=cfg.mlflow_tracking_uri,
        run_name=f"{cfg.mlflow_run_name}_{split_method}",
    )

    # Log additional parameters
    mlflow_logger.log_hyperparams(
        {
            "split_method": split_method,
            "train_ratio": cfg.train_ratio,
            "val_ratio": cfg.val_ratio,
            "max_epochs": cfg.max_epochs,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "random_state": cfg.random_state,
            "random_split_state": cfg.random_split_state,
            "vectorized_path": data_module.gold_splitter.descriptor.table_path,
        }
    )

    mlflow_logger.experiment.log_dict(
        {
            "gold_train_indices": data_module.gold_train_indices,
            "gold_val_indices": data_module.gold_val_indices,
            "sk_train_indices": data_module.sk_train_indices,
            "sk_val_indices": data_module.sk_val_indices,
        },
        "indices.json",
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_auroc",
        dirpath=f"./checkpoints/cifar10_{split_method}_split",
        filename="cifar10-{epoch:02d}-{val_auroc:.4f}",
        save_top_k=1,
        mode="max",
        verbose=True,
    )

    # Initialize trainer
    debug_train_count = cfg.debug_train_count
    debug_count = cfg.debug_count
    trainer = Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="auto",
        devices=1,
        logger=mlflow_logger,
        callbacks=[checkpoint_callback],
        deterministic=False,
        log_every_n_steps=50,
        limit_train_batches=debug_train_count if debug_train_count is not None else 1.0,
        limit_val_batches=debug_count if debug_count is not None else 1.0,
        limit_test_batches=debug_count if debug_count is not None else 1.0,
    )

    # Train the model
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Training with {split_method.upper()} split method")
    logger.info(f"{'=' * 60}\n")

    train_dataloader = (
        data_module.sk_train_dataloader()
        if split_method == "random"
        else data_module.gold_train_dataloader()
    )
    val_dataloader = (
        data_module.sk_val_dataloader()
        if split_method == "random"
        else data_module.gold_val_dataloader()
    )
    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )

    # Test the model
    data_module.setup(stage="test")

    test_results = trainer.test(model, data_module, ckpt_path="best")

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Training completed for {split_method.upper()} split method")
    logger.info(f"Best validation AUROC: {checkpoint_callback.best_model_score:.4f}")
    logger.info(f"{'=' * 60}\n")

    best_val_auroc = checkpoint_callback.best_model_score
    assert best_val_auroc is not None
    return {
        "split_method": split_method,
        "best_val_auroc": best_val_auroc.item(),
        "test_results": test_results[0]["test_auroc"],
    }


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    """
    Main function to run experiments with different split strategies
    Uses Hydra for configuration management
    """
    # Print configuration
    logger.info("Starting CIFAR-10 Split Comparison Experiment")
    logger.info(f"Configuration:\n{cfg}")

    # Create necessary directories
    os.makedirs(cfg.data_dir, exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)

    results = []

    seed_everything(cfg.random_state, workers=True)

    data_module = CIFAR10DataModule(
        cfg=cfg,
    )
    data_module.prepare_data()
    data_module.setup(stage="fit")

    # Run experiments based on split method argument
    if cfg.split_method == "both":
        split_methods = [
            "gold",
            "random",
        ]
    else:
        split_methods = [cfg.split_method]

    for split_method in split_methods:
        result = run_experiment(
            split_method=split_method,
            data_module=data_module,
            cfg=cfg,
        )
        results.append(result)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 60)
    for result in results:
        logger.info(f"\n{result['split_method'].upper()} Split Method:")
        logger.info(f"  Best Validation AUROC: {result['best_val_auroc']:.4f}")
        logger.info(f"  Test AUROC: {result['test_results']:.4f}")

    if len(results) > 1:
        logger.info(
            f"\nAUROC Difference (Gold - Random): "
            f"{results[1]['test_results'] - results[0]['test_results']:.4f}"
        )

    logger.info("\n" + "=" * 60)


if __name__ == "__main__":
    main()
