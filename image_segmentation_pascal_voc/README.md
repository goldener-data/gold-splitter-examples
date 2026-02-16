# Pascal VOC Segmentation Split Comparison Experiment

This experiment compares two data splitting strategies for training
image segmentation models on the Pascal VOC 2012 dataset.

## Table of Contents

- [Main Components](#main-components)
- [Quick Start](#quick-start)
- [Technical Details](#technical-details)
- [Key Differences from Image Classification](#key-differences-from-image-classification)
- [Split Strategies](#split-strategies)
- [Viewing Results](#viewing-results)
- [Output Interpretation](#output-interpretation)
- [Extending the Experiment](#extending-the-experiment)
- [References](#references)


## Main Components

- **Configuration**: The experiment is configured from a config file loaded from Hydra for flexible configuration management.
It allows specifying the hyperparameters and logging parameters for the model training/evaluation
as well as the data split method to use and the settings for the GoldSplitter.

- **VOCSegmentationDataModule**: A specific Pytorch DataModule allowing to load data from the Pascal VOC 2012 dataset
(1,464 training images, 1,449 validation images). Depending on the configuration, only a subset of the
training images is used for training/validation. Duplication of some samples is as well possible.

- **VOCSegmentationLightningModule**: A specific Pytorch Lightning LightningModule allowing to train and evaluate different
image segmentation models (U-Net, ViT-based segmentation) for the Pascal VOC dataset.

- **Trainer**: PyTorch Lightning Trainer for efficient training management allowing to handle training, validation and
testing loops. It checkpoints the best model based on validation IoU metric.

- **Logging**: MLFlow for experiment tracking allowing to compare the different splitting strategies based on the logged metrics.

## Quick Start

```bash
# Install dependencies (from repo root)
uv sync --extra vision

# Make sure you're in the experiment directory
cd image_segmentation_pascal_voc

# Run both split methods (uses default config)
uv run python voc_experiment.py

# View results
mlflow ui
```

Then navigate to `http://localhost:5000` in your browser.


## Technical Details

### Dataset: Pascal VOC 2012

- **Task**: Semantic segmentation
- **Classes**: 21 classes (20 object categories + background)
  - Background, aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow,
    dining table, dog, horse, motorbike, person, potted plant, sheep, sofa, train, tv/monitor
- **Training samples**: 1,464 images with segmentation masks
- **Validation samples**: 1,449 images with segmentation masks
- **Image size**: Variable, resized to 224×224 for training


### Training Configuration

- **Optimizer**: Adam
  - Learning rate: 0.001 (default, configurable)
  - Weight decay: 0 (no L2 regularization)

- **Loss Function**: CrossEntropyLoss
  - Applied to pixel-wise predictions
  - Ignore index: 255 (void/unlabeled pixels)

- **Batch Size**: 16 (default, configurable)
- **Max Epochs**: 30 (default, configurable)

### Evaluation Metrics

**Primary Metric (for model selection)**:
- **IoU (Intersection over Union)**: Also known as Jaccard Index, measures the overlap between predicted and ground truth segmentation masks
  - Task: Multiclass segmentation
  - Used for: Model checkpoint selection (best validation IoU)
  - Formula: IoU = |A ∩ B| / |A ∪ B|

**Secondary Metrics**:
- **Pixel Accuracy**: Percentage of correctly classified pixels

All metrics are computed and logged for both training and validation sets at each epoch.

### Model Selection

The best model is selected based on **maximum validation IoU**:

```python
checkpoint_callback = ModelCheckpoint(
    monitor='val_iou',
    mode='max',
    save_top_k=1
)
```

## Key Differences from Image Classification

This experiment extends the image classification experiment with several key differences:

### 1. Task Type
- **Classification**: Assign a single label to an entire image
- **Segmentation**: Assign a label to each pixel in the image

### 2. Dataset
- **Classification**: CIFAR-10 (50,000 training images, 10 classes)
- **Segmentation**: Pascal VOC 2012 (1,464 training images, 21 classes)

### 3. Gold Splitting Approach

**Image Classification (CIFAR-10)**:
- Uses the **class token** from ViT (Vision Transformer)
- The class token is a single embedding that represents the entire image
- Extracted from `FilterLocation.START` (first token in the sequence)

```python
# Classification approach
vectorizer=TensorVectorizer(
    keep=Filter2DWithCount(keep=True, filter_location=FilterLocation.START),
    channel_pos=2,
)
```

**Image Segmentation (Pascal VOC)**:
- Uses **patch embeddings** corresponding to the segmentation mask
- Instead of a single token, we use all patch embeddings from ViT
- Extracted from `FilterLocation.ALL` (all patch tokens, excluding class token)
- This allows the splitting to be based on the spatial information relevant to segmentation

```python
# Segmentation approach
vectorizer=TensorVectorizer(
    keep=Filter2DWithCount(keep=True, filter_location=FilterLocation.ALL),
    channel_pos=2,
)
```

This difference is crucial because:
- **Segmentation requires spatial information**: We need features that capture local patterns across the image
- **Multiple regions per image**: A single image can contain multiple objects, each contributing to the split decision
- **Mask-aware splitting**: The patch embeddings naturally align with the segmentation masks, making the split more representative of the segmentation task

### 4. Model Architecture
- **Classification**: ResNet, ViT with classification head
- **Segmentation**: U-Net, ViT with segmentation head

### 5. Evaluation Metric
- **Classification**: AUROC (Area Under ROC Curve)
- **Segmentation**: IoU (Intersection over Union)

## Split Strategies

### 1. Random Split (Baseline)

```python
from sklearn.model_selection import train_test_split

train_indices, val_indices = train_test_split(
    range(len(dataset)),
    test_size=0.3,
    random_state=42,
    shuffle=True,
)
```

**Characteristics**:
- Uniform probability for each sample
- Simple random selection
- Standard practice in ML
- Fast and straightforward

### 2. GoldSplitter (Smart Split)

The smart split is done from the **patch embeddings** of the Dinov3 ViT-S model that correspond to the segmentation mask.

```python
from image_segmentation_pascal_voc.utils import get_gold_splitter

gold_splitter = get_gold_splitter(cfg.gold_splitter)
split_table = gold_splitter.split_in_table(dataset)
splits = gold_splitter.get_split_indices(
    split_table, selection_key="selected", idx_key="idx"
)
train_indices = list(splits["train"])
val_indices = list(splits["val"])
```

**Characteristics**:
- Uses patch-level features from segmentation masks
- Considers spatial distribution of objects
- Aims for optimal distribution of diverse segmentation patterns
- May lead to more representative validation sets
- Potentially better generalization for segmentation tasks

**Technical Implementation**:
- Extracts features from ViT's intermediate layer (blocks.11)
- Uses ALL patch tokens (excluding class token) to capture spatial information
- Each image contributes multiple feature vectors (one per patch)
- Splitting is based on these patch-level representations

### Evaluation Criteria

Compare the two methods on:
- **Convergence Speed**: Epochs to reach best performance
- **Stability**: Variance in validation metrics across epochs
- **Test Performance**: Final IoU on held-out test set
- **Segmentation Quality**: Visual quality of predicted masks

## Viewing Results

### MLFlow UI

After running the experiment, start the MLFlow UI:
```bash
mlflow ui
```

Then open `http://localhost:5000` in your browser to compare results between split methods.

## Output Interpretation

### Key Metrics to Compare

1. **Validation IoU**: Higher is better
   - Target: > 0.5 for reasonable segmentation
   - Good: > 0.6
   - Excellent: > 0.7

2. **Pixel Accuracy**: Higher is better
   - Baseline: ~0.7 (simple class imbalance)
   - Good: > 0.85
   - Excellent: > 0.90

3. **Convergence**: Fewer epochs to reach plateau is better

### Questions to Answer

- Does GoldSplitter lead to faster convergence?
- Is the validation performance more stable with GoldSplitter?
- Does GoldSplitter result in better test set performance?
- Are there differences in per-class IoU between methods?

## Extending the Experiment

### Adding New Models

Edit `model.py` to add new segmentation architectures:

```python
elif model_type == "deeplabv3":
    self.model = torchvision.models.segmentation.deeplabv3_resnet50(
        num_classes=self.num_classes
    )
```

### Modifying Split Strategy

Edit `utils.py` to change the vectorization approach:

```python
# Example: Use different ViT layer
extractor = TorchGoldFeatureExtractor(
    TorchGoldFeatureExtractorConfig(
        model=timm.create_model(...),
        layers=["blocks.8"],  # Earlier layer
    )
)
```

### Adjusting Hyperparameters

Edit `config/config.yaml` or use command line overrides:

```bash
python voc_experiment.py exp.batch_size=32 exp.learning_rate=0.0001
```

## Dependencies

Dependencies are managed at the repository root level in `pyproject.toml`.

Key dependencies for this experiment:
- [Torchvision](https://pytorch.org/vision/stable/index.html): Pascal VOC dataset and transforms
- [timm](https://github.com/huggingface/pytorch-image-models): Pre-trained vision models
- [Pillow](https://python-pillow.org/): Image loading
- [TorchMetrics](https://torchmetrics.readthedocs.io/): Jaccard Index (IoU) computation

## References

- [Pascal VOC Dataset](http://host.robots.ox.ac.uk/pascal/VOC/)
- [Goldener Library](https://github.com/goldener-data/goldener)
- [U-Net Paper](https://arxiv.org/abs/1505.04597)
- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)
