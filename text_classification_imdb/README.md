# Text Classification: IMDb Movie Reviews

This example compares two data splitting strategies for binary sentiment classification on the
[IMDb Large Movie Review Dataset](https://huggingface.co/datasets/imdb):

1. **Random Split**: Traditional stratified random split using scikit-learn.
2. **Smart Split**: Intelligent split using GoldSplitter from the Goldener library.

## Models

| Name | Architecture | Description |
|------|-------------|-------------|
| `cnn` | 1D CNN | Embedding layer → parallel Conv1D with kernel sizes 3/4/5 → adaptive max-pool → dropout → linear |
| `bert` | BERT-Base | `bert-base-uncased` with a linear classification head on the CLS token |

Both models share the same **WordPiece** tokenizer (`bert-base-uncased`).

## Metrics

- **BinaryAUROC** – area under the ROC curve for binary classification.
- **Accuracy** – fraction of correctly classified reviews.

## Quick Start

```bash
# Install dependencies (from repo root)
uv sync --extra text

# Run experiment (defaults to CNN with GoldSplitter)
cd text_classification_imdb
uv run python imdb_experiment.py

# Run with BERT model
uv run python imdb_experiment.py exp.model=bert exp.learning_rate=2e-5

# Compare both split strategies with both models
uv run python imdb_experiment.py exp.split_method=all exp.model=cnn
uv run python imdb_experiment.py exp.split_method=all exp.model=bert exp.learning_rate=2e-5
```

## Configuration

The experiment is configured via [Hydra](https://hydra.cc). Key parameters in `config/config.yaml`:

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `exp` | `model` | `cnn` | Model architecture (`cnn` or `bert`) |
| `exp` | `split_method` | `gold` | Split strategy (`random`, `gold`, or `all`) |
| `exp` | `val_ratio` | `0.2` | Fraction of training data used for validation |
| `exp` | `max_epochs` | `10` | Number of training epochs |
| `exp` | `learning_rate` | `0.001` | Optimizer learning rate |
| `data` | `tokenizer_name` | `bert-base-uncased` | HuggingFace tokenizer (WordPiece) |
| `data` | `max_length` | `256` | Maximum token sequence length |
| `gold_splitter` | `n_clusters` | `50` | Number of k-means clusters in GoldSplitter |
| `gold_splitter` | `pretrained_model` | `bert-base-uncased` | BERT model used for feature extraction |

## Feature Extraction for GoldSplitter

GoldSplitter needs dense vector representations to measure dataset coverage.
For this text example, each review is embedded by passing the tokenized text through a
frozen `bert-base-uncased` model and extracting the **CLS token** hidden state from the
last encoder layer. This produces a 768-dimensional feature vector per sample.

## Run Scripts

Pre-built shell scripts are provided in the `run/` directory:

```bash
# CNN model – multiple random-split seeds for statistical comparison
bash run/imdb_cnn.sh

# BERT model – multiple random-split seeds for statistical comparison
bash run/imdb_bert.sh
```

## Experiment Tracking

Results are logged with [MLflow](https://mlflow.org).  After running experiments, launch the UI:

```bash
mlflow ui --backend-store-uri mlruns
```
