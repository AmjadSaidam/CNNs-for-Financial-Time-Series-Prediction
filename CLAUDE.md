# CNNs for Financial Time Series Prediction

## Project Overview

Applies convolutional neural networks (CNNs) to asset price prediction by converting financial time series into 2D images using the **Gramian Angular Field (GAF)** technique, then classifying next-day price direction (up/down) as a binary problem.

Reference paper: [paper/Convolutional_Neural_Networks_for_Asset_Price_Prediction.pdf](paper/Convolutional_Neural_Networks_for_Asset_Price_Prediction.pdf)

## Key Idea

Price windows → normalize to `[-1, 1]` → arccos transform → GAF matrix (cos(φ_i + φ_j)) → 52×52 grayscale image → CNN classifies direction (0 = down, 1 = up)

## Project Structure

```
code/
  cnn.py                  # CNN model, training, evaluation, dataset class
  data_pre_process.py     # GAF preprocessing pipeline
  model_application.ipynb # End-to-end notebook: data → EDA → train → backtest
paper/
  Convolutional_Neural_Networks_for_Asset_Price_Prediction.pdf
```

## Core Classes

### [code/data_pre_process.py](code/data_pre_process.py) — `PreProcess`
- `pre_process(batches)` → dict with keys: `norm_data`, `labels`, `gram`, `original_data`
- Input: pandas DataFrame with columns `date`, `high`, `low`, `adj_close`
- Labels: `1` if price goes down next day, `0` if up (mapped from sign of return)
- GAF computed via `__gram_matrix`: `G_ij = cos(arccos(x_i) + arccos(x_j))`

### [code/cnn.py](code/cnn.py) — `CNN_model`, `CustomImageDataset`, `CNN`
- `CNN_model`: two conv+pool blocks (Conv2d → MaxPool2d), then a configurable FCN; outputs 2 logits
- `CustomImageDataset`: wraps GAF images + labels for PyTorch DataLoader; images are `(1, H, W)` float32
- `CNN`: wrapper with `CNN_train(loader, epochs, lr, comet)` and `CNN_evaluate(loader)` returning `(loss, accuracy, signals)`
- Signals from `CNN_evaluate` are in `{-1, 1}` space (mapped from `{0, 1}` predictions)

## Data

Google stock data downloaded from Kaggle via `kagglehub`:
- Dataset: `umerhaddii/google-stock-data-2024`
- File: `GOOG_2004-08-19_2025-08-20.csv`

## Workflow (notebook)

1. Download data via `kagglehub`
2. `PreProcess.pre_process(batches=100)` → GAF images of shape `(52, 52)` each
3. 70/30 train/test split
4. `CustomImageDataset` + `DataLoader` with `batch_size=1`
5. Grid search over conv kernel sizes `[2,3,4]` and pool kernel sizes `[2,3,4]`; 10 epochs each
6. Visualize CNN feature maps per layer (`conv1`, `pool1`, `conv2`, `pool2`)
7. Backtest with `quantstats` — strategy signals vs. buy-and-hold benchmark

## Key Dependencies

```
torch, pandas, numpy, comet_ml, kagglehub, sklearn, quantstats, seaborn, matplotlib, tqdm
```

## Experiment Tracking

Uses **comet_ml** (call `comet_ml.login()` before running). Pass `comet=True` to `CNN_train` to log per-step loss. Project name: `gaf-cnn`.

## Running

Open and run [code/model_application.ipynb](code/model_application.ipynb) from the `code/` directory so that relative imports (`import cnn`, `from data_pre_process import PreProcess`) resolve correctly.

## Model Architecture (defaults)

| Layer   | Config                              |
|---------|-------------------------------------|
| conv1   | Conv2d(1→24, kernel=3)              |
| pool1   | MaxPool2d(kernel=4)                 |
| conv2   | Conv2d(24→36, kernel=3)             |
| pool2   | MaxPool2d(kernel=4)                 |
| flatten | —                                   |
| fc      | LazyLinear→ReLU → 2×(Linear→ReLU) → Linear(→2) |

Output: 2 logits fed into CrossEntropyLoss during training, Softmax+argmax during evaluation.