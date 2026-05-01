# Brain Tumor Detection

A deep learning project that classifies brain MRI images into four categories:

| Class | Description |
|---|---|
| `glioma` | Glioma tumor |
| `meningioma` | Meningioma tumor |
| `pituitary` | Pituitary tumor |
| `notumor` | No tumor detected |

> **Medical disclaimer:** This project is for learning and research purposes only. It is not a diagnostic medical device and should not be used for clinical decisions.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Dataset Setup](#dataset-setup)
- [Installation](#installation)
- [Train a Model](#train-a-model)
- [Predict on an Image](#predict-on-an-image)
- [Run the Web App](#run-the-web-app)
- [Notebook](#notebook)
- [How It Works](#how-it-works)
- [Troubleshooting](#troubleshooting)
- [Acknowledgments](#acknowledgments)

---

## Project Structure

```text
Brain Tumor Detection/
├── Dataset/
│   ├── Training/          ← place dataset here (ignored by Git)
│   └── Testing/           ← place dataset here (ignored by Git)
├── Images/
├── Model/
│   └── brain_tumor_model.ipynb
├── artifacts/             ← generated after training (ignored by Git)
│   ├── best_model.keras
│   ├── class_indices.json
│   ├── metadata.json
│   └── metrics.json
├── src/
│   └── brain_tumor_detection/
│       ├── config.py
│       ├── data.py
│       ├── evaluate.py
│       ├── models.py
│       ├── predict.py
│       └── train.py
├── web_app/
│   ├── app.py
│   ├── static/
│   │   ├── css/styles.css
│   │   └── js/app.js
│   └── templates/
│       └── index.html
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Prerequisites

- **Python** 3.9, 3.10, or 3.11 (TensorFlow 2.16 does not support Python 3.12+ on all platforms)
- **pip** 23+
- A Kaggle account to download the dataset

---

## Dataset Setup

1. Download the dataset from Kaggle:
   [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

2. Extract and place the folders exactly like this:

```text
Dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

The `Dataset/Training` and `Dataset/Testing` folders are listed in `.gitignore` and will never be committed.

---

## Installation

### 1. Create and activate a virtual environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 3. Install the package

This installs the project as an editable package along with all dependencies:

```bash
python -m pip install -e .
```

Alternatively, if you only want to install raw dependencies without the package:

```bash
python -m pip install -r requirements.txt
```

### 4. (Optional) Install Jupyter for the notebook

```bash
python -m pip install -e ".[notebook]"
```

---

## Train a Model

Make sure the dataset is in place before running training.

### Basic training (default CNN, 20 epochs)

```bash
python -m brain_tumor_detection.train
```

### Full command with all options

```bash
python -m brain_tumor_detection.train \
  --dataset-dir Dataset \
  --output-dir artifacts \
  --model cnn \
  --epochs 20 \
  --batch-size 16 \
  --seed 42
```

### Available model architectures

| Value | Description |
|---|---|
| `cnn` | Custom CNN with batch norm and dropout (default) |
| `mlp` | Simple MLP baseline |
| `vgg16` | VGG16 with ImageNet weights (transfer learning) |

```bash
# Train with VGG16
python -m brain_tumor_detection.train --model vgg16 --epochs 20

# Train with MLP
python -m brain_tumor_detection.train --model mlp --epochs 20
```

### Training outputs

After training completes, the `artifacts/` folder will contain:

| File | Description |
|---|---|
| `best_model.keras` | Best checkpoint saved by validation loss |
| `class_indices.json` | Class-to-index mapping used during training |
| `metadata.json` | Model name, image size, class names, dataset path |
| `metrics.json` | Test accuracy, precision, recall, F1, confusion matrix |

---

## Predict on an Image

Requires a trained model in `artifacts/` first.

```bash
python -m brain_tumor_detection.predict "path/to/mri-image.jpg"
```

### With explicit paths

```bash
python -m brain_tumor_detection.predict "path/to/mri-image.jpg" \
  --model-path artifacts/best_model.keras \
  --class-indices artifacts/class_indices.json
```

### Example output

```json
{
  "image": "path/to/mri-image.jpg",
  "predicted_class": "glioma",
  "confidence": 0.9421,
  "probabilities": {
    "glioma": 0.9421,
    "meningioma": 0.0312,
    "notumor": 0.0047,
    "pituitary": 0.0220
  }
}
```

---

## Run the Web App

The web app lets you upload an MRI image through a browser and see the prediction result.

### 1. Train a model first

The app requires `artifacts/best_model.keras` and `artifacts/class_indices.json` to exist.

### 2. Start the Flask server

```bash
python web_app/app.py
```

### 3. Open in your browser

```
http://127.0.0.1:5000
```

Upload a `.jpg`, `.jpeg`, `.png`, or `.bmp` MRI image to get the predicted class and per-class confidence scores.

### Custom model paths (optional)

You can point the app at different artifact files using environment variables:

**Windows:**
```bash
set MODEL_PATH=artifacts\best_model.keras
set CLASS_INDICES_PATH=artifacts\class_indices.json
python web_app/app.py
```

**macOS / Linux:**
```bash
MODEL_PATH=artifacts/best_model.keras \
CLASS_INDICES_PATH=artifacts/class_indices.json \
python web_app/app.py
```

---

## Notebook

The original Jupyter notebook is at `Model/brain_tumor_model.ipynb`. It uses the same dataset layout and is useful for visualization and experimentation.

```bash
# Make sure Jupyter is installed
python -m pip install -e ".[notebook]"

# Launch
jupyter notebook Model/brain_tumor_model.ipynb
```

For repeatable, scriptable training, prefer the CLI (`src/brain_tumor_detection/train.py`) over the notebook.

---

## How It Works

| Module | Role |
|---|---|
| `config.py` | Shared constants: image size, batch size, paths, seed |
| `data.py` | Builds dataframes from class folders, creates Keras `ImageDataGenerator` objects for train/val/test |
| `models.py` | Three model builders: CNN, MLP, and VGG16 transfer learning |
| `train.py` | CLI entry point — loads data, trains with early stopping and checkpointing, saves artifacts |
| `evaluate.py` | Computes accuracy, precision, recall, F1, and confusion matrix on the test set |
| `predict.py` | Loads a saved model, preprocesses a single image, returns label and probabilities |
| `web_app/app.py` | Flask server — accepts image uploads and runs inference via `predict.py` |

### Default CNN architecture

- Convolution blocks for spatial feature extraction
- Batch normalization for training stability
- Dropout for regularization
- Global average pooling to reduce parameter count
- Softmax output for 4-class classification

### VGG16 transfer learning

- Pretrained on ImageNet
- Early convolutional layers are frozen
- Final layers and a custom classification head are trainable

---

## Troubleshooting

**`Dataset folder not found`**
Confirm that `Dataset/Training` and `Dataset/Testing` exist and contain the four class subfolders.

**`No supported image files found`**
Check that the class folders contain `.jpg`, `.jpeg`, `.png`, or `.bmp` files.

**`Model artifacts are missing`**
Run training before using the prediction CLI or the web app.

**`TensorFlow installation fails`**
Verify your Python version is 3.9, 3.10, or 3.11. Python 3.12+ may not be supported by TensorFlow 2.16 on all platforms.

**Web app shows "Model artifacts are missing"**
The app looks for `artifacts/best_model.keras` and `artifacts/class_indices.json` relative to the project root. Run training first, or set the `MODEL_PATH` and `CLASS_INDICES_PATH` environment variables to point at your files.

---

## Acknowledgments

Dataset: [Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

Related references:

- [TheNaiveSamosa GitHub](https://github.com/TheNaiveSamosa)
- [BMC Medical Informatics and Decision Making](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-023-02114-6)
- [Algorithms, 2023](https://www.mdpi.com/1999-4893/16/4/176)
- [PMC9468505](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9468505/)
- [PMC9854739](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9854739/)
