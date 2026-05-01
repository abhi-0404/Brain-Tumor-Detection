"""
Download the Brain Tumor MRI dataset from Kaggle using kagglehub
and place it where the training script expects it.

Usage:
    python download_dataset.py

Requirements:
    pip install kagglehub

Kaggle authentication:
    Either set KAGGLE_USERNAME and KAGGLE_KEY environment variables,
    or place your kaggle.json at ~/.kaggle/kaggle.json
    Get your API key from: https://www.kaggle.com/settings -> API -> Create New Token
"""

from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub

# Where the training script expects the data
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "Dataset"
TRAINING_DIR = DATASET_DIR / "Training"
TESTING_DIR = DATASET_DIR / "Testing"


def main() -> None:
    # Skip if already in place
    if TRAINING_DIR.exists() and TESTING_DIR.exists():
        print(f"Dataset already exists at: {DATASET_DIR}")
        print("Delete Dataset/Training and Dataset/Testing to re-download.")
        return

    print("Downloading dataset from Kaggle...")
    kaggle_path = Path(kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset"))
    print(f"Downloaded to: {kaggle_path}")

    # kagglehub downloads to a cache folder. The dataset contains
    # Training/ and Testing/ either directly or one level deep.
    training_src = _find_folder(kaggle_path, "Training")
    testing_src = _find_folder(kaggle_path, "Testing")

    if training_src is None or testing_src is None:
        print(f"\nCould not auto-detect Training/Testing folders inside: {kaggle_path}")
        print("Please copy them manually:")
        print(f"  {kaggle_path}  ->  Dataset/")
        return

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Copying Training/ -> {TRAINING_DIR}")
    shutil.copytree(training_src, TRAINING_DIR)

    print(f"Copying Testing/  -> {TESTING_DIR}")
    shutil.copytree(testing_src, TESTING_DIR)

    print("\nDataset ready. You can now run:")
    print("  python -m brain_tumor_detection.train")


def _find_folder(root: Path, name: str) -> Path | None:
    """Search up to 2 levels deep for a folder with the given name."""
    # Check root itself
    candidate = root / name
    if candidate.is_dir():
        return candidate
    # Check one level deeper
    for child in root.iterdir():
        if child.is_dir():
            candidate = child / name
            if candidate.is_dir():
                return candidate
    return None


if __name__ == "__main__":
    main()
