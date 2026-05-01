"""Dataset loading and Keras image generator helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from .config import BATCH_SIZE, IMAGE_SIZE, SEED, VALID_IMAGE_SUFFIXES


def build_dataframe(split_dir: Path) -> pd.DataFrame:
    """Return a dataframe with image file paths and labels from a class-folder split."""
    split_dir = Path(split_dir)
    if not split_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {split_dir}")

    records = []
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in VALID_IMAGE_SUFFIXES:
                records.append({"filepaths": str(image_path), "labels": class_dir.name})

    if not records:
        raise ValueError(f"No supported image files found in {split_dir}")

    return pd.DataFrame(records)


def load_dataset_frames(dataset_dir: Path, validation_fraction: float = 0.5, seed: int = SEED) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load Training and Testing folders, then split Testing into validation and test sets."""
    dataset_dir = Path(dataset_dir)
    train_df = build_dataframe(dataset_dir / "Training")
    holdout_df = build_dataframe(dataset_dir / "Testing")

    valid_df, test_df = train_test_split(
        holdout_df,
        train_size=validation_fraction,
        shuffle=True,
        random_state=seed,
        stratify=holdout_df["labels"],
    )

    return (
        train_df.reset_index(drop=True),
        valid_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def create_generators(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    image_size: tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
):
    """Create train, validation, and test generators from prepared dataframes."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.12,
        brightness_range=(0.9, 1.1),
        horizontal_flip=True,
    )
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255)

    common_args = {
        "x_col": "filepaths",
        "y_col": "labels",
        "target_size": image_size,
        "class_mode": "categorical",
        "color_mode": "rgb",
        "batch_size": batch_size,
    }

    train_gen = train_datagen.flow_from_dataframe(train_df, shuffle=True, seed=seed, **common_args)
    valid_gen = eval_datagen.flow_from_dataframe(valid_df, shuffle=False, seed=seed, **common_args)
    test_gen = eval_datagen.flow_from_dataframe(test_df, shuffle=False, seed=seed, **common_args)

    return train_gen, valid_gen, test_gen

