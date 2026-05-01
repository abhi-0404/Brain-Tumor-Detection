"""Command-line training entry point."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from .config import BATCH_SIZE, CHANNELS, DEFAULT_ARTIFACT_DIR, DEFAULT_DATASET_DIR, IMAGE_SIZE, SEED
from .data import create_generators, load_dataset_frames
from .evaluate import evaluate_model, save_json
from .models import MODEL_BUILDERS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a brain tumor MRI classifier.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR, help="Folder containing Training and Testing directories.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACT_DIR, help="Folder for model and metric artifacts.")
    parser.add_argument("--model", choices=MODEL_BUILDERS.keys(), default="cnn", help="Model architecture to train.")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Images per batch.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    return parser.parse_args()


def set_reproducibility(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_callbacks(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        ModelCheckpoint(
            filepath=str(output_dir / "best_model.keras"),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    ]


def main() -> None:
    args = parse_args()
    set_reproducibility(args.seed)

    train_df, valid_df, test_df = load_dataset_frames(args.dataset_dir, seed=args.seed)
    train_gen, valid_gen, test_gen = create_generators(
        train_df,
        valid_df,
        test_df,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    class_names = list(train_gen.class_indices.keys())
    input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], CHANNELS)
    model = MODEL_BUILDERS[args.model](input_shape=input_shape, class_count=len(class_names))

    print(f"Training {args.model} model with {len(class_names)} classes: {class_names}")
    model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=valid_gen,
        callbacks=build_callbacks(args.output_dir),
        verbose=1,
    )

    best_model = tf.keras.models.load_model(args.output_dir / "best_model.keras")
    metrics = evaluate_model(best_model, test_gen, class_names)
    save_json(metrics, args.output_dir / "metrics.json")
    save_json(train_gen.class_indices, args.output_dir / "class_indices.json")

    metadata = {
        "model": args.model,
        "image_size": IMAGE_SIZE,
        "class_names": class_names,
        "dataset_dir": str(args.dataset_dir),
    }
    save_json(metadata, args.output_dir / "metadata.json")

    print(json.dumps({"test_accuracy": metrics["accuracy"], "test_f1_macro": metrics["f1_macro"]}, indent=2))


if __name__ == "__main__":
    main()

