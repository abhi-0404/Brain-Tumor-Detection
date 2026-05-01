"""Prediction helpers and command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

from .config import DEFAULT_ARTIFACT_DIR, IMAGE_SIZE


def load_class_names(class_indices_path: Path) -> list[str]:
    """Load class labels ordered by their training index."""
    class_indices = json.loads(Path(class_indices_path).read_text(encoding="utf-8"))
    return [label for label, _ in sorted(class_indices.items(), key=lambda item: item[1])]


def predict_image(model_path: Path, image_path: Path, class_indices_path: Path, image_size: tuple[int, int] = IMAGE_SIZE) -> dict:
    """Predict the tumor class for a single MRI image."""
    model = tf.keras.models.load_model(model_path)
    class_names = load_class_names(class_indices_path)

    img = image.load_img(image_path, target_size=image_size, color_mode="rgb")
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    probabilities = model.predict(arr, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))

    return {
        "image": str(image_path),
        "predicted_class": class_names[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "probabilities": {class_names[i]: float(probabilities[i]) for i in range(len(class_names))},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a brain tumor class for one MRI image.")
    parser.add_argument("image_path", type=Path, help="Path to an MRI image.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_ARTIFACT_DIR / "best_model.keras")
    parser.add_argument("--class-indices", type=Path, default=DEFAULT_ARTIFACT_DIR / "class_indices.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = predict_image(args.model_path, args.image_path, args.class_indices)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
