"""Shared project configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "Dataset"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

IMAGE_SIZE = (224, 224)
CHANNELS = 3
BATCH_SIZE = 16
SEED = 42
VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

