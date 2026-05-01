"""Evaluation helpers for trained models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score


def evaluate_model(model, test_gen, class_names: list[str]) -> dict:
    """Evaluate a Keras model and return common classification metrics."""
    loss, accuracy = model.evaluate(test_gen, verbose=1)
    probabilities = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)
    y_true = test_gen.classes

    return {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "sklearn_accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, target_names=class_names, digits=4, output_dict=True, zero_division=0),
    }


def save_json(data: dict, output_path: Path) -> None:
    """Write JSON with readable indentation."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

