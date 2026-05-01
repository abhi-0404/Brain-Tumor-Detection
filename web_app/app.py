"""Flask app for brain tumor MRI image classification."""

from __future__ import annotations

import logging
import os
import traceback
from pathlib import Path
from tempfile import NamedTemporaryFile

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from brain_tumor_detection.config import DEFAULT_ARTIFACT_DIR, VALID_IMAGE_SUFFIXES
from brain_tumor_detection.predict import predict_image

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", DEFAULT_ARTIFACT_DIR / "best_model.keras"))
CLASS_INDICES_PATH = Path(os.environ.get("CLASS_INDICES_PATH", DEFAULT_ARTIFACT_DIR / "class_indices.json"))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in VALID_IMAGE_SUFFIXES


@app.route("/", methods=["GET", "POST"])
def index():
    try:
        result = None
        error = None
        model_ready = MODEL_PATH.exists() and CLASS_INDICES_PATH.exists()
        
        logger.debug(f"Model ready: {model_ready}, MODEL_PATH: {MODEL_PATH}, CLASS_INDICES_PATH: {CLASS_INDICES_PATH}")

        if request.method == "POST":
            uploaded_file = request.files.get("image")
            if uploaded_file is None or uploaded_file.filename == "":
                error = "Please choose an MRI image before submitting."
            elif not allowed_file(uploaded_file.filename):
                error = "Unsupported file type. Please upload a JPG, JPEG, PNG, or BMP image."
            elif not model_ready:
                error = "Model artifacts are missing. Train the model first to create artifacts/best_model.keras and artifacts/class_indices.json."
            else:
                suffix = Path(secure_filename(uploaded_file.filename)).suffix.lower()
                with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    uploaded_file.save(temp_file.name)
                    temp_path = Path(temp_file.name)

                try:
                    result = predict_image(MODEL_PATH, temp_path, CLASS_INDICES_PATH)
                    result["confidence_percent"] = round(result["confidence"] * 100, 2)
                    result["probability_items"] = sorted(
                        [
                            {"label": label, "percent": round(probability * 100, 2)}
                            for label, probability in result["probabilities"].items()
                        ],
                        key=lambda item: item["percent"],
                        reverse=True,
                    )
                finally:
                    temp_path.unlink(missing_ok=True)

        logger.debug(f"Rendering template with result={result}, error={error}, model_ready={model_ready}")
        
        return render_template(
            "index.html",
            result=result,
            error=error,
            model_ready=model_ready,
            model_path=str(MODEL_PATH),
            class_indices_path=str(CLASS_INDICES_PATH),
        )
    except Exception as e:
        logger.error(f"Error in index route: {e}")
        logger.error(traceback.format_exc())
        return f"Error: {str(e)}\n\n{traceback.format_exc()}", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
