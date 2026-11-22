"""Flask app for brain tumor MRI image classification."""

from __future__ import annotations

import base64
import logging
import os
import traceback
from pathlib import Path
from tempfile import NamedTemporaryFile

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from brain_tumor_detection.config import DEFAULT_ARTIFACT_DIR, VALID_IMAGE_SUFFIXES
from brain_tumor_detection.predict import predict_image

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

MODEL_PATH = Path(os.environ.get("MODEL_PATH", DEFAULT_ARTIFACT_DIR / "best_model.keras"))
CLASS_INDICES_PATH = Path(os.environ.get("CLASS_INDICES_PATH", DEFAULT_ARTIFACT_DIR / "class_indices.json"))

MIME_MAP = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".bmp": "image/bmp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in VALID_IMAGE_SUFFIXES


@app.route("/", methods=["GET", "POST"])
def index():
    try:
        result = None
        error = None
        uploaded_image_b64 = None
        uploaded_image_mime = None
        uploaded_filename = None
        model_ready = MODEL_PATH.exists() and CLASS_INDICES_PATH.exists()

        if request.method == "POST":
            uploaded_file = request.files.get("image")
            if uploaded_file is None or uploaded_file.filename == "":
                error = "Please choose an MRI image before submitting."
            elif not allowed_file(uploaded_file.filename):
                error = "Unsupported file type. Please upload a JPG, JPEG, PNG, or BMP image."
            elif not model_ready:
                error = "Model artifacts are missing. Train the model first."
            else:
                suffix = Path(secure_filename(uploaded_file.filename)).suffix.lower()
                uploaded_filename = uploaded_file.filename
                uploaded_image_mime = MIME_MAP.get(suffix, "image/jpeg")

                with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    uploaded_file.save(tmp.name)
                    temp_path = Path(tmp.name)

                try:
                    # Encode image as base64 so it persists through the page reload
                    uploaded_image_b64 = base64.b64encode(temp_path.read_bytes()).decode("utf-8")

                    result = predict_image(MODEL_PATH, temp_path, CLASS_INDICES_PATH)
                    result["confidence_percent"] = round(result["confidence"] * 100, 2)
                    result["probability_items"] = sorted(
                        [
                            {"label": label, "percent": round(prob * 100, 2)}
                            for label, prob in result["probabilities"].items()
                        ],
                        key=lambda item: item["percent"],
                        reverse=True,
                    )
                finally:
                    temp_path.unlink(missing_ok=True)

        return render_template(
            "index.html",
            result=result,
            error=error,
            model_ready=model_ready,
            model_name=MODEL_PATH.name,
            class_indices_name=CLASS_INDICES_PATH.name,
            uploaded_image_b64=uploaded_image_b64,
            uploaded_image_mime=uploaded_image_mime,
            uploaded_filename=uploaded_filename,
        )
    except Exception as e:
        logger.error(traceback.format_exc())
        return f"Error: {str(e)}\n\n{traceback.format_exc()}", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
