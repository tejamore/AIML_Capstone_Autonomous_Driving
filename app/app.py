"""
app.py
------
Flask web application for the AIML Capstone: Autonomous Driving.

Two capabilities matching the capstone brief:
  Part 1 - Object Detection: upload a road image, the app localizes and
           classifies vehicles (car / bus / motorbike / bicycle / train)
           with bounding boxes using a MobileNet-SSD deep learning model.

  Part 2 - Data Science: an analytics dashboard answering the EDA
           questions about Tesla Autopilot deaths and road safety.

Run locally:    python app.py
Run on Render: gunicorn app:app --bind 0.0.0.0:$PORT
"""

import base64
import logging
import os
import traceback
from typing import Tuple

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

# Delay importing heavy native packages (cv2, numpy) and the detector module
# until needed to avoid import-time worker crashes on memory-constrained hosts.
from analysis import build_dashboard, load_data

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload limit

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}

# Configure structured logging
logging.basicConfig(level=logging.INFO)


def _allowed_file(filename: str) -> bool:
    """Check if the filename has a supported image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def home():
    """Render application home page."""
    return render_template("index.html")


@app.route("/detect", methods=["GET", "POST"])
def detect():
    """Handle vehicle detection uploads and render results."""
    if request.method == "GET":
        return render_template("detect.html", result_image=None, detections=None)

    # Lazily import heavy packages to guard app initialization
    try:
        import cv2
        import numpy as np
        from detector import detect_vehicles
    except ImportError as exc:
        app.logger.exception("Detector import failed: %s", exc)
        return render_template(
            "detect.html",
            error=(
                f"Detector dependencies are missing: {exc}. "
                "Ensure OpenCV and NumPy are installed."
            ),
            result_image=None,
            detections=None,
        )

    # 1. Check if image parameter is present in request
    if "image" not in request.files:
        return render_template(
            "detect.html",
            error="Please choose an image file first.",
            result_image=None,
            detections=None,
        )

    file = request.files["image"]

    # 2. Check for empty submission
    if not file or file.filename == "":
        return render_template(
            "detect.html",
            error="No file selected. Please upload an image.",
            result_image=None,
            detections=None,
        )

    # 3. Validate file extension & sanitize filename
    filename = secure_filename(file.filename)
    if not _allowed_file(filename):
        return render_template(
            "detect.html",
            error="Unsupported file type. Allowed formats: PNG, JPG, JPEG, BMP, WEBP.",
            result_image=None,
            detections=None,
        )

    try:
        # Reset stream pointer to ensure complete read
        file.stream.seek(0)
        file_bytes = np.frombuffer(file.read(), np.uint8)
        
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError("Failed to decode image file. File may be corrupted.")

        # Run detection pipeline
        annotated, detections, inference_ms = detect_vehicles(image_bgr)

        # Encode resulting frame to base64 JPEG
        success, buffer = cv2.imencode(".jpg", annotated)
        if not success:
            raise ValueError("Failed to encode annotated output image.")

        encoded = base64.b64encode(buffer).decode("utf-8")

        return render_template(
            "detect.html",
            result_image=encoded,
            detections=detections,
            inference_ms=round(inference_ms, 1),
            error=None,
        )

    except Exception as exc:
        app.logger.error("Detection failed: %s\n%s", exc, traceback.format_exc())
        return render_template(
            "detect.html",
            error=f"Processing error: {exc}",
            result_image=None,
            detections=None,
        )


@app.route("/analytics")
def analytics():
    """Render EDA Analytics Dashboard."""
    try:
        df = load_data()
        dashboard = build_dashboard(df)
        return render_template(
            "analytics.html",
            charts=dashboard.get("charts", {}),
            stats=dashboard.get("stats", {}),
            error=None,
        )
    except Exception as exc:
        app.logger.error("Analytics failed: %s\n%s", exc, traceback.format_exc())
        return render_template(
            "analytics.html",
            charts={},
            stats={},
            error=f"Failed to load dashboard data: {exc}",
        )


@app.route("/api/health")
def health():
    """Health check endpoint for deployment monitoring."""
    return jsonify({"status": "healthy", "service": "aiml-capstone"}), 200


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle standard HTTP 413 Payload Too Large error."""
    return (
        render_template(
            "detect.html",
            error="File size exceeds the 8 MB maximum limit.",
            result_image=None,
            detections=None,
        ),
        413,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
