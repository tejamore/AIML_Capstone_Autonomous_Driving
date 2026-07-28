"""
app.py
------
Flask web application for the AIML Capstone: Autonomous Driving.

Two capabilities, matching the two parts of the capstone brief:

  Part 1 - Object Detection: upload a road image, the app localizes and
           classifies vehicles (car / bus / motorbike / bicycle / train)
           with bounding boxes using a MobileNet-SSD deep learning model.

  Part 2 - Data Science: an analytics dashboard answering the EDA
           questions about Tesla Autopilot deaths and road safety.

Run locally:   python app.py
Run on Render: gunicorn app:app --chdir app --bind 0.0.0.0:$PORT
"""

import base64
import io
import os
import traceback

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from detector import detect_vehicles
from analysis import load_data, build_dashboard

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload limit

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/detect", methods=["GET", "POST"])
def detect():
    if request.method == "GET":
        return render_template("detect.html", result_image=None, detections=None)

    if "image" not in request.files or request.files["image"].filename == "":
        return render_template(
            "detect.html", error="Please choose an image file first.",
            result_image=None, detections=None,
        )

    file = request.files["image"]
    if not _allowed_file(file.filename):
        return render_template(
            "detect.html", error="Unsupported file type. Use PNG/JPG/JPEG/BMP/WEBP.",
            result_image=None, detections=None,
        )

    try:
        secure_filename(file.filename)  # sanitize (defensive; not persisted to disk)
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError("Could not decode image.")

        annotated, detections, inference_ms = detect_vehicles(image_bgr)

        _, buffer = cv2.imencode(".jpg", annotated)
        encoded = base64.b64encode(buffer).decode("utf-8")

        return render_template(
            "detect.html",
            result_image=encoded,
            detections=detections,
            inference_ms=round(inference_ms, 1),
            error=None,
        )
    except Exception as exc:  # noqa: BLE001
        app.logger.error("Detection failed: %s\n%s", exc, traceback.format_exc())
        return render_template(
            "detect.html",
            error=f"Detection failed: {exc}",
            result_image=None, detections=None,
        )


@app.route("/analytics")
def analytics():
    try:
        df = load_data()
        dashboard = build_dashboard(df)
        return render_template(
            "analytics.html",
            charts=dashboard["charts"],
            stats=dashboard["stats"],
            error=None,
        )
    except Exception as exc:  # noqa: BLE001
        app.logger.error("Analytics failed: %s\n%s", exc, traceback.format_exc())
        return render_template("analytics.html", charts={}, stats={}, error=str(exc))


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
