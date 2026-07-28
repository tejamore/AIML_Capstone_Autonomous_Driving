"""
detector.py
-----------
Vehicle detection module for the Autonomous Driving capstone (Part 1).

Uses OpenCV's DNN module with a pre-trained MobileNet-SSD (Caffe) model.
This keeps the deployed app small and fast enough to run comfortably on a
free Render web service (no PyTorch / TensorFlow required), while still
satisfying the assignment's requirement of a deep-learning model that:
    1. predicts the TYPE of vehicle present in an image, and
    2. localizes it with a rectangular bounding box.

The model weights are downloaded once (at build time or on first request)
and cached locally under MODEL_DIR.
"""

import os
import time
import logging
import urllib.request

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
PROTOTXT_PATH = os.path.join(MODEL_DIR, "MobileNetSSD_deploy.prototxt")
MODEL_PATH = os.path.join(MODEL_DIR, "MobileNetSSD_deploy.caffemodel")

# Known-good mirrors for the classic MobileNet-SSD (VOC-trained) weights.
# Multiple mirrors are tried in order in case one host is unreachable.
PROTOTXT_URLS = [
    "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt",
    "https://raw.githubusercontent.com/djmv/MobilNet_SSD_opencv/master/MobileNetSSD_deploy.prototxt",
]
MODEL_URLS = [
    "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel",
    "https://github.com/djmv/MobilNet_SSD_opencv/raw/master/MobileNetSSD_deploy.caffemodel",
]

# The 21 classes the network was trained on (VOC + background)
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor",
]

# Subset of classes we treat as "vehicles" for this project
VEHICLE_CLASSES = {"bicycle", "bus", "car", "motorbike", "train", "aeroplane"}

BOX_COLOR = (36, 130, 255)   # BGR - orange
TEXT_COLOR = (255, 255, 255)

_net = None  # lazy-loaded singleton


def _download_first_available(urls, dest_path):
    last_error = None
    for url in urls:
        try:
            logger.info("Downloading %s -> %s", url, dest_path)
            urllib.request.urlretrieve(url, dest_path)
            if os.path.getsize(dest_path) > 1000:  # sanity check, not an error page
                return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if os.path.exists(dest_path):
                os.remove(dest_path)
    raise RuntimeError(
        f"Could not download required file to {dest_path} from any mirror: {last_error}"
    )


def _ensure_weights():
    """Download the prototxt/caffemodel files if they are not already cached."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(PROTOTXT_PATH):
        _download_first_available(PROTOTXT_URLS, PROTOTXT_PATH)

    if not os.path.exists(MODEL_PATH):
        _download_first_available(MODEL_URLS, MODEL_PATH)


def load_model():
    """Load (and cache) the OpenCV DNN network."""
    global _net
    if _net is None:
        _ensure_weights()
        _net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
    return _net


def detect_vehicles(image_bgr, conf_threshold: float = 0.4):
    """
    Run vehicle detection on a BGR image (as read by cv2.imread).

    Returns
    -------
    annotated_image : np.ndarray
        Copy of the input image with bounding boxes + labels drawn on it.
    detections : list[dict]
        One entry per detected vehicle:
        {"label": str, "confidence": float, "box": [x1, y1, x2, y2]}
    inference_ms : float
        Wall-clock time spent on the forward pass, in milliseconds.
    """
    net = load_model()
    (h, w) = image_bgr.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(image_bgr, (300, 300)), 0.007843, (300, 300), 127.5
    )
    net.setInput(blob)

    start = time.time()
    raw_detections = net.forward()
    inference_ms = (time.time() - start) * 1000.0

    annotated = image_bgr.copy()
    results = []

    for i in range(raw_detections.shape[2]):
        confidence = float(raw_detections[0, 0, i, 2])
        if confidence < conf_threshold:
            continue

        class_id = int(raw_detections[0, 0, i, 1])
        label = VOC_CLASSES[class_id] if class_id < len(VOC_CLASSES) else "unknown"

        if label not in VEHICLE_CLASSES:
            continue

        box = raw_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        results.append(
            {
                "label": label,
                "confidence": round(confidence, 3),
                "box": [int(x1), int(y1), int(x2), int(y2)],
            }
        )

        cv2.rectangle(annotated, (x1, y1), (x2, y2), BOX_COLOR, 2)
        caption = f"{label}: {confidence * 100:.1f}%"
        text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
        cv2.putText(
            annotated, caption, (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLOR, 2,
        )

    return annotated, results, inference_ms
