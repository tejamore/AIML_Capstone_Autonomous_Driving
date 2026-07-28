"""
download_models.py
-------------------
Run once during the Render build step (see render.yaml / INSTALLATION.md)
so the MobileNet-SSD weights are already cached before the web server
starts serving traffic (avoids a slow first request).

Usage:  python download_models.py
"""
from detector import _ensure_weights  # noqa: E402

if __name__ == "__main__":
    print("Fetching MobileNet-SSD model files ...")
    _ensure_weights()
    print("Done. Model files cached under app/models/")
