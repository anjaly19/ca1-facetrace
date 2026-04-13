"""
Model Manager
Downloads and caches ArcFace pretrained ONNX model.
ArcFace is the industry-standard face recognition model (same tech used in production systems).
Model: buffalo_sc from InsightFace - ArcFace R50 trained on MS1MV3 (5.8M images, 85k identities)
"""

import os
import sys
import urllib.request
import zipfile
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# ArcFace ONNX model from InsightFace (public, free)
ARCFACE_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip"
ARCFACE_ZIP  = os.path.join(MODELS_DIR, "buffalo_sc.zip")
ARCFACE_DIR  = os.path.join(MODELS_DIR, "buffalo_sc")
ARCFACE_ONNX = os.path.join(ARCFACE_DIR, "w600k_mbf.onnx")

# Detection model
DET_ONNX = os.path.join(ARCFACE_DIR, "det_500m.onnx")


def download_with_progress(url, dest):
    print(f"[MODEL] Downloading {os.path.basename(dest)}...")
    def progress(block, block_size, total):
        if total > 0:
            pct = min(block * block_size / total * 100, 100)
            mb = total / 1024 / 1024
            print(f"\r  {pct:.1f}% of {mb:.1f}MB", end='', flush=True)
    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print()


def ensure_arcface_model():
    """Download ArcFace model if not present. Returns path to .onnx file."""
    if os.path.exists(ARCFACE_ONNX) and os.path.exists(DET_ONNX):
        print(f"[MODEL] ArcFace model ready at {ARCFACE_ONNX}")
        return ARCFACE_ONNX, DET_ONNX

    if not os.path.exists(ARCFACE_ZIP):
        print("[MODEL] ArcFace model not found. Downloading buffalo_sc from InsightFace...")
        print("[MODEL] This is a one-time download (~85MB)")
        try:
            download_with_progress(ARCFACE_URL, ARCFACE_ZIP)
        except Exception as e:
            print(f"[MODEL] Download failed: {e}")
            print("[MODEL] Falling back to custom CNN mode")
            return None, None

    # Extract
    if not os.path.exists(ARCFACE_DIR):
        print("[MODEL] Extracting model...")
        os.makedirs(ARCFACE_DIR, exist_ok=True)
        try:
            with zipfile.ZipFile(ARCFACE_ZIP, 'r') as z:
                z.extractall(ARCFACE_DIR)
            print("[MODEL] Extracted successfully")
        except Exception as e:
            print(f"[MODEL] Extraction failed: {e}")
            return None, None

    # Find onnx files
    rec_path = None
    det_path = None
    for root, dirs, files in os.walk(ARCFACE_DIR):
        for f in files:
            full = os.path.join(root, f)
            if f.endswith('.onnx'):
                if 'det' in f or 'scrfd' in f:
                    det_path = full
                else:
                    rec_path = full

    if rec_path:
        print(f"[MODEL] Recognition model: {rec_path}")
    if det_path:
        print(f"[MODEL] Detection model: {det_path}")

    return rec_path, det_path
