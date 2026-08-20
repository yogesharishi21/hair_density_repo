"""
Stage 3: Hair / Follicle Detection

Two interchangeable backends behind one interface, `detect_hairs()`:

1. "classical"  - Frangi vesselness (ridge detector) + connected-component
                  analysis. Works out of the box, no training data needed.
                  Good baseline / fallback, weaker on curly or overlapping hair.

2. "yolo"       - Ultralytics YOLOv8, object-detection style, one box per hair
                  or per follicular unit. Needs a model trained on annotated
                  trichoscopy images (see MODEL_NOTES.md). Much more robust on
                  real clinical images, curly/dark/light hair, and follicular
                  unit grouping (1/2/3-hair units), which is what dermatologists
                  actually report.

Each detection is returned as a dict:
    {"x": cx, "y": cy, "bbox": (x1,y1,x2,y2), "confidence": float, "length_px": float}
so downstream stages don't care which backend produced it.
"""

from dataclasses import dataclass
from typing import List, Dict
import numpy as np
import cv2

from config import DetectionConfig

try:
    from skimage.filters import frangi
except ImportError:  # pragma: no cover
    frangi = None


# --------------------------------------------------------------------------
# Classical backend
# --------------------------------------------------------------------------

def detect_hairs_classical(image_bgr: np.ndarray, mask: np.ndarray, cfg: DetectionConfig) -> List[Dict]:
    if frangi is None:
        raise RuntimeError("scikit-image is required for the classical detector (pip install scikit-image).")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray, gray, mask=mask)

    # CLAHE improves contrast of thin hair shafts against skin before ridge detection.
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    gray_f = gray_eq.astype(np.float64) / 255.0
    scale_range = cfg.frangi_scale_range
    vessel_response = frangi(
        gray_f,
        sigmas=range(scale_range[0], scale_range[1] + 1, cfg.frangi_scale_step),
        black_ridges=True,   # hairs are darker than skin in most trichoscopy captures
    )

    vessel_response = cv2.normalize(vessel_response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    thresh_val = np.percentile(vessel_response[mask > 0], cfg.binary_threshold_percentile) \
        if np.any(mask > 0) else np.percentile(vessel_response, cfg.binary_threshold_percentile)
    _, binary = cv2.threshold(vessel_response, thresh_val, 255, cv2.THRESH_BINARY)
    binary = cv2.bitwise_and(binary, binary, mask=mask)

    # Light cleanup: remove single-pixel speckle without erasing thin hair shafts.
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    detections = []
    for label_id in range(1, num_labels):  # skip background label 0
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < cfg.min_hair_area_px or area > cfg.max_hair_area_px:
            continue

        x = stats[label_id, cv2.CC_STAT_LEFT]
        y = stats[label_id, cv2.CC_STAT_TOP]
        w = stats[label_id, cv2.CC_STAT_WIDTH]
        h = stats[label_id, cv2.CC_STAT_HEIGHT]

        aspect_ratio = max(w, h) / max(1, min(w, h))
        if aspect_ratio < cfg.min_aspect_ratio:
            continue  # discard blob-like noise (pores, dust) - real hairs are elongated

        cx, cy = centroids[label_id]
        # confidence heuristic: normalized vessel-response strength inside the component
        component_mask = (labels == label_id)
        strength = float(vessel_response[component_mask].mean()) / 255.0

        detections.append({
            "x": float(cx),
            "y": float(cy),
            "bbox": (int(x), int(y), int(x + w), int(y + h)),
            "confidence": round(min(1.0, strength * (aspect_ratio / cfg.min_aspect_ratio)), 3),
            "length_px": float(max(w, h)),
        })

    return detections


# --------------------------------------------------------------------------
# YOLO backend
# --------------------------------------------------------------------------

_yolo_model_cache = {}


def detect_hairs_yolo(image_bgr: np.ndarray, mask: np.ndarray, cfg: DetectionConfig) -> List[Dict]:
    """
    Requires `pip install ultralytics` and a weights file trained on
    hair / follicular-unit annotations (see MODEL_NOTES.md for dataset
    and training guidance). Loads/caches the model by weights path.
    """
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError(
            "ultralytics is not installed. Run: pip install ultralytics"
        ) from e

    if cfg.yolo_weights_path not in _yolo_model_cache:
        _yolo_model_cache[cfg.yolo_weights_path] = YOLO(cfg.yolo_weights_path)
    model = _yolo_model_cache[cfg.yolo_weights_path]

    masked_img = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
    results = model.predict(
        source=masked_img,
        conf=cfg.yolo_conf_threshold,
        iou=cfg.yolo_iou_threshold,
        verbose=False,
    )[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        detections.append({
            "x": cx,
            "y": cy,
            "bbox": (int(x1), int(y1), int(x2), int(y2)),
            "confidence": round(conf, 3),
            "length_px": float(max(x2 - x1, y2 - y1)),
        })

    return detections


# --------------------------------------------------------------------------
# Unified entry point
# --------------------------------------------------------------------------

def detect_hairs(image_bgr: np.ndarray, mask: np.ndarray, cfg: DetectionConfig) -> List[Dict]:
    if cfg.backend == "yolo":
        return detect_hairs_yolo(image_bgr, mask, cfg)
    elif cfg.backend == "classical":
        return detect_hairs_classical(image_bgr, mask, cfg)
    else:
        raise ValueError(f"Unknown detection backend: {cfg.backend}")
