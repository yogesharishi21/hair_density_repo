"""
Stage 5 & 6: Hair Count / Area Calculation and Density Calculation

Density (hairs/cm^2) is only as good as your pixel-to-real-world calibration.
Three calibration modes are supported (see config.CalibrationConfig.mode):

  "reference_circle" - detect a circular marker of known physical diameter
                        placed in the same image plane, derive px_per_cm from it.
  "known_dpi"         - use a fixed DPI/magnification value from your capture
                        device's spec sheet (e.g. a calibrated dermatoscope).
  "manual_scale"      - directly supply px_per_cm measured once for your rig
                        (e.g. by photographing a ruler and measuring).

Get calibration wrong and every downstream density number is wrong -- treat
it as a hardware setup step, not a software default.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np

from config import CalibrationConfig


@dataclass
class DensityResult:
    hair_count: int
    roi_area_px: float
    roi_area_cm2: float
    density_per_cm2: float
    px_per_cm: float
    mean_confidence: float
    calibration_mode: str
    calibration_ok: bool
    notes: list

    def as_dict(self):
        return {
            "hair_count": self.hair_count,
            "roi_area_cm2": round(self.roi_area_cm2, 4),
            "density_per_cm2": round(self.density_per_cm2, 1),
            "px_per_cm_used": round(self.px_per_cm, 2),
            "mean_confidence": round(self.mean_confidence, 3),
            "calibration_mode": self.calibration_mode,
            "calibration_ok": self.calibration_ok,
            "notes": self.notes,
        }


def detect_reference_circle_px_per_cm(image_bgr: np.ndarray, known_diameter_cm: float) -> Optional[float]:
    """
    Detect a circular calibration marker (e.g. a printed dot sticker of known
    diameter placed next to the scalp region) via Hough Circle Transform and
    derive a pixels-per-cm scale factor. Returns None if no confident circle found.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    h, w = gray.shape[:2]

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=w / 4,
        param1=100, param2=40,
        minRadius=int(0.01 * min(h, w)), maxRadius=int(0.2 * min(h, w)),
    )
    if circles is None:
        return None

    # Assume the most prominent circle is the marker.
    circle = np.round(circles[0, 0]).astype(int)
    radius_px = circle[2]
    diameter_px = radius_px * 2
    return diameter_px / known_diameter_cm


def resolve_px_per_cm(image_bgr: np.ndarray, cfg: CalibrationConfig) -> Tuple[float, bool, list]:
    notes = []
    if cfg.mode == "reference_circle":
        val = detect_reference_circle_px_per_cm(image_bgr, cfg.reference_circle_diameter_cm)
        if val is not None:
            return val, True, notes
        notes.append("No reference circle detected; falling back to manual_scale value. "
                      "Place a visible circular marker of known diameter in frame.")
        return cfg.px_per_cm, False, notes

    elif cfg.mode == "known_dpi":
        px_per_cm = cfg.dpi / 2.54
        return px_per_cm, True, notes

    elif cfg.mode == "manual_scale":
        notes.append("Using manually supplied px_per_cm. Verify this was measured for your exact "
                      "camera + working distance -- it does not auto-adjust per image.")
        return cfg.px_per_cm, True, notes

    else:
        notes.append(f"Unknown calibration mode '{cfg.mode}', defaulting to manual_scale value.")
        return cfg.px_per_cm, False, notes


def compute_density(
    detections: List[Dict],
    roi_bbox: Tuple[int, int, int, int],
    image_bgr: np.ndarray,
    calibration_cfg: CalibrationConfig,
) -> DensityResult:
    x, y, w, h = roi_bbox
    roi_area_px = float(w * h)

    px_per_cm, calibration_ok, notes = resolve_px_per_cm(image_bgr, calibration_cfg)
    px_per_cm2 = px_per_cm ** 2

    roi_area_cm2 = roi_area_px / px_per_cm2 if px_per_cm2 > 0 else 0.0
    hair_count = len(detections)
    density = (hair_count / roi_area_cm2) if roi_area_cm2 > 0 else 0.0
    mean_conf = float(np.mean([d["confidence"] for d in detections])) if detections else 0.0

    return DensityResult(
        hair_count=hair_count,
        roi_area_px=roi_area_px,
        roi_area_cm2=roi_area_cm2,
        density_per_cm2=density,
        px_per_cm=px_per_cm,
        mean_confidence=mean_conf,
        calibration_mode=calibration_cfg.mode,
        calibration_ok=calibration_ok,
        notes=notes,
    )
