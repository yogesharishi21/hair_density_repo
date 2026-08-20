"""
Stage 1: Image Quality Check
Rejects/flags images that are too blurry, too small, too dark/bright,
or too flat in contrast before any expensive processing happens.
"""

from dataclasses import dataclass, field
import cv2
import numpy as np

from config import QualityThresholds


@dataclass
class QualityReport:
    passed: bool
    width: int
    height: int
    blur_score: float
    mean_intensity: float
    std_intensity: float
    issues: list = field(default_factory=list)

    def as_dict(self):
        return {
            "passed": self.passed,
            "width": self.width,
            "height": self.height,
            "blur_score": round(self.blur_score, 2),
            "mean_intensity": round(self.mean_intensity, 2),
            "std_intensity": round(self.std_intensity, 2),
            "issues": self.issues,
        }


def _laplacian_variance(gray: np.ndarray) -> float:
    """Higher variance => sharper image. Standard focus-measure operator."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def check_image_quality(image_bgr: np.ndarray, thresholds: QualityThresholds) -> QualityReport:
    if image_bgr is None or image_bgr.size == 0:
        return QualityReport(False, 0, 0, 0, 0, 0, ["Image failed to load / empty."])

    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    blur_score = _laplacian_variance(gray)
    mean_intensity = float(gray.mean())
    std_intensity = float(gray.std())

    issues = []
    if w < thresholds.min_width or h < thresholds.min_height:
        issues.append(f"Resolution too low ({w}x{h}); need >= {thresholds.min_width}x{thresholds.min_height}.")
    if blur_score < thresholds.blur_laplacian_min:
        issues.append(f"Image too blurry (sharpness={blur_score:.1f}, min={thresholds.blur_laplacian_min}).")
    if mean_intensity < thresholds.dark_mean_threshold:
        issues.append(f"Image too dark (mean intensity={mean_intensity:.1f}).")
    if mean_intensity > thresholds.bright_mean_threshold:
        issues.append(f"Image overexposed (mean intensity={mean_intensity:.1f}).")
    if std_intensity < thresholds.min_dynamic_range:
        issues.append(f"Low contrast / flat lighting (std={std_intensity:.1f}).")

    return QualityReport(
        passed=(len(issues) == 0),
        width=w, height=h,
        blur_score=blur_score,
        mean_intensity=mean_intensity,
        std_intensity=std_intensity,
        issues=issues,
    )
