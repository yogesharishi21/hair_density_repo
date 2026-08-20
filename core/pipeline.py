"""
Full pipeline orchestrator, mirroring the flow:

  Scalp Image -> Quality Check -> Scalp Segmentation -> Hair Detection
  -> Post-processing -> Count/Area -> Density -> Final Result
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
import cv2
import numpy as np

from config import AppConfig
from core.quality_check import check_image_quality, QualityReport
from core.segmentation import segment_scalp_classical, apply_mask
from core.detection import detect_hairs
from core.postprocess import postprocess_detections
from core.density import compute_density, DensityResult


@dataclass
class PipelineResult:
    success: bool
    quality: Optional[QualityReport] = None
    roi_bbox: Optional[tuple] = None
    raw_detections: List[Dict] = field(default_factory=list)
    final_detections: List[Dict] = field(default_factory=list)
    density: Optional[DensityResult] = None
    error: Optional[str] = None
    annotated_image: Optional[np.ndarray] = None


def draw_annotations(image_bgr: np.ndarray, roi_bbox, detections: List[Dict]) -> np.ndarray:
    out = image_bgr.copy()
    if roi_bbox is not None:
        x, y, w, h = roi_bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 200, 0), 2)
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return out


def run_pipeline(image_bgr: np.ndarray, cfg: AppConfig, bypass_quality_gate: bool = False) -> PipelineResult:
    # Stage 1: Image Quality Check
    quality = check_image_quality(image_bgr, cfg.quality)
    if not quality.passed and not bypass_quality_gate:
        return PipelineResult(success=False, quality=quality,
                               error="Image failed quality gate: " + "; ".join(quality.issues))

    # Stage 2: Scalp Region Detection / Segmentation
    mask, roi_bbox = segment_scalp_classical(image_bgr, cfg.segmentation)

    # Stage 3: Hair / Follicle Detection
    raw_detections = detect_hairs(image_bgr, mask, cfg.detection)

    # Stage 4: Post-Processing (duplicate removal, confidence filtering)
    final_detections = postprocess_detections(raw_detections, cfg.postprocess)

    # Stage 5 & 6: Hair Count / Area + Density Calculation
    density = compute_density(final_detections, roi_bbox, image_bgr, cfg.calibration)

    annotated = draw_annotations(image_bgr, roi_bbox, final_detections)

    return PipelineResult(
        success=True,
        quality=quality,
        roi_bbox=roi_bbox,
        raw_detections=raw_detections,
        final_detections=final_detections,
        density=density,
        annotated_image=annotated,
    )
