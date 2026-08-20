"""
Stage 2: Scalp Region Detection / Segmentation
Classical fallback: HSV skin-tone thresholding + morphological cleanup
+ largest-contour selection, producing a binary mask of the scalp region
and its bounding ROI.

For production accuracy, swap `segment_scalp_classical` for a trained
segmentation model (e.g. a small U-Net or DeepLabV3 fine-tuned on
trichoscopy images) behind the same function signature.
"""

import cv2
import numpy as np

from config import SegmentationConfig


def segment_scalp_classical(image_bgr: np.ndarray, cfg: SegmentationConfig):
    """
    Returns:
        mask (uint8 HxW, 0/255): scalp region mask
        bbox (x, y, w, h): bounding box of the largest scalp region
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array(cfg.hsv_lower, dtype=np.uint8)
    upper = np.array(cfg.hsv_upper, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.morph_kernel, cfg.morph_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    h, w = mask.shape[:2]
    min_area = cfg.min_region_area_frac * h * w

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    if not contours:
        # Fallback: use the whole image as the ROI (e.g. tight dermatoscope crops
        # that are already all scalp, so skin-tone thresholding finds nothing).
        full_mask = np.full((h, w), 255, dtype=np.uint8)
        return full_mask, (0, 0, w, h)

    largest = max(contours, key=cv2.contourArea)
    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, [largest], -1, 255, thickness=cv2.FILLED)
    x, y, bw, bh = cv2.boundingRect(largest)

    return clean_mask, (x, y, bw, bh)


def apply_mask(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
