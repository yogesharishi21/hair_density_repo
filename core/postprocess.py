"""
Stage 4: Post-Processing
- Confidence filtering
- Duplicate / overlapping detection removal (greedy NMS on centers)
"""

from typing import List, Dict
import numpy as np

from config import PostProcessConfig


def filter_by_confidence(detections: List[Dict], min_conf: float) -> List[Dict]:
    return [d for d in detections if d["confidence"] >= min_conf]


def remove_duplicates(detections: List[Dict], min_distance_px: float) -> List[Dict]:
    """Greedy suppression: keep highest-confidence detection, drop others
    whose center lies within `min_distance_px` of an already-kept one."""
    if not detections:
        return []

    sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept: List[Dict] = []
    kept_points = []

    for det in sorted_dets:
        pt = np.array([det["x"], det["y"]])
        is_dup = False
        for kp in kept_points:
            if np.linalg.norm(pt - kp) < min_distance_px:
                is_dup = True
                break
        if not is_dup:
            kept.append(det)
            kept_points.append(pt)

    return kept


def postprocess_detections(detections: List[Dict], cfg: PostProcessConfig) -> List[Dict]:
    filtered = filter_by_confidence(detections, cfg.confidence_min)
    deduped = remove_duplicates(filtered, cfg.dedup_distance_px)
    return deduped
