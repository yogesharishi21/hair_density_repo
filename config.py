"""
Central configuration for the Hair Density Analyzer.
Tweak these values to match your camera / dermatoscope setup.
"""

from dataclasses import dataclass, field


@dataclass
class QualityThresholds:
    min_width: int = 800
    min_height: int = 800
    blur_laplacian_min: float = 60.0       # below this -> image judged "too blurry"
    dark_mean_threshold: float = 40.0      # mean pixel intensity below -> too dark
    bright_mean_threshold: float = 235.0   # mean pixel intensity above -> blown out / overexposed
    min_dynamic_range: float = 25.0        # std-dev of intensity; low -> flat/low-contrast image


@dataclass
class SegmentationConfig:
    # HSV bounds used for scalp/skin-tone masking (classical fallback).
    # Tuned broadly for skin tones under white/warm dermatoscope light; adjust for your data.
    hsv_lower: tuple = (0, 10, 60)
    hsv_upper: tuple = (30, 180, 255)
    morph_kernel: int = 15
    min_region_area_frac: float = 0.05     # discard segmented regions smaller than this fraction of image


@dataclass
class DetectionConfig:
    backend: str = "classical"   # "classical" or "yolo"
    yolo_weights_path: str = "models/hair_follicle_yolov8.pt"
    yolo_conf_threshold: float = 0.35
    yolo_iou_threshold: float = 0.45
    # classical (Frangi vesselness) detector settings
    frangi_scale_range: tuple = (1, 3)
    frangi_scale_step: int = 1
    binary_threshold_percentile: float = 92.0   # top X percentile of vesselness response kept
    min_hair_area_px: int = 4
    max_hair_area_px: int = 400
    min_aspect_ratio: float = 1.6          # hairs are elongated; filters out round noise/pores


@dataclass
class PostProcessConfig:
    dedup_distance_px: int = 6      # merge detections whose centers are closer than this
    confidence_min: float = 0.3


@dataclass
class CalibrationConfig:
    # How pixels map to real-world cm^2. Three supported modes, set `mode`:
    #  "reference_circle" -> a circular marker of known diameter is visible in-frame
    #  "known_dpi"         -> the capture device has a fixed, known DPI/magnification
    #  "manual_scale"      -> user directly supplies px_per_cm
    mode: str = "manual_scale"
    reference_circle_diameter_cm: float = 1.0
    dpi: float = 300.0
    px_per_cm: float = 118.0   # default guess; MUST be calibrated per device for real use


@dataclass
class AppConfig:
    quality: QualityThresholds = field(default_factory=QualityThresholds)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    postprocess: PostProcessConfig = field(default_factory=PostProcessConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    roi_size_cm: float = 1.0   # standard trichoscopy analysis window is often 1cm x 1cm
