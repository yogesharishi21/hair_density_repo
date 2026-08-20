# Hair Density Analyzer

Computer-vision pipeline + Streamlit app that estimates hair density
(hairs/cm²) from a scalp/trichoscopy photo, following this pipeline:

```
Scalp Image -> Quality Check -> Scalp Segmentation -> Hair Detection
            -> Post-Processing -> Count/Area -> Density -> Final Result
```

## Setup

```bash
pip install -r requirements.txt
```

`ultralytics` is only required if you use the YOLO detection backend
(`detection.backend = "yolo"`). The default `"classical"` backend needs
just OpenCV + scikit-image and works with no trained model.

## Run the app

```bash
streamlit run app.py
```

Upload an image, set your calibration in the sidebar, and read off count/density.

## Project layout

```
config.py              All tunable thresholds/settings in one place (dataclasses)
core/
  quality_check.py      Stage 1: blur / resolution / exposure checks
  segmentation.py        Stage 2: scalp region mask (classical HSV + contours)
  detection.py            Stage 3: hair detection - "classical" (Frangi vesselness)
                           or "yolo" (Ultralytics, needs trained weights)
  postprocess.py          Stage 4: confidence filter + duplicate removal
  density.py               Stage 5/6: count, ROI area, px->cm calibration, density
  pipeline.py               Orchestrates all stages end-to-end
app.py                  Streamlit UI
train_yolo.py           Guide + reference script for training a custom YOLO hair detector
```

## Model choice

- **Default / no-training-needed**: classical Frangi ridge-filter detector
  (`core/detection.py::detect_hairs_classical`). Good baseline, verified
  working in this repo, weaker on curly/overlapping/low-contrast hair.
- **Recommended for production accuracy**: fine-tuned **YOLOv8** (Ultralytics)
  on annotated trichoscopy images. See `train_yolo.py` for the full data
  format, training command, and how to plug in the resulting weights.
  Swap `detection.backend = "yolo"` once you have `models/hair_follicle_yolov8.pt`.
- If you need per-strand **thickness** (for miniaturization/vellus-hair ratio,
  which matters clinically for androgenetic alopecia grading), upgrade to
  **YOLOv8-seg** (instance segmentation) instead of plain detection boxes.

## ⚠️ Calibration is the part you must not skip

Density is `hair_count / (ROI_area_px / px_per_cm²)`. If `px_per_cm` is wrong,
every density number is wrong, silently. Three modes are supported in
`config.CalibrationConfig`:

1. `manual_scale` (default): you supply `px_per_cm` once, measured for your
   exact camera + working distance (e.g. photograph a ruler and count pixels/cm).
2. `known_dpi`: use your dermatoscope/camera's documented DPI/magnification.
3. `reference_circle`: place a printed circular marker of known diameter in
   every shot; the app detects it via Hough circles and self-calibrates per image.

For repeatable clinical-style tracking, a fixed-magnification dermatoscope
(e.g. many commercial trichoscopes shoot a calibrated 1cm² field of view) is
strongly preferred over a phone camera at arbitrary zoom/distance.

## Accuracy notes / limitations (read before trusting numbers)

- The classical backend was validated on a synthetic test image (150 known
  strokes) and detected ~83 after confidence filtering + dedup — a useful
  baseline, not clinical-grade. Expect it to under- or over-count on real
  photos depending on hair color, curl, and lighting; tune the thresholds
  in the sidebar's "Advanced" sections per your dataset.
- This tool is for research/self-tracking use. It does not diagnose
  alopecia or any medical condition — pair it with a dermatologist for
  clinical decisions.
- Overlapping/matted hair, motion blur, and non-scalp hair (eyebrows, beard
  in frame) will bias counts; keep the capture protocol consistent.
