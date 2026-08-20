"""
Hair Density Analyzer - Streamlit App

Run with:
    streamlit run app.py
"""

import copy
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from config import AppConfig
from core.pipeline import run_pipeline

st.set_page_config(page_title="Hair Density Analyzer", layout="wide")


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def build_config_from_sidebar() -> AppConfig:
    cfg = AppConfig()

    st.sidebar.header("1. Detection backend")
    cfg.detection.backend = st.sidebar.selectbox(
        "Backend", ["classical", "yolo"],
        help="'classical' works immediately, no training needed. "
             "'yolo' needs a trained weights file (see MODEL_NOTES.md)."
    )
    if cfg.detection.backend == "yolo":
        cfg.detection.yolo_weights_path = st.sidebar.text_input(
            "YOLO weights path", value=cfg.detection.yolo_weights_path
        )
        cfg.detection.yolo_conf_threshold = st.sidebar.slider("YOLO confidence threshold", 0.05, 0.95, 0.35)

    st.sidebar.header("2. Calibration (pixels -> cm)")
    cfg.calibration.mode = st.sidebar.selectbox(
        "Calibration mode", ["manual_scale", "known_dpi", "reference_circle"],
        help="This determines real-world density. Wrong calibration = wrong hairs/cm^2."
    )
    if cfg.calibration.mode == "manual_scale":
        cfg.calibration.px_per_cm = st.sidebar.number_input(
            "Pixels per cm (measure once for your camera setup)",
            min_value=1.0, value=cfg.calibration.px_per_cm, step=1.0
        )
    elif cfg.calibration.mode == "known_dpi":
        cfg.calibration.dpi = st.sidebar.number_input("Capture device DPI", min_value=1.0, value=cfg.calibration.dpi)
    elif cfg.calibration.mode == "reference_circle":
        cfg.calibration.reference_circle_diameter_cm = st.sidebar.number_input(
            "Reference marker diameter (cm)", min_value=0.01, value=1.0, step=0.1
        )

    with st.sidebar.expander("3. Advanced: quality thresholds"):
        cfg.quality.blur_laplacian_min = st.slider("Min sharpness (Laplacian var)", 0.0, 300.0, cfg.quality.blur_laplacian_min)
        cfg.quality.min_width = st.number_input("Min width (px)", value=cfg.quality.min_width)
        cfg.quality.min_height = st.number_input("Min height (px)", value=cfg.quality.min_height)

    with st.sidebar.expander("4. Advanced: detection tuning (classical)"):
        cfg.detection.binary_threshold_percentile = st.slider(
            "Vesselness threshold percentile", 50.0, 99.5, cfg.detection.binary_threshold_percentile
        )
        cfg.detection.min_hair_area_px = st.number_input("Min hair blob area (px)", value=cfg.detection.min_hair_area_px)
        cfg.detection.max_hair_area_px = st.number_input("Max hair blob area (px)", value=cfg.detection.max_hair_area_px)
        cfg.detection.min_aspect_ratio = st.slider("Min elongation (aspect ratio)", 1.0, 5.0, cfg.detection.min_aspect_ratio)

    with st.sidebar.expander("5. Advanced: post-processing"):
        cfg.postprocess.confidence_min = st.slider("Min confidence", 0.0, 1.0, cfg.postprocess.confidence_min)
        cfg.postprocess.dedup_distance_px = st.number_input("Dedup distance (px)", value=cfg.postprocess.dedup_distance_px)

    bypass = st.sidebar.checkbox("Bypass quality gate (debug only)", value=False)
    return cfg, bypass


def main():
    st.title("🔬 Hair Density Analyzer")
    st.caption(
        "Upload a trichoscopy / scalp macro photo. The app checks image quality, "
        "segments the scalp region, detects individual hairs, and reports density (hairs/cm²)."
    )

    cfg, bypass_quality = build_config_from_sidebar()

    uploaded = st.file_uploader("Upload scalp image", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded is None:
        st.info("Upload an image to begin. For a 1cm² trichoscopy read, use a dermatoscope "
                "at fixed magnification and a well-lit, in-focus close-up shot.")
        return

    pil_img = Image.open(uploaded)
    img_bgr = pil_to_bgr(pil_img)

    col_in, col_out = st.columns(2)
    with col_in:
        st.subheader("Input")
        st.image(bgr_to_rgb(img_bgr), use_container_width=True)

    with st.spinner("Running pipeline..."):
        result = run_pipeline(img_bgr, cfg, bypass_quality_gate=bypass_quality)

    with col_out:
        st.subheader("Result")
        if not result.success:
            st.error(result.error)
            if result.quality:
                st.json(result.quality.as_dict())
            st.warning("Fix the flagged issue(s) and re-upload, or tick 'Bypass quality gate' to proceed anyway.")
            return

        st.image(bgr_to_rgb(result.annotated_image), use_container_width=True,
                  caption="Blue box = detected scalp ROI. Green boxes = detected hairs.")

    st.divider()
    d = result.density.as_dict()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hair count", d["hair_count"])
    c2.metric("Density", f"{d['density_per_cm2']:.1f} hairs/cm²")
    c3.metric("ROI area", f"{d['roi_area_cm2']:.3f} cm²")
    c4.metric("Mean confidence", f"{d['mean_confidence']:.2f}")

    if not d["calibration_ok"]:
        st.warning("⚠️ Calibration fallback was used — density numbers may be inaccurate. " + " ".join(d["notes"]))
    elif d["notes"]:
        st.info(" ".join(d["notes"]))

    with st.expander("Image quality report"):
        st.json(result.quality.as_dict())

    with st.expander("Raw vs filtered detections"):
        st.write(f"Raw detections before post-processing: **{len(result.raw_detections)}**")
        st.write(f"Final detections after confidence filter + dedup: **{len(result.final_detections)}**")

    with st.expander("Clinical reference ranges (for context only, not a diagnosis)"):
        st.markdown(
            "- Normal scalp density is commonly cited around **150–300 hairs/cm²**, "
            "varying by ethnicity, age, and scalp region.\n"
            "- Androgenetic alopecia workups often flag values notably below this range, "
            "alongside miniaturization (thin/vellus hair ratio) — which this app does not yet measure.\n"
            "- These figures are general references from trichology literature, not a substitute "
            "for evaluation by a dermatologist."
        )

    st.download_button(
        "Download annotated image",
        data=cv2.imencode(".png", result.annotated_image)[1].tobytes(),
        file_name="hair_density_annotated.png",
        mime="image/png",
    )


if __name__ == "__main__":
    main()
