"""
Reference training script for a custom hair / follicular-unit YOLOv8 detector.

This is NOT runnable out of the box -- you need annotated data first.
It documents the exact workflow so you (or your annotation team) know what
to produce and how to plug it in.

--------------------------------------------------------------------------
1. DATA YOU NEED
--------------------------------------------------------------------------
Collect close-up scalp/trichoscopy images (ideally at a fixed magnification,
e.g. 20x-70x dermatoscope) and annotate every visible hair shaft (or every
follicular unit, if you want clinical 1/2/3-hair grouping) with a tight
bounding box, in YOLO format:

    dataset/
      images/
        train/*.jpg
        val/*.jpg
      labels/
        train/*.txt   # one line per hair: "0 x_center y_center width height" (normalized 0-1)
        val/*.txt
      data.yaml

    data.yaml:
        path: dataset
        train: images/train
        val: images/val
        names:
          0: hair

Annotation tools: CVAT, Roboflow, or Label Studio all export YOLO-format labels.

Public starting points (still require re-annotation/licensing checks for your use case):
  - Search "trichoscopy dataset", "hair follicle detection dataset" on Roboflow Universe
  - Some open hair-counting research datasets exist for FUE/FUT hair transplant grading;
    verify license terms before clinical or commercial use.

If you cannot get real annotated data yet, use the "classical" backend in this
app (Frangi-filter based) as your baseline -- it needs no training data.

--------------------------------------------------------------------------
2. TRAINING
--------------------------------------------------------------------------
    pip install ultralytics

    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")   # start from a small pretrained checkpoint
    model.train(
        data="dataset/data.yaml",
        epochs=150,
        imgsz=960,          # hairs are tiny -- use a large input size
        batch=16,
        patience=30,
        mosaic=0.0,         # mosaic augmentation can distort thin structures; consider disabling
        degrees=15,         # mild rotation augmentation is fine, hairs have no canonical orientation
    )

--------------------------------------------------------------------------
3. USE THE TRAINED WEIGHTS IN THIS APP
--------------------------------------------------------------------------
Copy the resulting best.pt (e.g. runs/detect/train/weights/best.pt) to:

    models/hair_follicle_yolov8.pt

Then in config.py (or the Streamlit sidebar) set:

    detection.backend = "yolo"
    detection.yolo_weights_path = "models/hair_follicle_yolov8.pt"

--------------------------------------------------------------------------
4. WHY YOLO OVER OTHER MODEL CHOICES
--------------------------------------------------------------------------
- YOLOv8/v9 (Ultralytics): fast, easy to train on a few hundred to a few
  thousand annotated images, good balance of speed/accuracy for small
  elongated objects like hair shafts. Recommended default.
- Detectron2 / Faster R-CNN: higher accuracy ceiling, slower, more setup
  overhead -- worth it only if you have a large annotated dataset and need
  maximum precision (e.g. counting miniaturized/vellus hairs separately).
- Instance segmentation (YOLOv8-seg, Mask R-CNN): use if you need hair
  *length/thickness* per strand, not just count -- segmentation masks let
  you measure shaft width for miniaturization ratio, which plain bounding
  boxes cannot give you.
- Classical CV (Frangi/Hessian ridge filters, as implemented in
  core/detection.py): zero training data required, works today, but is
  more sensitive to lighting, hair color/contrast, and curl, and cannot
  distinguish real hairs from other thin dark structures as reliably as a
  trained detector.
"""

print(__doc__)
