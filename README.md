# YOLO Foto Blender

Full pipeline for:
- Generation of annotated synthetic images via Blender (add-on `yolo_foto.py`)
- Conversion and splitting of dataset (COCO → YOLO)
- Training with YOLOv8
- Fast evaluation with real images
- Live inference with one or more webcams

## Structure

- `blender_addon/`: Blender add-on with automatic rendering + COCO annotations
- `dataset/`: scripts to prepare and train YOLO dataset
- `evaluation/`: scripts for fast evaluation of the model on real photos
- `webcam/`: scripts for inference with webcam and latency testing

## Requirements

- Python 3.10+
- Ultralytics `YOLO` (v8+)
- OpenCV
- Pillow-heif (for .heic)
- Blender 3.6.12