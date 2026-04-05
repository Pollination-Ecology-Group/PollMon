# Pollinator YOLO Training + Overlay

Training pipeline and real-time screen overlay for pollinator detection.

## Setup

    pip install -r enclosedTest/requirements.txt

## Usage

Prepare dataset:

    python enclosedTest/prepare_dataset.py

Train:

    python enclosedTest/train_yolo.py

Resume / add more time:

    python enclosedTest/train_yolo.py --resume --time-minutes 60

If new images were added, re-run `prepare_dataset.py` first, then resume.

Run overlay:

    python enclosedTest/run_overlay.py

## Config

Everything is in `config.yaml`. Key fields:

- `paths.source_images_dir` — annotated images folder
- `train.model` — base weights (yolo26m.pt etc)
- `train.time_minutes` — training time budget
- `infer.model_path` — trained weights for inference
- `overlay.capture_scale` — lower = faster, 0.35-0.6 is good
- `overlay.monitor_index` — which monitor (1 = primary)

YOLO26 weights: yolo26n/s/m/l/x.pt. Overlay is Windows only.
