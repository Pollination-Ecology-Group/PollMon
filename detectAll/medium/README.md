# Detect All Pollinators — Medium (YOLO26m)

Single-class ("pollinator") detection using the medium YOLO26m model.
All 25 species are merged into one class for binary pollinator/no-pollinator detection.

## Setup

    pip install -r detectAll/medium/requirements.txt

## Usage

Prepare dataset (remaps all species labels to single "pollinator" class):

    python detectAll/medium/prepare_dataset.py

Train:

    python detectAll/medium/train_yolo.py

Resume / add more time:

    python detectAll/medium/train_yolo.py --resume --time-minutes 60

If new images were added, re-run `prepare_dataset.py` first, then resume.

Run overlay:

    python detectAll/medium/run_overlay.py

## Config

Everything is in `config.yaml`. Key fields:

- `paths.source_images_dir` — annotated images folder (same source as multi-class)
- `paths.backgrounds_dir` — negative sample images (no pollinators)
- `train.model` — base weights (yolo26m.pt)
- `train.time_minutes` — training time budget
- `infer.model_path` — trained weights for inference
- `overlay.capture_scale` — lower = faster, 0.35-0.6 is good
- `overlay.monitor_index` — which monitor (1 = primary)

YOLO26 weights: yolo26n/s/m/l/x.pt. Overlay is Windows only.
