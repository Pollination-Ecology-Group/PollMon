"""python FinalSpeciesDetection/process_video.py <input_video> [--output <path>]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from common import load_config, resolve_path
from tracker import Detection, TrackedObject, MultiObjectTracker, create_tracker_from_config


# ── persistent species label per track ────────────────────────────────

class SpeciesAccumulator:
    """Accumulates classification evidence across frames per track.
    """

    def __init__(self) -> None:
        # track_id → {species: cumulative_confidence}
        self._sums: Dict[int, Dict[str, float]] = {}

    def update(self, track_id: int, label: str, confidence: float) -> Tuple[str, float]:
        """Add evidence and return the current best species + its total."""
        buckets = self._sums.setdefault(track_id, {})
        buckets[label] = buckets.get(label, 0.0) + confidence
        best_label = max(buckets, key=buckets.get)  # type: ignore[arg-type]
        return best_label, buckets[best_label]

    def get(self, track_id: int) -> Optional[Tuple[str, float]]:
        """Return stored best label for a coasting track (no new evidence)."""
        buckets = self._sums.get(track_id)
        if not buckets:
            return None
        best = max(buckets, key=buckets.get)  # type: ignore[arg-type]
        return best, buckets[best]

    def prune(self, active_ids: set[int]) -> None:
        dead = [tid for tid in self._sums if tid not in active_ids]
        for tid in dead:
            del self._sums[tid]


class DisplaySmoother:
    """EMA-smooths bounding boxes per track so they don't visually jump.
    """

    def __init__(self, pos_alpha: float = 0.55, size_alpha: float = 0.20) -> None:
        self.pos_alpha = pos_alpha
        self.size_alpha = size_alpha
        # track_id → (cx, cy, w, h)
        self._state: Dict[int, Tuple[float, float, float, float]] = {}

    def smooth(self, track_id: int, x1: float, y1: float, x2: float, y2: float
               ) -> Tuple[int, int, int, int]:
        """Return EMA-smoothed integer bbox coords."""
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1

        prev = self._state.get(track_id)
        if prev is None:
            self._state[track_id] = (cx, cy, w, h)
        else:
            pa, sa = self.pos_alpha, self.size_alpha
            cx = pa * cx + (1 - pa) * prev[0]
            cy = pa * cy + (1 - pa) * prev[1]
            w  = sa * w  + (1 - sa) * prev[2]
            h  = sa * h  + (1 - sa) * prev[3]
            self._state[track_id] = (cx, cy, w, h)

        sx1 = int(round(cx - w / 2))
        sy1 = int(round(cy - h / 2))
        sx2 = int(round(cx + w / 2))
        sy2 = int(round(cy + h / 2))
        return sx1, sy1, sx2, sy2

    def prune(self, active_ids: set[int]) -> None:
        dead = [tid for tid in self._state if tid not in active_ids]
        for tid in dead:
            del self._state[tid]


# ── drawing helpers ──────────────────────────────────────────────────

_COLORS = [
    (0, 255, 0), (255, 100, 0), (0, 200, 255), (255, 0, 150),
    (100, 255, 100), (255, 200, 0), (0, 150, 255), (200, 0, 255),
    (255, 255, 0), (0, 255, 200), (128, 0, 255), (255, 128, 0),
]


def draw_box(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    label: str,
    conf: float,
    color: Tuple[int, int, int],
    thickness: int = 2,
    font_scale: float = 0.6,
) -> None:
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    text = f"{label} {conf:.2f}"
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    ty = max(th + 4, y1 - 6)
    cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 4, ty + baseline), (0, 0, 0), -1)
    cv2.putText(frame, text, (x1 + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, 1, cv2.LINE_AA)


# ── main ─────────────────────────────────────────────────────────────

def main() -> None:
    # ---- venv re-exec ----
    project_root = Path(__file__).resolve().parents[1]
    venv_python = project_root / ".venv" / "Scripts" / "python.exe"
    if (
        venv_python.exists()
        and sys.prefix == sys.base_prefix
        and os.environ.get("POLLINATOR_REEXEC") != "1"
    ):
        os.environ["POLLINATOR_REEXEC"] = "1"
        os.execv(str(venv_python), [str(venv_python), *sys.argv])

    if sys.prefix != sys.base_prefix:
        try:
            import torch; from ultralytics import YOLO  # noqa
        except Exception:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r",
                 str(Path(__file__).parent / "requirements.txt")],
                check=True,
            )
            import torch
            from ultralytics import YOLO

    # ---- args ----
    parser = argparse.ArgumentParser(
        description="Full-frame detection + classification with tracking on video.")
    parser.add_argument("input", type=str, help="Path to input video")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to output video (default: <input>_annotated.<ext>)")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (e.g. cpu, 0)")
    args = parser.parse_args()

    config = load_config(Path(__file__).parent / "config.yaml")
    infer_cfg = config["infer"]
    video_cfg = config.get("video", {})

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    suffix = video_cfg.get("output_suffix", "_annotated")
    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = input_path.with_stem(input_path.stem + suffix)

    # ---- config values ----
    det_imgsz  = int(video_cfg.get("det_imgsz", 1280))
    det_conf   = float(video_cfg.get("det_conf", 0.20))
    det_iou    = float(video_cfg.get("det_iou", 0.4))
    cls_imgsz  = int(video_cfg.get("cls_imgsz", 448))
    cls_conf   = float(video_cfg.get("cls_conf", 0.3))
    crop_pad   = float(video_cfg.get("crop_padding", 0.30))
    use_half   = bool(video_cfg.get("half", True))
    codec      = video_cfg.get("codec", "mp4v")
    coast_sec  = float(video_cfg.get("coast_seconds", 1.0))

    # ---- device ----
    if args.device is not None:
        device = int(args.device) if args.device.isdigit() else args.device
    else:
        dev = infer_cfg.get("device", "auto")
        if isinstance(dev, str) and dev.lower() in {"auto", "cuda"}:
            device = 0 if torch.cuda.is_available() else "cpu"
        else:
            device = dev
    half = use_half and device != "cpu" and torch.cuda.is_available()

    # ---- models ----
    det_model_path = infer_cfg["detection_model_path"]
    cls_model_path = infer_cfg["classification_model_path"]
    print(f"[Models] Detection:      {det_model_path}")
    print(f"[Models] Classification: {cls_model_path}")

    det_model = YOLO(det_model_path)
    det_model.fuse()
    cls_model = YOLO(cls_model_path)
    cls_model.fuse()

    # ---- open video (early — we need fps for coast frame calc) ----
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[Video] {input_path.name}: {frame_w}x{frame_h} @ {fps:.1f} FPS, "
          f"{total_frames} frames")
    print(f"[Detection] imgsz={det_imgsz} (YOLO will letterbox from {frame_w}x{frame_h})")

    # ---- tracker + species accumulator + display smoother ----
    tracker = create_tracker_from_config(video_cfg)
    # Override coasting budget so tracks survive ~coast_sec of occlusion
    coast_frames = max(int(coast_sec * fps), tracker.config.max_age_coasting)
    tracker.config.max_age_coasting = coast_frames
    species_acc = SpeciesAccumulator()
    disp_smooth = DisplaySmoother(
        pos_alpha=float(video_cfg.get("display_pos_alpha", 0.55)),
        size_alpha=float(video_cfg.get("display_size_alpha", 0.20)),
    )

    print(f"[Tracker] IoU={tracker.config.iou_threshold}, "
          f"max_coast={tracker.config.max_age_coasting} frames "
          f"({coast_sec:.1f}s @ {fps:.0f}fps), "
          f"center_dist_factor={tracker.config.center_distance_factor}")

    # ---- output writer ----
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open output video writer: {output_path}")

    print(f"[Output] {output_path}")
    print()

    # ---- class-to-color mapping ----
    color_map: dict[str, Tuple[int, int, int]] = {}
    color_idx = 0

    def get_color(label: str) -> Tuple[int, int, int]:
        nonlocal color_idx
        if label not in color_map:
            color_map[label] = _COLORS[color_idx % len(_COLORS)]
            color_idx += 1
        return color_map[label]

    # ---- process frames ----
    frame_num = 0
    t_start = time.time()
    last_report = 0.0

    # Adaptive font/box size based on resolution
    font_scale = max(0.5, min(2.0, frame_w / 1920.0))
    box_thickness = max(1, int(frame_w / 960))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        # ── Stage 1: full-frame detection ────────────────────────────
        det_results = det_model.predict(
            source=frame,
            imgsz=det_imgsz,
            conf=det_conf,
            iou=det_iou,
            device=device,
            half=half,
            verbose=False,
        )

        raw_detections: List[Detection] = []
        if det_results:
            for box in det_results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf_val = float(box.conf[0].item())
                raw_detections.append(Detection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    label="pollinator",
                    confidence=conf_val,
                ))

        # ── Stage 2: tracker update ──────────────────────────────────
        active_tracks = tracker.update(raw_detections)

        # ── Stage 3: classify actively-detected tracks ─────────────
        #   Only tracks that were matched to a detection THIS frame
        #   (time_since_update == 0) get re-classified.  Coasting tracks
        #   (detector lost the insect briefly) keep their stored label.
        tracks_to_classify: List[Tuple[TrackedObject, np.ndarray]] = []
        coasting_tracks: List[TrackedObject] = []

        for track in active_tracks:
            if track.time_since_update > 0:
                # Coasting — no fresh detection, keep stored label
                coasting_tracks.append(track)
                continue
            tx1, ty1, tx2, ty2 = track.get_display_bbox()
            bw, bh = tx2 - tx1, ty2 - ty1
            px, py = bw * crop_pad, bh * crop_pad
            cx1 = max(0, int(tx1 - px))
            cy1 = max(0, int(ty1 - py))
            cx2 = min(frame_w, int(tx2 + px))
            cy2 = min(frame_h, int(ty2 + py))
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                coasting_tracks.append(track)  # treat as coasting
                continue
            crop_resized = cv2.resize(crop, (cls_imgsz, cls_imgsz),
                                      interpolation=cv2.INTER_LINEAR)
            tracks_to_classify.append((track, crop_resized))

        # Batch classify the freshly-detected tracks
        annotations: List[Tuple[int, int, int, int, str, float]] = []
        if tracks_to_classify:
            crops = [c for _, c in tracks_to_classify]
            cls_results = cls_model.predict(
                source=crops,
                imgsz=cls_imgsz,
                device=device,
                half=half,
                verbose=False,
            )

            for (track, _crop), cls_res in zip(tracks_to_classify, cls_results):
                probs = cls_res.probs
                top1_idx = int(probs.top1)
                top1_conf = float(probs.top1conf)
                species = cls_res.names[top1_idx]

                if top1_conf < cls_conf or species == "background":
                    raw_label = "pollinator"
                    raw_conf = 0.0  # low, won't override accumulated evidence
                else:
                    raw_label = species
                    raw_conf = top1_conf

                # Accumulate evidence — species with highest total wins
                stable_label, _ = species_acc.update(
                    track.track_id, raw_label, raw_conf)

                tx1, ty1, tx2, ty2 = track.get_display_bbox()
                sx1, sy1, sx2, sy2 = disp_smooth.smooth(
                    track.track_id, tx1, ty1, tx2, ty2)
                annotations.append((
                    sx1, sy1, sx2, sy2,
                    stable_label, track.smoothed_confidence,
                ))

        # Coasting tracks — reuse accumulated label
        for track in coasting_tracks:
            stored = species_acc.get(track.track_id)
            if stored is None:
                label, _ = "pollinator", 0.0
            else:
                label, _ = stored
            tx1, ty1, tx2, ty2 = track.get_display_bbox()
            sx1, sy1, sx2, sy2 = disp_smooth.smooth(
                track.track_id, tx1, ty1, tx2, ty2)
            annotations.append((
                sx1, sy1, sx2, sy2,
                label, track.smoothed_confidence,
            ))

        # Drop memory for tracks the tracker has removed
        active_ids = {t.track_id for t in active_tracks}
        species_acc.prune(active_ids)
        disp_smooth.prune(active_ids)

        # ── Stage 4: draw ────────────────────────────────────────────
        for bx1, by1, bx2, by2, label, conf_val in annotations:
            if conf_val < det_conf:
                continue  # hide low-confidence tracks
            draw_box(frame, bx1, by1, bx2, by2, label, conf_val,
                     get_color(label), box_thickness, font_scale)

        writer.write(frame)

        # ── progress ─────────────────────────────────────────────────
        now = time.time()
        if now - last_report >= 2.0 or frame_num == total_frames:
            elapsed = now - t_start
            fps_actual = frame_num / max(0.001, elapsed)
            pct = frame_num / max(1, total_frames) * 100.0
            remaining = (total_frames - frame_num) / max(0.1, fps_actual)
            print(f"\r  [{pct:5.1f}%] Frame {frame_num}/{total_frames}  "
                  f"{fps_actual:.1f} fps  "
                  f"ETA {int(remaining // 60)}m {int(remaining % 60)}s  "
                  f"({len(annotations)} tracks)   ", end="", flush=True)
            last_report = now

    # ---- cleanup ----
    cap.release()
    writer.release()
    elapsed_total = time.time() - t_start
    avg_fps = frame_num / max(0.001, elapsed_total)

    print(f"\n\n[Done] {frame_num} frames in "
          f"{int(elapsed_total // 60)}m {int(elapsed_total % 60)}s "
          f"({avg_fps:.1f} avg fps)")
    print(f"[Output] {output_path}")


if __name__ == "__main__":
    main()
