"""python enclosedTest/train_yolo.py [--resume] [--time-minutes N]"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from pathlib import Path

from common import load_config, resolve_path

# Prevent thread oversubscription in DataLoader workers
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
os.environ.setdefault("OPENCV_OPENCL_DEVICE", "disabled")

STATE_FILE = Path(__file__).parent / "training_state.json"



def save_state(save_dir: Path) -> None:
    weights = save_dir / "weights"
    STATE_FILE.write_text(json.dumps({
        "save_dir": str(save_dir),
        "last_weights": str(weights / "last.pt"),
        "best_weights": str(weights / "best.pt"),
    }, indent=2), encoding="utf-8")


def load_state() -> dict | None:
    if not STATE_FILE.exists():
        return None
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None


def _checkpoint_is_healthy(path: Path) -> bool:
    """Check checkpoint for NaN/Inf weights."""
    import torch

    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"  [Health] Cannot load: {exc}")
        return False

    if not isinstance(ckpt, dict):
        return True

    model_obj = ckpt.get("ema") or ckpt.get("model")
    if model_obj is None:
        print("  [Health] 'model' and 'ema' are both None — empty checkpoint.")
        return False

    if hasattr(model_obj, "float"):
        model_obj = model_obj.float()
    if hasattr(model_obj, "state_dict"):
        state = model_obj.state_dict()
    elif isinstance(model_obj, dict):
        state = model_obj
    else:
        return True

    for name, param in state.items():
        if isinstance(param, torch.Tensor) and param.is_floating_point():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"  [Health] NaN/Inf in '{name}' — corrupted.")
                return False
    return True


def find_resume_weights(runs_dir: Path, run_name: str) -> Path | None:
    """Find the best healthy checkpoint."""
    candidates: list[Path] = []

    # From saved state
    state = load_state()
    if state:
        for key in ("last_weights", "best_weights"):
            p = Path(state.get(key, ""))
            if p.exists() and p not in candidates:
                candidates.append(p)

    # From run directory
    weights_dir = runs_dir / run_name / "weights"
    for name in ("last.pt", "best.pt"):
        p = weights_dir / name
        if p.exists() and p not in candidates:
            candidates.append(p)

    # Epoch checkpoints, newest first
    if weights_dir.exists():
        for p in sorted(weights_dir.glob("epoch*.pt"), key=lambda p: p.stat().st_mtime, reverse=True):
            if p not in candidates:
                candidates.append(p)
        for p in sorted(weights_dir.glob("*.corrupted.pt"), key=lambda p: p.stat().st_mtime, reverse=True):
            if p not in candidates:
                candidates.append(p)

    for path in candidates:
        print(f"[Resume] Checking {path.name} … ", end="")
        if _checkpoint_is_healthy(path):
            print("✓ healthy")
            return path
        print("✗ CORRUPTED — skipping")
        try:
            path.rename(path.with_suffix(".corrupted.pt"))
        except Exception:
            pass

    return None


def preprocess_images(dataset_dir: Path, target_size: int, io_workers: int = 8) -> None:
    """Shrink images to target_size once for faster disk caching."""
    marker = dataset_dir / f".resized_{target_size}"
    if marker.exists():
        print(f"[Preprocess] Images already optimised for {target_size}px — skipping.")
        return
    if not dataset_dir.exists():
        return

    from PIL import Image
    from concurrent.futures import ThreadPoolExecutor
    import time

    exts = {".jpg", ".jpeg", ".png"}
    all_images: list[Path] = []
    for split in ("train", "val", "test"):
        img_dir = dataset_dir / "images" / split
        if img_dir.exists():
            all_images.extend(p for p in img_dir.iterdir() if p.suffix.lower() in exts)

    if not all_images:
        print("[Preprocess] No images found — skipping.")
        return

    print(f"[Preprocess] Optimising {len(all_images)} images (max dim → {target_size}px) …")
    t0 = time.time()

    def resize_one(path: Path) -> bool:
        try:
            with Image.open(path) as im:
                w, h = im.size
                if max(w, h) <= target_size:
                    return False
                scale = target_size / max(w, h)
                im = im.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
                im.save(path, quality=95)
            npy = path.with_suffix(".npy")
            if npy.exists():
                npy.unlink()
            return True
        except Exception:
            return False

    with ThreadPoolExecutor(max_workers=max(io_workers, 4)) as pool:
        results = list(pool.map(resize_one, all_images))

    resized = sum(results)

    if resized > 0:
        for cf in (dataset_dir / "labels").glob("*.cache"):
            cf.unlink()

    marker.write_text(f"resized={resized} total={len(all_images)} target={target_size}")
    print(f"[Preprocess] Done in {time.time() - t0:.0f}s — resized {resized}/{len(all_images)}.")



def main() -> None:
    # Re-exec inside venv if needed
    project_root = Path(__file__).resolve().parents[1]
    venv_python = project_root / ".venv" / "Scripts" / "python.exe"
    if (
        venv_python.exists()
        and sys.prefix == sys.base_prefix
        and os.environ.get("POLLINATOR_REEXEC") != "1"
    ):
        os.environ["POLLINATOR_REEXEC"] = "1"
        os.execv(str(venv_python), [str(venv_python), *sys.argv])

    # Auto-install dependencies
    req = Path(__file__).parent / "requirements.txt"
    if sys.prefix != sys.base_prefix:
        try:
            import torch; from ultralytics import YOLO  # noqa
        except Exception:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req)], check=True)

    import torch
    from ultralytics import YOLO

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from best healthy checkpoint")
    parser.add_argument("--time-minutes", type=int, default=None, help="Training time budget (minutes)")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--imgsz", type=int, default=None, help="Override image size")
    args = parser.parse_args()

    cfg = load_config(Path(__file__).parent / "config.yaml")
    train_cfg = cfg["train"]

    data_yaml = resolve_path(cfg["paths"]["data_yaml"])
    runs_dir = resolve_path(cfg["paths"]["runs_dir"])
    dataset_dir = resolve_path(cfg["paths"]["output_dataset_dir"])

    imgsz = args.imgsz or int(train_cfg["imgsz"])
    batch = args.batch or int(train_cfg["batch"])
    time_minutes = args.time_minutes if args.time_minutes is not None else int(train_cfg["time_minutes"])
    time_hours = max(0.0, time_minutes / 60.0)
    workers = int(train_cfg["workers"])

    preprocess_images(dataset_dir, imgsz, workers)

    if args.resume:
        ckpt = find_resume_weights(runs_dir, str(train_cfg["name"]))
        if not ckpt:
            raise FileNotFoundError("Resume requested but no healthy checkpoint found.")
        print(f"[Resume] Loading weights from: {ckpt}")
        model = YOLO(str(ckpt))
    else:
        model = YOLO(train_cfg["model"])


    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Install CUDA-enabled PyTorch.")

    torch.cuda.empty_cache()
    gc.collect()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    try:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    except Exception:
        pass

    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"[GPU] {torch.cuda.get_device_name(0)} — {gpu_mem:.1f}GB VRAM (no fraction cap)")


    cpu_count = os.cpu_count() or 4
    optimal_workers = min(workers, cpu_count + 4, 24)
    print(f"[Cache] disk cache, {optimal_workers} workers")

    train_kwargs = dict(
        data=str(data_yaml),
        epochs=100_000,           # Effectively unlimited — time stops training
        time=time_hours,
        imgsz=imgsz,
        batch=batch,
        device=int(train_cfg["device"]),
        workers=optimal_workers,
        cache="disk",
        patience=int(train_cfg["patience"]),
        close_mosaic=int(train_cfg["close_mosaic"]),
        cos_lr=bool(train_cfg["cos_lr"]),
        optimizer=str(train_cfg["optimizer"]),
        amp=bool(train_cfg["amp"]),
        project=str(runs_dir),
        name=str(train_cfg["name"]),
        exist_ok=True,
        seed=int(train_cfg.get("seed", 1337)),
        deterministic=bool(train_cfg.get("deterministic", False)),
    )

    # Merge extra_args (augmentations, LR, etc.)
    extra = train_cfg.get("extra_args", {})
    if isinstance(extra, dict):
        train_kwargs.update(extra)


    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"  Model:  {train_cfg['model'] if not args.resume else ckpt}")
    print(f"  Resume: {args.resume}")
    print(f"  Image:  {imgsz}px")
    print(f"  Batch:  {batch}")
    print(f"  Workers:{optimal_workers}")
    print(f"  Cache:  disk")
    print(f"  AMP:    {train_kwargs['amp']}")
    print(f"  LR:     {train_kwargs.get('lr0', '(default)')}")
    print(f"  Time:   {time_hours:.2f}h ({time_minutes} min)")
    print(f"  GPU:    {torch.cuda.get_device_name(0)} ({gpu_mem:.1f}GB)")
    print("=" * 60 + "\n")

    results = model.train(**train_kwargs)
    save_state(Path(results.save_dir))
    print(f"Training complete. Results in {results.save_dir}")


if __name__ == "__main__":
    main()
