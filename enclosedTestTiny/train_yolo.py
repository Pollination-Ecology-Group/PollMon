"""python enclosedTestTiny/train_yolo.py [--resume] [--time-minutes N]"""
from __future__ import annotations

import argparse
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
    weights_dir = save_dir / "weights"
    state = {
        "save_dir": str(save_dir),
        "last_weights": str(weights_dir / "last.pt"),
        "best_weights": str(weights_dir / "best.pt"),
    }
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def load_state() -> dict | None:
    if not STATE_FILE.exists():
        return None
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None


def find_resume_weights(runs_dir: Path, run_name: str) -> Path | None:
    state = load_state()
    if state:
        last_weights = Path(state.get("last_weights", ""))
        if last_weights.exists():
            return last_weights

    explicit_last = runs_dir / run_name / "weights" / "last.pt"
    if explicit_last.exists():
        return explicit_last

    if runs_dir.exists():
        candidates = sorted(
            (path for path in runs_dir.glob("**/weights/last.pt") if path.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]

    return None


def estimate_dataset_ram(dataset_dir: Path, sample_count: int = 200) -> float:
    """Estimate RAM needed to cache all dataset images (in GB)."""
    from PIL import Image
    import random

    exts = {".jpg", ".jpeg", ".png"}
    all_images: list[Path] = []
    for split in ("train", "val", "test"):
        img_dir = dataset_dir / "images" / split
        if img_dir.exists():
            all_images.extend(p for p in img_dir.iterdir() if p.suffix.lower() in exts)

    if not all_images:
        return 0.0

    random.seed(42)
    sample = random.sample(all_images, min(sample_count, len(all_images)))
    total_bytes = 0
    for p in sample:
        try:
            with Image.open(p) as im:
                w, h = im.size
                total_bytes += w * h * 3
        except Exception:
            pass

    if not sample:
        return 0.0

    avg_bytes = total_bytes / len(sample)
    estimated_gb = avg_bytes * len(all_images) / (1024 ** 3)
    return estimated_gb


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
    elapsed = time.time() - t0

    # Delete stale .cache metadata if images were resized
    if resized > 0:
        labels_dir = dataset_dir / "labels"
        if labels_dir.exists():
            for cf in labels_dir.glob("*.cache"):
                cf.unlink()
                print(f"  Removed stale cache: {cf.name}")

    marker.write_text(f"resized={resized} total={len(all_images)} target={target_size}")
    print(
        f"[Preprocess] Done in {elapsed:.0f}s — "
        f"resized {resized}/{len(all_images)}, "
        f"{len(all_images) - resized} already ≤ {target_size}px."
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    venv_python = project_root / ".venv" / "Scripts" / "python.exe"
    if (
        venv_python.exists()
        and sys.prefix == sys.base_prefix
        and os.environ.get("POLLINATOR_REEXEC") != "1"
    ):
        os.environ["POLLINATOR_REEXEC"] = "1"
        os.execv(str(venv_python), [str(venv_python), *sys.argv])

    requirements_path = Path(__file__).parent / "requirements.txt"

    def ensure_requirements_installed() -> None:
        if sys.prefix == sys.base_prefix:
            return
        try:
            import torch  # noqa: F401
            from ultralytics import YOLO  # noqa: F401
        except Exception:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)],
                check=True,
            )

    ensure_requirements_installed()

    import torch  # noqa: E402
    import psutil  # noqa: E402
    from ultralytics import YOLO  # noqa: E402

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume the last training run")
    parser.add_argument("--time-minutes", type=int, default=None, help="Override training time (minutes)")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--imgsz", type=int, default=None, help="Override image size")
    args = parser.parse_args()

    config = load_config(Path(__file__).parent / "config.yaml")

    data_yaml = resolve_path(config["paths"]["data_yaml"])
    runs_dir = resolve_path(config["paths"]["runs_dir"])

    train_cfg = config["train"]
    model_name = train_cfg["model"]

    # Pre-resize images
    output_dataset_dir = resolve_path(config["paths"]["output_dataset_dir"])
    preprocess_images(
        output_dataset_dir,
        int(args.imgsz or train_cfg["imgsz"]),
        int(train_cfg.get("workers", 8)),
    )

    continue_training = bool(args.resume)
    checkpoint_weights = None
    
    if continue_training:
        checkpoint_weights = find_resume_weights(runs_dir, str(train_cfg["name"]))
        if not checkpoint_weights:
            raise FileNotFoundError(
                "Resume requested but no checkpoint was found. "
                "Expected training_state.json or runs/<name>/weights/last.pt."
            )
        print(f"[Resume] Loading weights from: {checkpoint_weights}")
        model = YOLO(str(checkpoint_weights))
    else:
        try:
            model = YOLO(model_name)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Model weights not found: {model_name}. "
                "Set train.model in config.yaml to a valid Ultralytics weight, "
                "e.g., yolo26n.pt/yolo26s.pt/yolo26m.pt/yolo26l.pt/yolo26x.pt."
            ) from exc

    time_minutes = args.time_minutes if args.time_minutes is not None else int(train_cfg["time_minutes"])
    time_hours = max(0.0, float(time_minutes) / 60.0)

    require_cuda = bool(train_cfg.get("require_cuda", False))
    cuda_device_index = int(train_cfg.get("cuda_device_index", 0))
    matmul_precision = str(train_cfg.get("matmul_precision", "high"))
    allow_tf32 = bool(train_cfg.get("allow_tf32", True))
    max_gpu_mem_fraction = float(train_cfg.get("max_gpu_memory_fraction", 0.90))

    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Install a CUDA-enabled PyTorch build and NVIDIA drivers, "
            "or set train.require_cuda to false and train.device to 'cpu' in config.yaml."
        )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        try:
            torch.set_float32_matmul_precision(matmul_precision)
        except Exception:
            pass
        try:
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            torch.backends.cudnn.allow_tf32 = allow_tf32
        except Exception:
            pass
        # Allow reduced-precision FP16 reductions
        try:
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        except Exception:
            pass
        gpu_mem_total = torch.cuda.get_device_properties(cuda_device_index).total_memory / (1024**3)
        print(f"[GPU] Using full {gpu_mem_total:.1f}GB dedicated VRAM (no fraction cap)")

    device_setting = train_cfg.get("device", 0)
    if isinstance(device_setting, str) and device_setting.lower() in {"auto", "cuda"}:
        device_setting = cuda_device_index if torch.cuda.is_available() else "cpu"
    if str(device_setting) != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Install a CUDA-enabled PyTorch build and NVIDIA drivers, "
            "or set train.device to 'cpu' in config.yaml."
        )

    # RAM-aware cache and worker strategy
    SAFETY_BUFFER_GB = 8.0

    memory = psutil.virtual_memory()
    total_ram_gb = memory.total / (1024 ** 3)
    available_gb = memory.available / (1024 ** 3)
    used_gb = (memory.total - memory.available) / (1024 ** 3)

    print(f"\n[RAM] System: {total_ram_gb:.1f} GB total, {available_gb:.1f} GB available, {used_gb:.1f} GB used")

    cache_setting = train_cfg["cache"]
    worker_setting = int(train_cfg["workers"])
    cpu_count = os.cpu_count() or 4

    if cache_setting == "ram" or cache_setting is True:
        est_cache_gb = estimate_dataset_ram(output_dataset_dir)
        ram_after_cache = available_gb - est_cache_gb - SAFETY_BUFFER_GB

        print(f"[RAM] Estimated dataset cache: {est_cache_gb:.1f} GB")
        print(f"[RAM] After cache + {SAFETY_BUFFER_GB:.0f}GB safety buffer: {ram_after_cache:.1f} GB free")

        if est_cache_gb == 0:
            print("[Cache] No images found to estimate. Using disk cache.")
            cache_setting = "disk"
        elif ram_after_cache >= 0:
            cache_setting = "ram"
            optimal_workers = 0
            print(f"[Cache] ✓ RAM cache fits! Using RAM cache with workers=0")
            print(f"[Cache]   Data already in memory, no workers needed")
        else:
            deficit = abs(ram_after_cache)
            print(f"[Cache] ✗ RAM cache would exceed available memory by {deficit:.1f} GB")
            cache_setting = "disk"
            print(f"[Cache]   Falling back to disk cache with parallel workers")

    if cache_setting == "disk":
        optimal_workers = min(worker_setting, cpu_count + 4, 24)
        print(f"[Cache] Using disk cache with {optimal_workers} workers for parallel I/O")
    elif cache_setting is False or cache_setting == "false":
        optimal_workers = min(worker_setting, cpu_count + 4, 24)
        print(f"[Cache] Caching disabled. Using {optimal_workers} workers for I/O")

    print(f"[Workers] Final: {optimal_workers} data loader workers")

    if args.batch is not None:
        train_cfg["batch"] = int(args.batch)
    if args.imgsz is not None:
        train_cfg["imgsz"] = int(args.imgsz)

    # batch can be int, -1 (auto), or float (fraction)
    batch_val = train_cfg["batch"]
    if isinstance(batch_val, (int, float)):
        batch_val = int(batch_val) if float(batch_val) == int(batch_val) else float(batch_val)
    else:
        batch_val = int(batch_val)

    train_kwargs = dict(
        data=str(data_yaml),
        epochs=int(train_cfg["epochs"]),
        imgsz=int(train_cfg["imgsz"]),
        batch=batch_val,
        device=device_setting,
        workers=optimal_workers,
        cache=cache_setting,
        patience=int(train_cfg["patience"]),
        close_mosaic=int(train_cfg["close_mosaic"]),
        cos_lr=bool(train_cfg["cos_lr"]),
        optimizer=str(train_cfg["optimizer"]),
        amp=bool(train_cfg["amp"]),
        project=str(runs_dir),
        name=str(train_cfg["name"]),
        exist_ok=True,
    )

    if "seed" in train_cfg:
        train_kwargs["seed"] = int(train_cfg["seed"])
    if "deterministic" in train_cfg:
        train_kwargs["deterministic"] = bool(train_cfg["deterministic"])

    extra_args = train_cfg.get("extra_args", {})
    if isinstance(extra_args, dict):
        train_kwargs.update(extra_args)
        
    if time_hours > 0:
        train_kwargs["time"] = time_hours
        train_kwargs["epochs"] = 100000

    # Print training configuration summary
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION SUMMARY")
    print("="*60)
    print(f"  Model: {model_name if not continue_training else checkpoint_weights}")
    print(f"  Resume: {continue_training}")
    print(f"  Image size: {train_kwargs['imgsz']}px")
    print(f"  Batch size: {train_kwargs['batch']}")
    print(f"  Workers: {train_kwargs['workers']}")
    print(f"  Cache: {train_kwargs['cache']}")
    print(f"  Device: {train_kwargs['device']}")
    print(f"  AMP (mixed precision): {train_kwargs['amp']}")
    print(f"  Time limit: {time_hours:.2f}h ({time_minutes} min)")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(device_setting if isinstance(device_setting, int) else 0)
        gpu_mem = torch.cuda.get_device_properties(device_setting if isinstance(device_setting, int) else 0).total_memory / (1024**3)
        print(f"  GPU: {gpu_name} ({gpu_mem:.1f}GB)")
    print("="*60 + "\n")

    print(f"[Training] Starting with epochs={train_kwargs['epochs']}, time={time_hours:.2f}h, batch={train_kwargs['batch']}, imgsz={train_kwargs['imgsz']}")

    results = model.train(**train_kwargs)

    save_dir = Path(results.save_dir)
    save_state(save_dir)

    print(f"Training complete. Results in {save_dir}")


if __name__ == "__main__":
    main()
