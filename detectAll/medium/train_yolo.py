"""python detectAll/medium/train_yolo.py [--resume] [--time-minutes N]"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time as _time
from pathlib import Path

from common import load_config, resolve_path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
os.environ.setdefault("OPENCV_OPENCL_DEVICE", "disabled")

STATE_FILE = Path(__file__).parent / "training_state.json"


def save_state(save_dir: Path, epochs_completed: int = 0) -> None:
    weights_dir = save_dir / "weights"
    state = {
        "save_dir": str(save_dir),
        "last_weights": str(weights_dir / "last.pt"),
        "best_weights": str(weights_dir / "best.pt"),
        "epochs_completed": epochs_completed,
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
    """Find last.pt for true resume (preserves optimizer/scheduler/epoch)."""
    state = load_state()
    if state:
        last_w = Path(state.get("last_weights", ""))
        if last_w.exists():
            return last_w

    last_pt = runs_dir / run_name / "weights" / "last.pt"
    if last_pt.exists():
        return last_pt

    return None


class _TimeBudget:
    """Stop training cleanly between epochs based on a wall-clock budget.

    Uses trainer.stop = True so ultralytics finishes the current epoch
    (train + val + save) and writes last.pt.

    To preserve resume ability, monkey-patches strip_optimizer so it
    skips last.pt (keeping full optimizer/scheduler/epoch state).
    best.pt is still stripped normally for deployment.
    """

    def __init__(self, budget_seconds: float, state_file: Path, save_dir_holder: list):
        self.budget = budget_seconds
        self._t0: float = 0.0
        self._epoch_start: float = 0.0
        self._durations: list[float] = []
        self._state_file = state_file
        self._save_dir_holder = save_dir_holder
        self._orig_strip = None

    def _patch_strip_optimizer(self):
        """Replace strip_optimizer in the trainer module so last.pt keeps
        its optimizer state and epoch number for resume."""
        import ultralytics.engine.trainer as trainer_mod

        self._orig_strip = getattr(trainer_mod, "strip_optimizer", None)
        if self._orig_strip is None:
            return

        orig = self._orig_strip

        def _resume_safe_strip(f="best.pt", s="", updates=None):
            if Path(f).name == "last.pt":
                print(f"[TimeBudget] Preserving {Path(f).name} for resume (skip strip)")
                # Return the checkpoint dict (ultralytics uses it downstream)
                import torch as _torch
                x = _torch.load(f, map_location="cpu")
                if updates:
                    x.update(updates)
                return x
            return orig(f, s, updates) if updates is not None else orig(f, s)

        trainer_mod.strip_optimizer = _resume_safe_strip

    def _restore_strip_optimizer(self):
        if self._orig_strip is not None:
            import ultralytics.engine.trainer as trainer_mod
            trainer_mod.strip_optimizer = self._orig_strip

    def on_train_start(self, trainer):
        self._t0 = _time.monotonic()
        self._patch_strip_optimizer()
        print(f"[TimeBudget] {self.budget / 60:.0f} min budget started.")

    def on_train_epoch_start(self, trainer):
        self._epoch_start = _time.monotonic()
        if not self._durations:
            return
        elapsed = self._epoch_start - self._t0
        remaining = self.budget - elapsed
        avg_epoch = sum(self._durations) / len(self._durations)
        if remaining < avg_epoch:
            trainer.stop = True
            print(f"\n[TimeBudget] {remaining:.0f}s left < {avg_epoch:.0f}s avg/epoch "
                  f"-> epoch {trainer.epoch + 1} will be the last")

    def on_train_epoch_end(self, trainer):
        self._durations.append(_time.monotonic() - self._epoch_start)
        if hasattr(trainer, "save_dir"):
            self._save_dir_holder.clear()
            self._save_dir_holder.append(Path(trainer.save_dir))
            save_state(Path(trainer.save_dir), trainer.epoch + 1)

    def on_train_end(self, trainer):
        self._restore_strip_optimizer()
        total = _time.monotonic() - self._t0
        epochs = trainer.epoch + 1 if hasattr(trainer, "epoch") else len(self._durations)
        print(f"[TimeBudget] Finished {epochs} epochs in {total / 60:.1f} min "
              f"(budget: {self.budget / 60:.0f} min)")

    def register(self, model):
        model.add_callback("on_train_start", self.on_train_start)
        model.add_callback("on_train_epoch_start", self.on_train_epoch_start)
        model.add_callback("on_train_epoch_end", self.on_train_epoch_end)
        model.add_callback("on_train_end", self.on_train_end)


def estimate_dataset_ram(dataset_dir: Path, sample_count: int = 200) -> float:
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
    return avg_bytes * len(all_images) / (1024 ** 3)


def preprocess_images(dataset_dir: Path, target_size: int, io_workers: int = 8) -> None:
    marker = dataset_dir / f".resized_{target_size}"
    if marker.exists():
        print(f"[Preprocess] Images already optimised for {target_size}px -- skipping.")
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
        print("[Preprocess] No images found -- skipping.")
        return

    print(f"[Preprocess] Optimising {len(all_images)} images (max dim -> {target_size}px) ...")
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

    if resized > 0:
        labels_dir = dataset_dir / "labels"
        if labels_dir.exists():
            for cf in labels_dir.glob("*.cache"):
                cf.unlink()
                print(f"  Removed stale cache: {cf.name}")

    marker.write_text(f"resized={resized} total={len(all_images)} target={target_size}")
    print(
        f"[Preprocess] Done in {elapsed:.0f}s -- "
        f"resized {resized}/{len(all_images)}, "
        f"{len(all_images) - resized} already <= {target_size}px."
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    venv_python = project_root / ".venv" / "Scripts" / "python.exe"
    if (
        venv_python.exists()
        and sys.prefix == sys.base_prefix
        and os.environ.get("POLLINATOR_REEXEC") != "1"
    ):
        os.environ["POLLINATOR_REEXEC"] = "1"
        os.execv(str(venv_python), [str(venv_python), *sys.argv])

    requirements_path = Path(__file__).parent / "requirements.txt"
    if sys.prefix != sys.base_prefix:
        try:
            import torch; from ultralytics import YOLO  # noqa
        except Exception:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)],
                check=True,
            )

    import torch
    import psutil
    from ultralytics import YOLO

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last.pt (preserves optimizer, scheduler, epoch)")
    parser.add_argument("--time-minutes", type=int, default=None,
                        help="Training time budget (minutes)")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--imgsz", type=int, default=None, help="Override image size")
    args = parser.parse_args()

    config = load_config(Path(__file__).parent / "config.yaml")
    train_cfg = config["train"]

    data_yaml = resolve_path(config["paths"]["data_yaml"])
    runs_dir = resolve_path(config["paths"]["runs_dir"])
    output_dataset_dir = resolve_path(config["paths"]["output_dataset_dir"])
    model_name = train_cfg["model"]

    imgsz = args.imgsz or int(train_cfg["imgsz"])
    batch = args.batch or int(train_cfg["batch"])
    time_minutes = args.time_minutes if args.time_minutes is not None else int(train_cfg["time_minutes"])
    workers_cfg = int(train_cfg["workers"])

    preprocess_images(output_dataset_dir, imgsz, workers_cfg)

    # -- Model loading --
    if args.resume:
        ckpt = find_resume_weights(runs_dir, str(train_cfg["name"]))
        if not ckpt:
            raise FileNotFoundError(
                "Resume requested but last.pt not found.\n"
                "Resume only works after a time-budgeted run that was stopped by _TimeBudget.\n"
                "Start a fresh run first: python detectAll/medium/train_yolo.py --time-minutes N"
            )
        print(f"[Resume] Found checkpoint: {ckpt}")
        prev_state = load_state()
        if prev_state:
            print(f"[Resume] Previously completed {prev_state.get('epochs_completed', '?')} epochs")
        model = YOLO(str(ckpt))
    else:
        model = YOLO(model_name)

    # -- CUDA setup --
    require_cuda = bool(train_cfg.get("require_cuda", False))
    cuda_idx = int(train_cfg.get("cuda_device_index", 0))

    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc; gc.collect()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        try:
            torch.set_float32_matmul_precision(str(train_cfg.get("matmul_precision", "high")))
            torch.backends.cuda.matmul.allow_tf32 = bool(train_cfg.get("allow_tf32", True))
            torch.backends.cudnn.allow_tf32 = bool(train_cfg.get("allow_tf32", True))
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        except Exception:
            pass
        gpu_mem = torch.cuda.get_device_properties(cuda_idx).total_memory / (1024**3)
        print(f"[GPU] {torch.cuda.get_device_name(cuda_idx)} -- {gpu_mem:.1f}GB VRAM")

    device_setting = train_cfg.get("device", 0)
    if isinstance(device_setting, str) and device_setting.lower() in {"auto", "cuda"}:
        device_setting = cuda_idx if torch.cuda.is_available() else "cpu"

    # -- Cache strategy --
    SAFETY_BUFFER_GB = 8.0
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024 ** 3)
    print(f"\n[RAM] {mem.total / (1024**3):.1f} GB total, {available_gb:.1f} GB available")

    cache_setting = train_cfg["cache"]
    cpu_count = os.cpu_count() or 4

    if cache_setting == "ram" or cache_setting is True:
        est = estimate_dataset_ram(output_dataset_dir)
        headroom = available_gb - est - SAFETY_BUFFER_GB
        print(f"[RAM] Dataset estimate: {est:.1f} GB, headroom after buffer: {headroom:.1f} GB")
        if est > 0 and headroom >= 0:
            cache_setting = "ram"
            optimal_workers = 0
            print("[Cache] Using RAM cache, workers=0")
        else:
            cache_setting = "disk"
            optimal_workers = min(workers_cfg, cpu_count + 4, 24)
            print(f"[Cache] RAM insufficient, using disk cache, workers={optimal_workers}")
    else:
        optimal_workers = min(workers_cfg, cpu_count + 4, 24)
        print(f"[Cache] disk cache, workers={optimal_workers}")

    # -- batch value --
    batch_val = batch
    if isinstance(batch_val, float) and batch_val == int(batch_val):
        batch_val = int(batch_val)

    # -- Build train kwargs --
    # strip_optimizer monkey-patch keeps last.pt valid for resume.
    # Use the configured epoch count so cos_lr decays properly.
    train_kwargs = dict(
        data=str(data_yaml),
        epochs=int(train_cfg["epochs"]),
        imgsz=imgsz,
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

    # On resume, pass resume=True so ultralytics restores optimizer,
    # scheduler, and epoch counter from last.pt.
    if args.resume:
        train_kwargs["resume"] = True

    # -- Register time budget callback --
    save_dir_holder: list[Path] = []
    _TimeBudget(time_minutes * 60, STATE_FILE, save_dir_holder).register(model)

    # -- Summary --
    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"  Model:   {model_name if not args.resume else ckpt}")
    print(f"  Resume:  {args.resume}")
    print(f"  Image:   {imgsz}px")
    print(f"  Batch:   {batch_val}")
    print(f"  Workers: {optimal_workers}")
    print(f"  Cache:   {cache_setting}")
    print(f"  AMP:     {train_kwargs['amp']}")
    print(f"  Time:    {time_minutes} min (epoch-aware budget)")
    if torch.cuda.is_available():
        print(f"  GPU:     {torch.cuda.get_device_name(cuda_idx)} ({gpu_mem:.1f}GB)")
    print("=" * 60 + "\n")

    results = model.train(**train_kwargs)

    save_dir = Path(results.save_dir)
    # Preserve epochs_completed written by the callback (don't overwrite with 0)
    prev = load_state()
    save_state(save_dir, prev.get("epochs_completed", 0) if prev else 0)
    print(f"Training complete. Results in {save_dir}")


if __name__ == "__main__":
    main()
