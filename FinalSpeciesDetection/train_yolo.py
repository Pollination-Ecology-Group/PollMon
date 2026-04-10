"""python FinalSpeciesDetection/train_yolo.py [--resume] [--time-minutes N]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time as _time
from pathlib import Path

from common import load_config, resolve_path

STATE_FILE = Path(__file__).parent / "training_state.json"


def save_state(save_dir: Path, epochs_completed: int) -> None:
    data = {"save_dir": str(save_dir), "epochs_completed": epochs_completed}
    STATE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_state() -> dict | None:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return None


def find_resume_weights(runs_dir: Path, run_name: str) -> Path | None:
    last = runs_dir / run_name / "weights" / "last.pt"
    if last.exists():
        return last
    for sub in sorted(runs_dir.iterdir(), reverse=True):
        candidate = sub / "weights" / "last.pt"
        if candidate.exists() and run_name in sub.name:
            return candidate
    return None


class _TimeBudget:
    """Callback-based training time budget with strip_optimizer monkey-patch."""

    def __init__(self, budget_seconds: float, state_file: Path,
                 save_dir_holder: list[Path]) -> None:
        self.budget = budget_seconds
        self._state_file = state_file
        self._save_dir_holder = save_dir_holder
        self._t0 = 0.0
        self._epoch_start = 0.0
        self._durations: list[float] = []
        self._orig_strip = None

    # -- monkey-patch strip_optimizer so last.pt stays valid for resume --
    def _patch_strip_optimizer(self):
        import ultralytics.engine.trainer as trainer_mod
        self._orig_strip = trainer_mod.strip_optimizer

        def _passthrough(f="best.pt", s="", **kwargs):
            import torch
            p = Path(f)
            if p.name == "last.pt":
                return torch.load(f, map_location="cpu")
            if self._orig_strip is not None:
                return self._orig_strip(f, s, **kwargs)
            return torch.load(f, map_location="cpu")

        trainer_mod.strip_optimizer = _passthrough

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
                        help="Resume from last.pt")
    parser.add_argument("--time-minutes", type=int, default=None,
                        help="Training time budget (minutes)")
    parser.add_argument("--batch", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="Override image size")
    args = parser.parse_args()

    config = load_config(Path(__file__).parent / "config.yaml")
    train_cfg = config["train"]

    dataset_dir = resolve_path(config["paths"]["output_dataset_dir"])
    runs_dir    = resolve_path(config["paths"]["runs_dir"])
    model_name  = train_cfg["model"]

    imgsz = args.imgsz or int(train_cfg["imgsz"])
    batch = args.batch or int(train_cfg["batch"])
    time_minutes = (args.time_minutes if args.time_minutes is not None
                    else int(train_cfg["time_minutes"]))
    workers_cfg = int(train_cfg["workers"])

    # -- Model loading --
    if args.resume:
        ckpt = find_resume_weights(runs_dir, str(train_cfg["name"]))
        if not ckpt:
            raise FileNotFoundError(
                "Resume requested but last.pt not found.\n"
                "Start a fresh run first: python FinalSpeciesDetection/train_yolo.py --time-minutes N"
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
    cpu_count = os.cpu_count() or 4
    cache_setting = train_cfg.get("cache", "disk")
    optimal_workers = min(workers_cfg, cpu_count + 4, 24)

    mem = psutil.virtual_memory()
    print(f"\n[RAM] {mem.total / (1024**3):.1f} GB total, "
          f"{mem.available / (1024**3):.1f} GB available")
    print(f"[Cache] {cache_setting}, workers={optimal_workers}")

    # -- Build train kwargs --
    # YOLO classification uses `data=<path>` where <path> has train/ val/ sub-dirs
    train_kwargs = dict(
        data=str(dataset_dir),
        epochs=int(train_cfg["epochs"]),
        imgsz=imgsz,
        batch=batch,
        device=device_setting,
        workers=optimal_workers,
        cache=cache_setting,
        patience=int(train_cfg["patience"]),
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

    if args.resume:
        train_kwargs["resume"] = True

    # -- Register time budget callback --
    save_dir_holder: list[Path] = []
    _TimeBudget(time_minutes * 60, STATE_FILE, save_dir_holder).register(model)

    # -- Summary --
    print("\n" + "=" * 60)
    print("CLASSIFICATION TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"  Model:   {model_name if not args.resume else ckpt}")
    print(f"  Task:    classify")
    print(f"  Resume:  {args.resume}")
    print(f"  Image:   {imgsz}px")
    print(f"  Batch:   {batch}")
    print(f"  Workers: {optimal_workers}")
    print(f"  Cache:   {cache_setting}")
    print(f"  AMP:     {train_kwargs['amp']}")
    print(f"  Time:    {time_minutes} min (epoch-aware budget)")
    print(f"  Data:    {dataset_dir}")
    if torch.cuda.is_available():
        print(f"  GPU:     {torch.cuda.get_device_name(cuda_idx)} ({gpu_mem:.1f}GB)")
    print("=" * 60 + "\n")

    results = model.train(**train_kwargs)

    save_dir = Path(results.save_dir)
    prev = load_state()
    save_state(save_dir, prev.get("epochs_completed", 0) if prev else 0)
    print(f"Training complete. Results in {save_dir}")


if __name__ == "__main__":
    main()
