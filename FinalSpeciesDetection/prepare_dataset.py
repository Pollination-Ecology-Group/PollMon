"""python FinalSpeciesDetection/prepare_dataset.py
"""
from __future__ import annotations

import hashlib
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from common import ensure_dir, load_config, resolve_path


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

NEGATIVE_FOLDERS = {"BACKGROUNDS", "backgrounds", "NEGATIVE", "negative",
                    "NO_INSECT", "no_insect"}


def read_classes(classes_file: Path) -> List[str]:
    if not classes_file.exists():
        raise FileNotFoundError(f"classes.txt not found at {classes_file}")
    with classes_file.open("r", encoding="utf-8") as fh:
        return [l.strip() for l in fh if l.strip() and l.strip().upper() != "BACKGROUNDS"]


def list_images_by_class(source_dir: Path,
                         extensions: List[str]) -> Dict[str, List[Path]]:
    result: Dict[str, List[Path]] = {}
    for sub in sorted(source_dir.iterdir()):
        if not sub.is_dir() or sub.name.startswith("."):
            continue
        imgs: List[Path] = []
        for ext in extensions:
            imgs.extend(sorted(sub.glob(f"*{ext}")))
        if imgs:
            result[sub.name] = imgs
    return result


def has_label(image_path: Path) -> bool:
    """Return True when a non-empty YOLO label file exists."""
    label = image_path.with_suffix(".txt")
    if not label.exists():
        return False
    return label.stat().st_size > 0


def read_yolo_boxes(image_path: Path,
                    img_w: int, img_h: int) -> List[Tuple[int, int, int, int]]:
    """Read YOLO-format labels and return absolute (x1, y1, x2, y2) boxes."""
    label = image_path.with_suffix(".txt")
    if not label.exists():
        return []
    boxes: List[Tuple[int, int, int, int]] = []
    for line in label.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        abs_cx, abs_cy = cx * img_w, cy * img_h
        abs_w, abs_h = bw * img_w, bh * img_h
        x1 = max(0, int(abs_cx - abs_w / 2))
        y1 = max(0, int(abs_cy - abs_h / 2))
        x2 = min(img_w, int(abs_cx + abs_w / 2))
        y2 = min(img_h, int(abs_cy + abs_h / 2))
        boxes.append((x1, y1, x2, y2))
    return boxes


def union_box(boxes: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    """Return a single bounding box enclosing all boxes."""
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return x1, y1, x2, y2


def rng_for(seed: int, path: Path, tag: str) -> random.Random:
    key = f"{path.as_posix()}:{tag}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    return random.Random(seed + int(digest[:8], 16))


# ---------------------------------------------------------------------------
# augmentations
# ---------------------------------------------------------------------------

def make_zoom_in(image: Image.Image, boxes: List[Tuple[int, int, int, int]],
                 pad_frac: float, rng: random.Random) -> Image.Image:
    """Crop tightly around the insect (+ padding), making it appear bigger."""
    w, h = image.size
    if not boxes:
        # no bbox -> random centre crop at ~60 % of image
        cw, ch = int(w * 0.6), int(h * 0.6)
        cx, cy = w // 2, h // 2
        x1 = max(0, cx - cw // 2 + rng.randint(-w // 10, w // 10))
        y1 = max(0, cy - ch // 2 + rng.randint(-h // 10, h // 10))
        crop = image.crop((x1, y1, min(w, x1 + cw), min(h, y1 + ch)))
        return crop

    bx1, by1, bx2, by2 = union_box(boxes)
    bw, bh = bx2 - bx1, by2 - by1
    pad_x = int(bw * pad_frac)
    pad_y = int(bh * pad_frac)
    # random jitter in the padding
    jx = rng.randint(-pad_x // 3, pad_x // 3) if pad_x > 3 else 0
    jy = rng.randint(-pad_y // 3, pad_y // 3) if pad_y > 3 else 0
    cx1 = max(0, bx1 - pad_x + jx)
    cy1 = max(0, by1 - pad_y + jy)
    cx2 = min(w, bx2 + pad_x + jx)
    cy2 = min(h, by2 + pad_y + jy)
    crop = image.crop((cx1, cy1, cx2, cy2))
    return crop


def make_zoom_out(image: Image.Image, pad_frac: float,
                  rng: random.Random) -> Image.Image:
    """Pad the image so the insect appears smaller in the final frame."""
    w, h = image.size
    new_w = int(w / max(0.1, (1.0 - pad_frac)))
    new_h = int(h / max(0.1, (1.0 - pad_frac)))
    canvas = Image.new("RGB", (new_w, new_h), (0, 0, 0))
    # random placement within the canvas
    max_ox = new_w - w
    max_oy = new_h - h
    ox = rng.randint(0, max(0, max_ox))
    oy = rng.randint(0, max(0, max_oy))
    canvas.paste(image, (ox, oy))
    return canvas


def apply_color_aug(image: Image.Image, rng: random.Random,
                    cfg: dict) -> Image.Image:
    blur_p = float(cfg.get("blur_probability", 0))
    blur_min, blur_max = cfg.get("blur_radius_range", [0.8, 2.0])
    dark_p = float(cfg.get("darken_probability", 0))
    dark_min, dark_max = cfg.get("darken_factor_range", [0.6, 0.9])
    did = False
    if blur_p > 0 and rng.random() < blur_p:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(blur_min, blur_max)))
        did = True
    if dark_p > 0 and rng.random() < dark_p:
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(dark_min, dark_max))
        did = True
    if not did:
        image = ImageEnhance.Brightness(image).enhance(0.95)
    return image


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def save_image(image: Image.Image, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    kw = {}
    if dst.suffix.lower() in {".jpg", ".jpeg"}:
        kw = {"quality": 95, "subsampling": 0, "optimize": True}
    image.save(dst, **kw)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    config = load_config(Path(__file__).parent / "config.yaml")

    source_dir      = resolve_path(config["paths"]["source_images_dir"])
    backgrounds_dir = resolve_path(config["paths"]["backgrounds_dir"])
    output_dir      = resolve_path(config["paths"]["output_dataset_dir"])
    classes_file    = resolve_path(config["paths"]["classes_file"])

    ds = config["dataset"]
    train_ratio = float(ds["train_split"])
    val_ratio   = float(ds["val_split"])
    test_ratio  = float(ds.get("test_split", 0.0))
    include_unlabeled = bool(ds.get("include_unlabeled", False))
    exts = [e.lower() for e in ds["image_extensions"]]
    seed = int(ds["seed"])
    io_workers = int(ds.get("io_workers", 8))

    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.3f}")

    scale_cfg = ds.get("scale_augmentation", {})
    scale_enabled     = bool(scale_cfg.get("enabled", False))
    zoom_in_count     = int(scale_cfg.get("zoom_in_count", 1))
    zoom_out_count    = int(scale_cfg.get("zoom_out_count", 1))
    zoom_in_pad_frac  = float(scale_cfg.get("zoom_in_crop_fraction", 0.30))
    zoom_out_pad_frac = float(scale_cfg.get("zoom_out_pad_fraction", 0.50))

    color_cfg = ds.get("color_augmentation", {})
    color_enabled = bool(color_cfg.get("enabled", False))

    classes = read_classes(classes_file)
    print(f"Species classes: {len(classes)}")

    images_by_class = list_images_by_class(source_dir, exts)
    if not images_by_class:
        raise RuntimeError(f"No images found in {source_dir}")

    random.seed(seed)

    # ---- collect tasks ----
    # task = (type, src_path, dest_path, extra)
    #   type: "link" | "zoom_in" | "zoom_out" | "color"
    tasks: list = []

    split_names = ["train", "val"]
    if test_ratio > 0:
        split_names.append("test")

    def three_way_split(items: List[Path]) -> Tuple[List[Path], List[Path], List[Path]]:
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_items = items[:n_train]
        val_items = items[n_train:n_train + n_val]
        test_items = items[n_train + n_val:]
        return train_items, val_items, test_items

    totals = {"train": 0, "val": 0, "test": 0}

    for class_name, image_paths in images_by_class.items():
        if class_name.upper() in {n.upper() for n in NEGATIVE_FOLDERS}:
            continue  # backgrounds handled separately

        # only labelled images (non-empty .txt)
        if include_unlabeled:
            labelled = image_paths
        else:
            labelled = [p for p in image_paths if has_label(p)]

        if not labelled:
            print(f"  {class_name}: 0 labelled images -> skipped")
            continue

        shuffled = labelled[:]
        random.shuffle(shuffled)
        train_imgs, val_imgs, test_imgs = three_way_split(shuffled)

        parts = f"train={len(train_imgs)}, val={len(val_imgs)}"
        if test_ratio > 0:
            parts += f", test={len(test_imgs)}"
        print(f"  {class_name}: {len(labelled)} labelled  ({parts})")

        split_lists = [("train", train_imgs), ("val", val_imgs)]
        if test_ratio > 0:
            split_lists.append(("test", test_imgs))

        for split_name, img_list in split_lists:
            dest_cls = output_dir / split_name / class_name
            for img_path in img_list:
                # original
                tasks.append(("link", img_path,
                              dest_cls / img_path.name, None))

                # augmentations only on training set
                if split_name == "train":
                    if scale_enabled:
                        for zi in range(zoom_in_count):
                            tasks.append(("zoom_in", img_path,
                                          dest_cls / f"{img_path.stem}_zin{zi}{img_path.suffix}",
                                          zoom_in_pad_frac))
                        for zo in range(zoom_out_count):
                            tasks.append(("zoom_out", img_path,
                                          dest_cls / f"{img_path.stem}_zout{zo}{img_path.suffix}",
                                          zoom_out_pad_frac))

                    if color_enabled:
                        tasks.append(("color", img_path,
                                      dest_cls / f"{img_path.stem}_caug{img_path.suffix}",
                                      color_cfg))

            totals[split_name] += len(img_list)

    # ---- backgrounds ----
    bg_images: List[Path] = []
    if backgrounds_dir.exists():
        for ext in exts:
            bg_images.extend(sorted(backgrounds_dir.glob(f"*{ext}")))
    if bg_images:
        random.shuffle(bg_images)
        bg_train, bg_val, bg_test = three_way_split(bg_images)
        parts = f"train={len(bg_train)}, val={len(bg_val)}"
        if test_ratio > 0:
            parts += f", test={len(bg_test)}"
        print(f"  background: {len(bg_images)} images  ({parts})")
        bg_split_lists = [("train", bg_train), ("val", bg_val)]
        if test_ratio > 0:
            bg_split_lists.append(("test", bg_test))
        for split_name, img_list in bg_split_lists:
            dest_cls = output_dir / split_name / "background"
            for img_path in img_list:
                tasks.append(("link", img_path,
                              dest_cls / img_path.name, None))
        totals["train"] += len(bg_train)
        totals["val"] += len(bg_val)
        if test_ratio > 0:
            totals["test"] += len(bg_test)
    else:
        print("  WARNING: no background images found")

    parts = ", ".join(f"{k}={v}" for k, v in totals.items() if v > 0)
    print(f"\nTotal originals: {parts}")
    print(f"Total tasks (incl. augmentations): {len(tasks)}")

    # ---- execute tasks ----
    def worker(task):
        task_type, src, dst, extra = task
        if dst.exists():
            return
        if task_type == "link":
            link_or_copy(src, dst)
            return

        with Image.open(src) as im:
            im = im.convert("RGB")
            img_w, img_h = im.size

            if task_type == "zoom_in":
                boxes = read_yolo_boxes(src, img_w, img_h)
                r = rng_for(seed, src, f"zin_{dst.stem}")
                result = make_zoom_in(im, boxes, extra, r)
            elif task_type == "zoom_out":
                r = rng_for(seed, src, f"zout_{dst.stem}")
                result = make_zoom_out(im, extra, r)
            elif task_type == "color":
                r = rng_for(seed, src, f"caug_{dst.stem}")
                result = apply_color_aug(im, r, extra)
            else:
                return

            save_image(result, dst)

    with ThreadPoolExecutor(max_workers=max(1, io_workers)) as pool:
        futures = [pool.submit(worker, t) for t in tasks]
        done = 0
        for f in as_completed(futures):
            exc = f.exception()
            if exc:
                print(f"  ERROR: {exc}")
            done += 1
            if done % 500 == 0:
                print(f"  ... {done}/{len(tasks)} tasks done")

    # ---- summary ----
    for split_name in split_names:
        split_dir = output_dir / split_name
        if split_dir.exists():
            count = sum(1 for _ in split_dir.rglob("*.jpg"))
            count += sum(1 for _ in split_dir.rglob("*.jpeg"))
            count += sum(1 for _ in split_dir.rglob("*.png"))
            n_cls = sum(1 for d in split_dir.iterdir() if d.is_dir())
            print(f"  {split_name}: {count} images across {n_cls} classes")

    print(f"\nDataset ready at: {output_dir}")
    print("Pass this path to train_yolo.py  (YOLO cls reads train/ and val/ sub-dirs automatically)")
    if test_ratio > 0:
        print("Test split available at: " + str(output_dir / "test"))


if __name__ == "__main__":
    main()
