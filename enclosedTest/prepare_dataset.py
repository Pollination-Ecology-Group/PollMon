"""python enclosedTest/prepare_dataset.py"""
from __future__ import annotations

import hashlib
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageEnhance, ImageFilter

from common import ensure_dir, load_config, resolve_path


def read_classes(classes_file: Path) -> List[str]:
    if not classes_file.exists():
        raise FileNotFoundError(f"classes.txt not found at {classes_file}")
    with classes_file.open("r", encoding="utf-8") as file_handle:
        return [line.strip() for line in file_handle if line.strip() and line.strip().upper() != "BACKGROUNDS"]


# Folders that contain negative samples (no objects)
NEGATIVE_SAMPLE_FOLDERS = {"BACKGROUNDS", "backgrounds", "NEGATIVE", "negative", "NO_INSECT", "no_insect"}


def list_images_by_class(
    source_dir: Path, image_extensions: List[str]
) -> Dict[str, List[Path]]:
    result: Dict[str, List[Path]] = {}
    for subdir in sorted(source_dir.iterdir()):
        if not subdir.is_dir():
            continue
        if subdir.name.startswith("."):
            continue
        images: List[Path] = []
        for ext in image_extensions:
            images.extend(sorted(subdir.glob(f"*{ext}")))
        if images:
            result[subdir.name] = images
    return result


def is_negative_sample_folder(folder_name: str) -> bool:
    return folder_name in NEGATIVE_SAMPLE_FOLDERS


def split_items(items: List[Path], train_ratio: float, val_ratio: float) -> Tuple[List[Path], List[Path], List[Path]]:
    total = len(items)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    train_items = items[:train_count]
    val_items = items[train_count:train_count + val_count]
    test_items = items[train_count + val_count:]
    return train_items, val_items, test_items


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.link(src, dst)
        return
    except Exception:
        pass
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def ensure_label_file(image_path: Path, include_unlabeled: bool) -> Path | None:
    label_path = image_path.with_suffix(".txt")
    if label_path.exists():
        return label_path
    if include_unlabeled:
        label_path.write_text("", encoding="utf-8")
        return label_path
    return None


def create_temp_empty_label(image_path: Path) -> Path:
    """Empty label = nothing to detect in this image."""
    label_path = image_path.with_suffix(".txt")
    if not label_path.exists():
        label_path.write_text("", encoding="utf-8")
    return label_path


def rng_for_image(seed: int, image_path: Path, aug_index: int) -> random.Random:
    key = f"{image_path.as_posix()}:{aug_index}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    derived_seed = seed + int(digest[:8], 16)
    return random.Random(derived_seed)


def apply_augmentations(image: Image.Image, rng: random.Random, aug_cfg: dict) -> Image.Image:
    blur_probability = float(aug_cfg.get("blur_probability", 0.0))
    blur_radius_min, blur_radius_max = aug_cfg.get("blur_radius_range", [0.8, 2.0])
    darken_probability = float(aug_cfg.get("darken_probability", 0.0))
    darken_min, darken_max = aug_cfg.get("darken_factor_range", [0.6, 0.9])

    applied_any = False

    if blur_probability > 0 and rng.random() < blur_probability:
        radius = float(rng.uniform(float(blur_radius_min), float(blur_radius_max)))
        image = image.filter(ImageFilter.GaussianBlur(radius))
        applied_any = True

    if darken_probability > 0 and rng.random() < darken_probability:
        factor = float(rng.uniform(float(darken_min), float(darken_max)))
        image = ImageEnhance.Brightness(image).enhance(factor)
        applied_any = True

    if not applied_any:
        # avoid identical duplicates
        image = ImageEnhance.Brightness(image).enhance(0.95)

    return image


def save_augmented_pair(
    image_path: Path,
    label_path: Path,
    dest_image: Path,
    dest_label: Path,
    aug_cfg: dict,
    seed: int,
    aug_index: int,
) -> None:
    if dest_image.exists() and dest_label.exists():
        return

    rng = rng_for_image(seed, image_path, aug_index)
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = apply_augmentations(image, rng, aug_cfg)

        dest_image.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {}
        if dest_image.suffix.lower() in {".jpg", ".jpeg"}:
            save_kwargs = {"quality": 95, "subsampling": 0, "optimize": True}
        image.save(dest_image, **save_kwargs)

    link_or_copy(label_path, dest_label)


def main() -> None:
    config = load_config(Path(__file__).parent / "config.yaml")

    source_dir = resolve_path(config["paths"]["source_images_dir"])
    output_dataset_dir = resolve_path(config["paths"]["output_dataset_dir"])
    data_yaml_path = resolve_path(config["paths"]["data_yaml"])
    classes_file = resolve_path(config["paths"]["classes_file"])

    dataset_config = config["dataset"]
    train_ratio = float(dataset_config["train_split"])
    val_ratio = float(dataset_config["val_split"])
    include_unlabeled = bool(dataset_config["include_unlabeled"])
    image_extensions = [ext.lower() for ext in dataset_config["image_extensions"]]
    seed = int(dataset_config["seed"])
    io_workers = int(dataset_config.get("io_workers", 8))
    aug_cfg = dataset_config.get("augmentations", {})
    augment_enabled = bool(aug_cfg.get("enabled", False))
    aug_count_per_image = int(aug_cfg.get("count_per_image", 0))

    if abs((train_ratio + val_ratio) - 1.0) > 0.0001 and (train_ratio + val_ratio) > 1.0:
        raise ValueError("train_split + val_split must be <= 1.0")

    classes = read_classes(classes_file)

    images_by_class = list_images_by_class(source_dir, image_extensions)
    if not images_by_class:
        raise RuntimeError(f"No images found in {source_dir}")

    random.seed(seed)

    images_train_dir = ensure_dir(output_dataset_dir / "images" / "train")
    images_val_dir = ensure_dir(output_dataset_dir / "images" / "val")
    images_test_dir = ensure_dir(output_dataset_dir / "images" / "test")
    labels_train_dir = ensure_dir(output_dataset_dir / "labels" / "train")
    labels_val_dir = ensure_dir(output_dataset_dir / "labels" / "val")
    labels_test_dir = ensure_dir(output_dataset_dir / "labels" / "test")

    tasks = []

    def enqueue_copy(image_path: Path, label_path: Path, images_dir: Path, labels_dir: Path) -> None:
        tasks.append(("copy", image_path, label_path, images_dir / image_path.name, labels_dir / label_path.name, None))

    def enqueue_augmentations(image_path: Path, label_path: Path, images_dir: Path, labels_dir: Path) -> None:
        if not augment_enabled or aug_count_per_image <= 0:
            return
        for idx in range(aug_count_per_image):
            dest_name = f"{image_path.stem}_aug{idx}{image_path.suffix}"
            dest_label = f"{image_path.stem}_aug{idx}{label_path.suffix}"
            tasks.append((
                "augment",
                image_path,
                label_path,
                images_dir / dest_name,
                labels_dir / dest_label,
                idx,
            ))

    for class_name, image_paths in images_by_class.items():
        shuffled = image_paths[:]
        random.shuffle(shuffled)
        train_items, val_items, test_items = split_items(shuffled, train_ratio, val_ratio)

        is_negative = is_negative_sample_folder(class_name)
        if is_negative:
            print(f"  Processing negative samples from: {class_name} ({len(image_paths)} images)")

        for image_path in train_items:
            if is_negative:
                label_path = create_temp_empty_label(image_path)
            else:
                label_path = ensure_label_file(image_path, include_unlabeled)
            if label_path is None:
                continue
            enqueue_copy(image_path, label_path, images_train_dir, labels_train_dir)
            if not is_negative:
                enqueue_augmentations(image_path, label_path, images_train_dir, labels_train_dir)

        for image_path in val_items:
            if is_negative:
                label_path = create_temp_empty_label(image_path)
            else:
                label_path = ensure_label_file(image_path, include_unlabeled)
            if label_path is None:
                continue
            enqueue_copy(image_path, label_path, images_val_dir, labels_val_dir)
            if not is_negative:
                enqueue_augmentations(image_path, label_path, images_val_dir, labels_val_dir)

        for image_path in test_items:
            if is_negative:
                label_path = create_temp_empty_label(image_path)
            else:
                label_path = ensure_label_file(image_path, include_unlabeled)
            if label_path is None:
                continue
            enqueue_copy(image_path, label_path, images_test_dir, labels_test_dir)
            if not is_negative:
                enqueue_augmentations(image_path, label_path, images_test_dir, labels_test_dir)

    def worker(task):
        task_type, image_path, label_path, dest_image, dest_label, aug_index = task
        if task_type == "copy":
            link_or_copy(image_path, dest_image)
            link_or_copy(label_path, dest_label)
            return
        if task_type == "augment":
            save_augmented_pair(image_path, label_path, dest_image, dest_label, aug_cfg, seed, int(aug_index))

    with ThreadPoolExecutor(max_workers=max(1, io_workers)) as executor:
        futures = [executor.submit(worker, task) for task in tasks]
        for future in as_completed(futures):
            exception = future.exception()
            if exception:
                raise exception

    data_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    data_yaml_content = "\n".join(
        [
            f"path: {output_dataset_dir.as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "", 
            "names:",
        ]
        + [f"  {idx}: {name}" for idx, name in enumerate(classes)]
        + [""]
    )
    data_yaml_path.write_text(data_yaml_content, encoding="utf-8")

    negative_count = sum(
        len(imgs) for folder, imgs in images_by_class.items() 
        if is_negative_sample_folder(folder)
    )

    print("Dataset prepared.")
    print(f"Data YAML: {data_yaml_path}")
    print(f"Classes: {len(classes)}")
    if negative_count > 0:
        print(f"Negative samples (backgrounds): {negative_count}")


if __name__ == "__main__":
    main()
