#!/usr/bin/env python3
"""
Distribute background/negative images evenly across class folders.

This script:
1. Takes images from a BACKGROUNDS folder
2. Copies them evenly across all class folders in the target directory
3. Creates empty .txt annotation files (no bounding boxes) for each copied image
4. Does NOT touch existing images or annotations

Usage:
    python distribute_backgrounds.py --backgrounds <path> --target <path> [--dry-run]
"""

import argparse
import shutil
from pathlib import Path
from typing import List


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def get_image_files(folder: Path) -> List[Path]:
    """Get all image files in a folder."""
    images = []
    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(file)
    return sorted(images)


def get_class_folders(target_dir: Path) -> List[Path]:
    """Get all class folders (subdirectories) in target directory."""
    folders = []
    for item in sorted(target_dir.iterdir()):
        if item.is_dir() and not item.name.startswith("."):
            # Skip the BACKGROUNDS folder itself if it exists in target
            if item.name.upper() == "BACKGROUNDS":
                continue
            folders.append(item)
    return folders


def distribute_backgrounds(
    backgrounds_dir: Path,
    target_dir: Path,
    dry_run: bool = False,
    prefix: str = "bg_"
) -> None:
    """
    Distribute background images evenly across class folders.
    
    Args:
        backgrounds_dir: Folder containing background images
        target_dir: Target directory with class subfolders
        dry_run: If True, only print what would be done
        prefix: Prefix to add to copied files to identify them as backgrounds
    """
    # Get background images
    bg_images = get_image_files(backgrounds_dir)
    if not bg_images:
        print(f"No images found in {backgrounds_dir}")
        return
    
    print(f"Found {len(bg_images)} background images")
    
    # Get class folders
    class_folders = get_class_folders(target_dir)
    if not class_folders:
        print(f"No class folders found in {target_dir}")
        return
    
    print(f"Found {len(class_folders)} class folders:")
    for folder in class_folders:
        print(f"  - {folder.name}")
    
    # Calculate distribution
    num_classes = len(class_folders)
    images_per_class = len(bg_images) // num_classes
    remainder = len(bg_images) % num_classes
    
    print(f"\nDistribution plan:")
    print(f"  {images_per_class} images per class")
    print(f"  {remainder} extra images distributed to first {remainder} classes")
    
    # Distribute images
    image_index = 0
    total_copied = 0
    
    for i, class_folder in enumerate(class_folders):
        # Calculate how many images this class gets
        count = images_per_class + (1 if i < remainder else 0)
        
        if count == 0:
            continue
            
        print(f"\n{class_folder.name}: {count} images")
        
        for j in range(count):
            if image_index >= len(bg_images):
                break
                
            src_image = bg_images[image_index]
            
            # Create destination filename with prefix to identify as background
            dest_name = f"{prefix}{src_image.name}"
            dest_image = class_folder / dest_name
            dest_label = class_folder / f"{prefix}{src_image.stem}.txt"
            
            # Check if file already exists
            if dest_image.exists():
                print(f"  SKIP (exists): {dest_name}")
                image_index += 1
                continue
            
            if dry_run:
                print(f"  [DRY-RUN] Would copy: {src_image.name} -> {dest_name}")
                print(f"  [DRY-RUN] Would create empty: {dest_label.name}")
            else:
                # Copy image
                shutil.copy2(src_image, dest_image)
                # Create empty annotation file (no bounding boxes)
                dest_label.write_text("", encoding="utf-8")
                print(f"  Copied: {dest_name} + {dest_label.name}")
                total_copied += 1
            
            image_index += 1
    
    print(f"\n{'[DRY-RUN] Would copy' if dry_run else 'Copied'} {total_copied if not dry_run else image_index} background images")
    print("Each background image has an empty .txt annotation (no insects/bounding boxes)")


def main():
    parser = argparse.ArgumentParser(
        description="Distribute background images evenly across class folders"
    )
    parser.add_argument(
        "--backgrounds", "-b",
        type=Path,
        default=Path(r"V:\PollinatorINaturalistData\BACKGROUNDS"),
        help="Path to BACKGROUNDS folder containing negative sample images"
    )
    parser.add_argument(
        "--target", "-t",
        type=Path,
        default=Path(r"V:\PollinatorINaturalistData\extractedImagesIreenTest"),
        help="Target directory with class subfolders"
    )
    parser.add_argument(
        "--prefix", "-p",
        type=str,
        default="bg_",
        help="Prefix for copied background files (default: 'bg_')"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be done without actually copying files"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not args.backgrounds.exists():
        print(f"ERROR: Backgrounds folder not found: {args.backgrounds}")
        return 1
    
    if not args.target.exists():
        print(f"ERROR: Target folder not found: {args.target}")
        return 1
    
    print(f"Backgrounds folder: {args.backgrounds}")
    print(f"Target folder: {args.target}")
    print(f"File prefix: '{args.prefix}'")
    if args.dry_run:
        print("\n*** DRY RUN - No files will be modified ***\n")
    
    distribute_backgrounds(
        args.backgrounds,
        args.target,
        dry_run=args.dry_run,
        prefix=args.prefix
    )
    
    return 0


if __name__ == "__main__":
    exit(main())
