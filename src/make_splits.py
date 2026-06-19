import os
import random
import shutil
from pathlib import Path

random.seed(42)

RAW_ROOT = Path("data/raw/tiny-imagenet-200")
OUT_ROOT = Path("data/splits")

def copy_images_for_split(total_images: int):
    """
    Creates a dataset split with approximately `total_images` images
    sampled uniformly across classes.
    """

    train_dir = RAW_ROOT / "train"
    classes = [d.name for d in train_dir.iterdir() if d.is_dir()]
    classes.sort()

    # All images per class
    class_images = {}
    for cls in classes:
        img_dir = train_dir / cls / "images"
        imgs = list(img_dir.glob("*.JPEG"))
        class_images[cls] = imgs

    # Compute how many images per class
    per_class = total_images // len(classes)
    remainder = total_images % len(classes)

    split_dir = OUT_ROOT / str(total_images)
    if split_dir.exists():
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    # Copy sampled images
    for i, cls in enumerate(classes):
        n = per_class + (1 if i < remainder else 0)
        imgs = class_images[cls]
        sampled = random.sample(imgs, min(n, len(imgs)))

        out_cls_dir = split_dir / "train" / cls
        out_cls_dir.mkdir(parents=True, exist_ok=True)

        for img_path in sampled:
            shutil.copy(img_path, out_cls_dir / img_path.name)

    # Copy validation FULL (fixed for fairness)
    val_src = RAW_ROOT / "val"
    val_out = split_dir / "val"
    shutil.copytree(val_src, val_out)

    print(f"Split created: {split_dir}")

if __name__ == "__main__":
    for size in [1000, 5000, 10000, 25000]:
        copy_images_for_split(size)
