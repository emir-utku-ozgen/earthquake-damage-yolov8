"""Veri hazirlama: Kaggle'dan deprem datasetini indirip train/val olarak ayirir.

Dataset: buraktaci/turkiye-earthquake-2023 (kagglehub ile indirilir)
Cikti yapisi:
    data/maras_dataset/train/Damaged Building/...
    data/maras_dataset/train/Undamaged Building/...
    data/maras_dataset/val/Damaged Building/...
    data/maras_dataset/val/Undamaged Building/...
"""

import os
import random
import shutil

import kagglehub

# Tekrar uretilebilirlik icin sabit seed
random.seed(42)

# Calismayi bu dosyaya gore degil, proje koku (parent of src/) baz alarak yap
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "maras_dataset")

TRAIN_RATIO = 0.8
CLASS_NAMES = ["Damaged Building", "Undamaged Building"]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def find_earthquake_dir(base_path):
    """Indirilen dataset altinda 'Earthquake' klasorunu bulur."""
    for root, dirs, _ in os.walk(base_path):
        for d in dirs:
            if d.lower() == "earthquake":
                return os.path.join(root, d)
    # Bulunamazsa indirilen kokun kendisini dene
    return base_path


def find_class_dir(earthquake_dir, class_name):
    """Earthquake klasoru altinda istenen sinif klasorunu (esnek) bulur."""
    target = class_name.lower()
    for root, dirs, _ in os.walk(earthquake_dir):
        for d in dirs:
            if d.lower() == target:
                return os.path.join(root, d)
    return None


def list_images(directory):
    images = []
    for fname in os.listdir(directory):
        if fname.lower().endswith(IMAGE_EXTENSIONS):
            images.append(os.path.join(directory, fname))
    return images


def main():
    print("Dataset indiriliyor (kagglehub)...")
    dataset_path = kagglehub.dataset_download("buraktaci/turkiye-earthquake-2023")
    print(f"Indirilen konum: {dataset_path}")

    earthquake_dir = find_earthquake_dir(dataset_path)
    print(f"Earthquake klasoru: {earthquake_dir}")

    total_copied = 0
    for class_name in CLASS_NAMES:
        class_dir = find_class_dir(earthquake_dir, class_name)
        if class_dir is None:
            print(f"[UYARI] '{class_name}' klasoru bulunamadi, atlaniyor.")
            continue

        images = list_images(class_dir)
        if not images:
            print(f"[UYARI] '{class_name}' icinde resim bulunamadi, atlaniyor.")
            continue

        random.shuffle(images)
        split_idx = int(len(images) * TRAIN_RATIO)
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        for split_name, split_images in (("train", train_images), ("val", val_images)):
            dest_dir = os.path.join(OUTPUT_DIR, split_name, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            for src_path in split_images:
                dest_path = os.path.join(dest_dir, os.path.basename(src_path))
                shutil.copy2(src_path, dest_path)
                total_copied += 1

        print(
            f"'{class_name}': toplam {len(images)} resim -> "
            f"{len(train_images)} train, {len(val_images)} val"
        )

    print(f"\nTamamlandi. Toplam {total_copied} resim kopyalandi.")
    print(f"Cikti dizini: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
