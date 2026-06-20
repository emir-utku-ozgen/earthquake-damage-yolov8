"""Egitilen en guncel modelle rastgele bir 'Damaged Building' resmi uzerinde inference.

- runs/classify/ altindaki en son degistirilen run otomatik secilir.
- Sonuc cv2 + matplotlib ile gosterilir (hasarli -> kirmizi, saglam -> yesil baslik).
"""

import glob
import os
import random

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs", "classify")
VAL_DAMAGED_DIR = os.path.join(
    PROJECT_ROOT, "data", "maras_dataset", "val", "Damaged Building"
)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def find_latest_model():
    """runs/classify/ altindaki en guncel run'in best.pt (yoksa last.pt) yolunu doner."""
    all_runs = [
        os.path.join(RUNS_DIR, d)
        for d in os.listdir(RUNS_DIR)
        if os.path.isdir(os.path.join(RUNS_DIR, d))
    ]
    if not all_runs:
        raise FileNotFoundError(f"'{RUNS_DIR}' altinda hicbir run bulunamadi.")

    latest_run = max(all_runs, key=os.path.getmtime)

    best_path = os.path.join(latest_run, "weights", "best.pt")
    last_path = os.path.join(latest_run, "weights", "last.pt")
    if os.path.exists(best_path):
        return best_path
    if os.path.exists(last_path):
        return last_path
    raise FileNotFoundError(f"'{latest_run}' icinde model agirligi (best/last.pt) yok.")


def pick_random_image():
    images = [
        os.path.join(VAL_DAMAGED_DIR, f)
        for f in os.listdir(VAL_DAMAGED_DIR)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]
    if not images:
        raise FileNotFoundError(f"'{VAL_DAMAGED_DIR}' icinde resim bulunamadi.")
    return random.choice(images)


def main():
    model_path = find_latest_model()
    print(f"Model yukleniyor: {model_path}")
    model = YOLO(model_path)

    image_path = pick_random_image()
    print(f"Secilen resim: {image_path}")

    results = model(image_path)
    result = results[0]

    # En yuksek olasilikli sinif
    top_idx = int(result.probs.top1)
    confidence = float(result.probs.top1conf)
    label = result.names[top_idx]
    print(f"Tahmin: {label} ({confidence:.2%})")

    # Hasarliysa kirmizi, degilse yesil
    is_damaged = "damaged" in label.lower() and "undamaged" not in label.lower()
    title_color = "red" if is_damaged else "green"

    # cv2 BGR -> RGB
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 8))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.title(f"{label}  ({confidence:.2%})", color=title_color, fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
