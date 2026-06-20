"""YOLOv8 siniflandirma modeli egitimi (Apple Silicon M4 - MPS GPU).

Egitim, data/maras_dataset altindaki train/val klasorlerini kullanir.
"""

import os

from ultralytics import YOLO

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "maras_dataset")


def main():
    # Siniflandirma icin onceden egitilmis YOLOv8m-cls modeli
    model = YOLO("yolov8m-cls.pt")

    model.train(
        data=DATA_DIR,
        epochs=50,
        imgsz=224,
        lr0=0.001,
        optimizer="Adam",
        batch=32,
        workers=4,
        cache=True,
        device="mps",  # Apple Silicon M4 GPU
        name="yolov8_maras_adam_m4",
    )


if __name__ == "__main__":
    main()
