"""
Training script for the Construction Site Safety RT-DETR model.

This reproduces the exact training run used for the submitted weights
(app/models/best.pt). Originally run on Google Colab (free tier, Tesla T4,
16GB VRAM) - training took 3.34 hours for all 60 epochs.

Usage:
    pip install ultralytics roboflow
    python train.py --api-key YOUR_ROBOFLOW_API_KEY
"""

import argparse
from roboflow import Roboflow
from ultralytics import RTDETR


def download_dataset(api_key: str):
    """
    Downloads the exact dataset version used for this project.
    Dataset: Construction Site Safety, Roboflow Universe.
    Version 27 specifically - other versions may have different
    image counts/splits, so this is pinned deliberately.
    """
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("roboflow-universe-projects").project("construction-site-safety")
    dataset = project.version(27).download("yolov8")
    print(f"Dataset downloaded to: {dataset.location}")
    return dataset


def train(dataset_location: str, epochs: int = 60, batch: int = 8, imgsz: int = 640):
    """
    Fine-tunes RT-DETR-l (pretrained on COCO) on the PPE dataset.

    Hyperparameters were chosen as a sensible baseline rather than
    heavily tuned, given the project's time constraint:
      - epochs=60: trained to completion without early stopping
        (patience=15 never triggered)
      - batch=8: fits comfortably in a T4's 16GB VRAM at imgsz=640
      - imgsz=640: RT-DETR's standard input resolution
      - optimizer: left as 'auto' - Ultralytics selected AdamW,
        lr=0.000714, momentum=0.9
      - seed=42: for reproducibility of the specific run reported
        in the memo (exact numeric results may still vary slightly
        run-to-run due to non-deterministic CUDA operations)
    """
    model = RTDETR('rtdetr-l.pt')  # COCO-pretrained checkpoint, fine-tuned from here

    results = model.train(
        data=f'{dataset_location}/data.yaml',
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=15,
        save_period=5,
        seed=42,
        name='rtdetr_ppe_v1',
    )

    print("Training complete.")
    print(f"Best weights saved to: runs/detect/rtdetr_ppe_v1/weights/best.pt")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RT-DETR on the Construction Site Safety dataset")
    parser.add_argument("--api-key", required=True, help="Your Roboflow API key")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    dataset = download_dataset(args.api_key)
    train(dataset.location, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz)
