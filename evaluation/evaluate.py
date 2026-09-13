"""
Evaluation script for the Construction Site Safety RT-DETR model.

Runs the trained model against the TEST split (never used for training
or checkpoint selection) to get honest, held-out metrics - these are the
numbers reported in MEMO.md, not the validation-set numbers.

Usage:
    pip install ultralytics roboflow
    python evaluate.py --api-key YOUR_ROBOFLOW_API_KEY --weights app/models/best.pt
"""

import argparse
from roboflow import Roboflow
from ultralytics import RTDETR


def download_dataset(api_key: str):
    """Same dataset/version used for training - must match for a fair evaluation."""
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("roboflow-universe-projects").project("construction-site-safety")
    dataset = project.version(27).download("yolov8")
    return dataset


def evaluate(weights_path: str, dataset_location: str, imgsz: int = 640, batch: int = 8):
    """
    Evaluates on split='test' specifically, not the default validation split.
    Validation was used during training to pick the best checkpoint, so it's
    not a fully unseen set. Test images were never touched by training.
    """
    model = RTDETR(weights_path)

    metrics = model.val(
        data=f'{dataset_location}/data.yaml',
        split='test',
        imgsz=imgsz,
        batch=batch,
    )

    print("\n--- Test Set Results (held-out, never used in training) ---")
    print(f"Precision:   {metrics.box.mp:.3f}")
    print(f"Recall:      {metrics.box.mr:.3f}")
    print(f"mAP50:       {metrics.box.map50:.3f}")
    print(f"mAP50-95:    {metrics.box.map:.3f}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RT-DETR on the held-out test set")
    parser.add_argument("--api-key", required=True, help="Your Roboflow API key")
    parser.add_argument("--weights", default="app/models/best.pt", help="Path to trained weights")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    dataset = download_dataset(args.api_key)
    evaluate(args.weights, dataset.location, imgsz=args.imgsz, batch=args.batch)
