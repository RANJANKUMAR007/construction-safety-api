from ultralytics import RTDETR
from PIL import Image
import io
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best.pt")
CONFIDENCE_THRESHOLD = 0.25

model = RTDETR(MODEL_PATH)

def run_detection(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = model.predict(image, conf=CONFIDENCE_THRESHOLD, imgsz=640, verbose=False)
    result = results[0]

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        detections.append({
            "class": model.names[cls_id],
            "confidence": round(float(box.conf[0]), 4),
            "bbox": [round(x, 2) for x in box.xyxy[0].tolist()]
        })
    return detections