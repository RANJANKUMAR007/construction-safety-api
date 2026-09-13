from app.services.reasoning import match_workers_to_ppe, compute_worker_status, classify_intent

fake_detections = [
    {"class": "Person", "confidence": 0.95, "bbox": [100, 50, 250, 500]},
    {"class": "Hardhat", "confidence": 0.88, "bbox": [140, 40, 210, 100]},
    {"class": "Safety Vest", "confidence": 0.79, "bbox": [110, 200, 240, 350]},
    {"class": "Person", "confidence": 0.91, "bbox": [400, 60, 550, 510]},
    {"class": "NO-Hardhat", "confidence": 0.60, "bbox": [430, 50, 500, 110]},
    {"class": "NO-Safety Vest", "confidence": 0.35, "bbox": [410, 210, 540, 360]},
]

workers = match_workers_to_ppe(fake_detections)
for i, w in enumerate(workers):
    status = compute_worker_status(w)
    print(f"Worker {i+1}: {status}")

print()
print(classify_intent("How many workers are wearing helmets?"))
print(classify_intent("What is the capital of India?"))
print(classify_intent("What color is the worker's shirt?"))