# Student Name: Ranjan Kumar G
# Degree/Branch: B.Tech Artificial Inteliigence and Machine Learning
# College: Saveetha Engineering College
# Construction Site Safety — Detection & Reasoning API
An RT-DETR object detector fine-tuned on a construction-site PPE (personal protective equipment) dataset, exposed through two FastAPI endpoints: raw detection (`/detect`) and a hand-written natural-language reasoning layer (`/ask`) that reports compliance status and refuses to guess when detection confidence is too low.

See [`MEMO.md`](./MEMO.md) for the full write-up: dataset justification, evaluation analysis, five real failure cases, and reasoning-layer design decisions.

## Project Structure

```
construction-safety-api/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app entrypoint
│   ├── api/
│   │   ├── __init__.py
│   │   └── endpoints.py        # /detect and /ask route definitions
│   ├── models/
│   │   └── best.pt             # trained RT-DETR weights
│   └── services/
│       ├── __init__.py
│       ├── detector.py         # loads model, runs inference
│       └── reasoning.py        # intent routing, PPE matching, confidence guardrail
├── requirements.txt
├── test_reasoning_manual.py    # standalone sanity check for reasoning.py logic
├── MEMO.md
└── README.md
```

## Environment

Training and local inference were done on two different machines — documented honestly below rather than glossed over:

| | Training | Local inference / API |
|---|---|---|
| Platform | Google Colab (free tier), then Kaggle Notebooks for re-evaluation | Local Windows machine |
| Python | 3.13.15 | 3.11.9 (inside `venv`) |
| PyTorch | 2.11.0+cu128 | 2.14.0+cpu |
| GPU | Tesla T4 (16 GB) | None — CPU-only inference |
| Ultralytics | 8.4.145 / 8.4.146 | 8.4.146 |

**Note on CPU inference:** the API runs on CPU locally, so `/detect` and `/ask` take roughly 200-500ms per image instead of the ~50-85ms seen during GPU evaluation. This does not affect correctness, only latency, and is disclosed here rather than hidden.

## Reproducing the Dataset

Dataset: [Construction Site Safety](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety), Roboflow Universe, workspace `roboflow-universe-projects`, **version 27**, exported in YOLOv8 format.

```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("roboflow-universe-projects").project("construction-site-safety")
dataset = project.version(27).download("yolov8")
```

Resulting split: 2,603 train / 114 valid / 82 test images. Classes (10): `Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Person, Safety Cone, Safety Vest, machinery, vehicle`.

## Reproducing Training

```python
from ultralytics import RTDETR

model = RTDETR('rtdetr-l.pt')  # pretrained checkpoint, fine-tuned from here

results = model.train(
    data=f'{dataset.location}/data.yaml',
    epochs=60,
    imgsz=640,
    batch=8,
    patience=15,
    save_period=5,
    seed=42
)
```

- **Optimizer:** auto-selected by Ultralytics → AdamW, lr=0.000714, momentum=0.9
- **Training time:** 3.34 hours for all 60 epochs on a single Tesla T4 (no early stop triggered)
- **Hardware:** Google Colab free tier, Tesla T4, 16 GB VRAM

## Reproducing Evaluation

```python
from ultralytics import RTDETR

model = RTDETR('app/models/best.pt')
test_metrics = model.val(data=f'{dataset.location}/data.yaml', split='test', imgsz=640, batch=8)
```

**Test set results (82 images, held out from training and checkpoint selection):**

| Metric | Value |
|---|---|
| Precision | 0.917 |
| Recall | 0.802 |
| mAP50 | 0.845 |
| mAP50-95 | 0.563 |

Full per-class breakdown and interpretation in [`MEMO.md`](./MEMO.md).

## Running the API

```bash
# from the project root, with venv activated
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Interactive docs: `http://localhost:8000/docs`

### Sample request/response — `POST /detect`

```bash
curl -X POST "http://127.0.0.1:8000/detect" \
  -F "file=@test_image.jpg;type=image/jpeg"
```

```json
{
  "filename": "buildings-15-00354-g001.png",
  "count": 11,
  "detections": [
    {"class": "Safety Vest", "confidence": 0.9046, "bbox": [1295.85, 338.12, 1709.21, 770.97]},
    {"class": "NO-Hardhat", "confidence": 0.8947, "bbox": [1474.04, 214.27, 1648.88, 309.78]},
    {"class": "Safety Vest", "confidence": 0.8804, "bbox": [179.35, 420.13, 303.92, 620.79]},
    {"class": "Person", "confidence": 0.8793, "bbox": [1277.98, 206.89, 1919.06, 813.32]},
    {"class": "Safety Vest", "confidence": 0.8704, "bbox": [725.79, 449.88, 826.65, 574.47]},
    {"class": "NO-Mask", "confidence": 0.8557, "bbox": [1546.00, 330.39, 1643.48, 395.09]},
    {"class": "Person", "confidence": 0.8344, "bbox": [701.77, 380.56, 843.76, 674.51]},
    {"class": "NO-Hardhat", "confidence": 0.8003, "bbox": [233.40, 350.62, 293.95, 378.48]},
    {"class": "Person", "confidence": 0.7903, "bbox": [156.09, 349.48, 312.29, 777.97]},
    {"class": "NO-Mask", "confidence": 0.4982, "bbox": [249.46, 400.24, 277.58, 419.14]},
    {"class": "NO-Mask", "confidence": 0.3917, "bbox": [735.48, 439.46, 751.66, 463.50]}
  ]
}
```

### Sample request/response — `POST /ask`

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -F "file=@test_image.jpg;type=image/jpeg" \
  -F "question=Is anyone not wearing a safety vest?"
```

```json
{
  "question": "Is anyone not wearing a safety vest?",
  "answer": "No confirmed vest violations. 3 worker(s) confirmed compliant, 0 uncertain.",
  "detector_called": true,
  "reasoning_trace": {
    "intent": {"needs_detector": true, "reason": "absence", "target_item": "vest"},
    "total_detections": 11,
    "workers": [
      {"person_confidence": 0.8793, "status": "AT_RISK", "items": {"helmet": {"status": "VIOLATION", "confidence": 0.8947}, "vest": {"status": "COMPLIANT", "confidence": 0.9046}, "mask": {"status": "VIOLATION", "confidence": 0.8557}}}
    ]
  }
}
```

**Example of the confidence guardrail refusing to guess** (same image, different question):

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -F "file=@test_image.jpg;type=image/jpeg" \
  -F "question=Is anyone wearing a mask?"
```

```json
{
  "question": "Is anyone wearing a mask?",
  "answer": "1 of 3 worker(s) are missing a mask. 0 confirmed compliant, 2 uncertain.",
  "detector_called": true
}
```

Here, a `NO-Mask` detection at 0.39 confidence (below the 0.5 threshold) is correctly reported as uncertain rather than a confirmed violation — see `MEMO.md` for the full explanation.

## Test Plan Results

| Test case | Endpoint | Result |
|---|---|---|
| Valid image | `/detect` | ✅ Pass — returns structured detections |
| Invalid file type (.txt) | `/detect` | ✅ Pass — `400`, clear error message |
| Empty request (no file) | `/detect` | ✅ Pass — `422`, FastAPI's built-in required-field validation |
| No detections (blank scene) | `/detect` | ✅ Pass — `200`, `count: 0`, `detections: []` |
| Multiple detections | `/detect` | ✅ Pass — 11 objects returned correctly |
| Count question | `/ask` | ✅ Pass |
| Presence question | `/ask` | ✅ Pass |
| Absence question | `/ask` | ✅ Pass |
| Comparison question | `/ask` | ✅ Pass |
| Unrelated question | `/ask` | ✅ Pass — detector correctly skipped |
| Ambiguous single-entity question | `/ask` | ✅ Pass with documented limitation — see Known Limitations |
| Low-confidence case | `/ask` | ✅ Pass — correctly returns `INSUFFICIENT_INFO`, never guesses |

## Known Limitations

- The reasoning layer answers at the **scene level**. A question about "the person" (singular) when multiple workers are detected returns an aggregate answer across all workers, rather than isolating one specific individual. This is a scope decision made under the project's time constraint — the system still never falsely confirms compliance in this case.
- Intent routing uses keyword matching, not a learned classifier — reliable for the question patterns tested (including all examples given in the assignment brief), but not guaranteed to generalize to arbitrarily-phrased questions outside that pattern set.
- Body-region matching (head/torso zones relative to each person's box) is a heuristic, not learned or measured against ground truth — verified visually against real detections during testing, but not formally evaluated at scale.
- CPU-only local inference (see Environment table above) — functionally correct, slower than the GPU-trained/evaluated numbers would suggest.

## License / Attribution

Dataset: [Construction Site Safety](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety), Roboflow Universe (verify current license terms on the dataset page before any public/commercial use). Base model: RT-DETR via [Ultralytics](https://docs.ultralytics.com/models/rtdetr/).
