# Construction Site Safety Compliance & Reasoning API — Project Memo
# Student Name:Ranjan Kumar G
# Degree/Branch:B.Tech (Artficial Intelligence and Machine Lerning)
# College:Saveetha Engineering College
## Domain and Dataset Selection

I chose construction-site PPE (personal protective equipment) compliance monitoring over my original idea of motorcycle-helmet detection. I compared both before committing: motorcycle-helmet datasets I found on Roboflow topped out around 1,000-1,400 images, often stitched together from the same underlying source with heavy augmentation, with a documented class imbalance between "helmet" and "no-helmet." A hidden evaluation set punishes that kind of dataset, so I moved to PPE detection instead.

I used the "Construction Site Safety" dataset from Roboflow Universe (workspace: `roboflow-universe-projects`, version 27), which gave me 2,603 training images, 114 validation images, and 82 test images across 10 classes: Person, Hardhat, NO-Hardhat, Safety Vest, NO-Safety Vest, Mask, NO-Mask, Safety Cone, machinery, and vehicle. Five of these (Hardhat, NO-Hardhat, Safety Vest, NO-Safety Vest, Safety Cone) are not standard COCO classes, satisfying the non-COCO requirement without relying on a marginal or COCO-adjacent category.

I did not trust the dataset blindly. While hunting for failure cases, I noticed several of the worst-performing test images (filenames like `autox_mp4-XX` and `ka_01181...`) were actually frames from autonomous-racing footage, not construction sites — track-boundary cones from a Formula-Student-style vehicle, not worksite safety cones. This is a real sourcing inconsistency in the dataset, not something I introduced, and it partly explains why "Safety Cone" is my weakest-performing class.

## Split Strategy

I used the dataset's existing train/valid/test split (2,603 / 114 / 82) rather than re-splitting it myself. Re-shuffling a dataset that's already been through multiple community re-exports on Roboflow risks introducing duplicate or near-duplicate images across splits, since I can't fully audit the original collection process. Using the maintainer's split keeps my numbers comparable to what others report on the same dataset version and avoids a leakage risk I couldn't fully rule out myself.

## Evaluation

On my held-out test set (82 images, never used for training or checkpoint selection): **mAP50 = 0.845, mAP50-95 = 0.563, precision = 0.917, recall = 0.802.** This is a small, expected drop from validation (mAP50 0.88) — which is what I want to see. A large drop would suggest a distribution mismatch; no drop (or an increase) would make me suspicious of leakage.

What these numbers tell me: the model is reasonably reliable when it's confident, and precision consistently exceeds recall across almost every class — it leans toward missing things rather than inventing false alarms.

What they don't tell me: they don't show *where* the model fails or whether its mistakes matter for the real use case. Two classes with small test counts (Mask: 16 images, Safety Cone: 8 images) should be read as directional, not precise. More importantly, a specific pattern repeats on both validation and test: **NO-Hardhat recall (0.73) is meaningfully lower than Hardhat recall (0.91)**. The model is more likely to miss a real violation than to miss correct PPE usage — a safety-relevant asymmetry a single overall mAP number completely hides, and the direct justification for why my reasoning layer never lets detector silence imply compliance.

## Failure Case Analysis

Found by scoring every test image on prediction/ground-truth mismatches (missed detections + false positives) and inspecting the worst ones — not by cherry-picking convenient examples.

1. **Dense cone clustering (missed 35, +63 false positives).** A distant row of tiny track-boundary cones in an out-of-domain racing frame. RT-DETR's fixed-query decoder loses precision when many same-class, small objects sit close together — it drops real cones and produces duplicate/overlapping boxes on others simultaneously.
2. **Machinery/vehicle ambiguity + occluded workers (missed 22, +18).** An excavator and truck get double-labeled as both "machinery" and "vehicle" (a real ambiguity in the dataset's own class boundaries), while small workers partially visible near/inside vehicle cabs go undetected — a distinct occlusion problem.
3. **Crowd density (missed 8, +12).** A tightly packed group of ~22 workers shows the same duplicate-box pattern as case 1, this time on people — evidence this is a general model limitation (dense same-class clustering), not a cone-specific quirk.
4. **Pure over-detection, zero misses (+7).** Every real object in this racing-footage frame was found correctly, but the model produced extra duplicate boxes on top of correct ones — redundant localization, distinct from lost detections.
5. **Low-confidence detection through glass (missed 4, +1).** A truck driver visible through a windshield was detected, but at ~0.37 confidence — too weak to count as a solid match. Small, partially occluded, unusual viewing angle relative to most training images.

None of these are catastrophic for the stated use case except case 2 (occluded workers near machinery — a genuine safety-relevant miss). Cases 1 and 4 come from out-of-domain footage a real construction-safety deployment wouldn't encounter.

## Reasoning Layer

The `/ask` endpoint routes every question through a hand-written decision function before deciding whether to call the detector. It first checks for attributes the detector fundamentally can't know (color, age, identity) and unrelated questions (no image-relevant vocabulary at all, e.g. "what is the capital of India") — both skip the detector entirely. For everything else, it classifies the question type (count / presence / absence / comparison / general) and optionally extracts a specific PPE item (helmet, vest, or mask) if the question names one, so "is anyone missing a vest?" and "is anyone missing a helmet?" get genuinely different, item-specific answers instead of the same generic report.

When detection is needed, each detected Person is matched to nearby PPE boxes using body-relative regions (head/torso, scaled to the person's own box size). A PPE claim (compliant or violation) is only made if the best matching detection clears a 0.5 confidence threshold — the same threshold used during evaluation, so live behavior stays consistent with reported metrics. Below that, the item is marked `INSUFFICIENT_INFO` rather than guessed either way.

**Real example of the guardrail working, captured during live testing:** on a test image, the detector produced a `NO-Safety Vest` prediction at 0.35 confidence for one worker — below the 0.5 threshold. The reasoning layer correctly reported "insufficient information" for that worker's vest status rather than declaring a violation, even though the raw signal leaned toward "no vest." This is real output from the running API, not a hypothetical.

**Known limitation:** the reasoning layer answers at the scene level. A question phrased about one specific individual ("is *the* person wearing a helmet") when multiple workers are detected returns an aggregate answer across all workers rather than isolating one — an acknowledged scope decision made under the project's time constraint, not a guessing behavior. The system still never falsely confirms compliance in this case.
